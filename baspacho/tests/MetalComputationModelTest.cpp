/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Fits a ComputationModel for the Metal backend on the machine running the test (eg. the
// paravirtualized GPU of the CI runners). With the stats enabled, the Metal backend times each
// factor operation (potrf, trsm, syrk/gemm, assemble) as its GPU execution time, run alone in its
// command buffer. The operations of the factorization of a few problems are collected, the fixed
// cost of a command buffer (measured on trivial operations) is subtracted, since in a normal
// factorization the operations share command buffers, and the model is fitted by
// fitComputationModel (non-negative least squares on the relative errors, see doc/analysis.md).
// The parameters are printed in the format of ComputationModel.cpp, with the errors of the fit.

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>
#include "baspacho/baspacho/ComputationModel.h"
#include "baspacho/baspacho/ComputationModelFit.h"
#include "baspacho/baspacho/MetalDefs.h"
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/SparseStructure.h"

using namespace BaSpaCho;
using namespace std;

namespace {

struct Problem {
  vector<int64_t> paramSize;
  SparseStructure ss;  // lower half by rows, diagonal included
};

SparseStructure lowerCsr(int64_t n, vector<pair<int64_t, int64_t>> pairs) {
  for (auto& p : pairs) {
    if (p.first < p.second) {
      swap(p.first, p.second);
    }
  }
  for (int64_t i = 0; i < n; i++) {
    pairs.emplace_back(i, i);
  }
  sort(pairs.begin(), pairs.end());
  pairs.erase(unique(pairs.begin(), pairs.end()), pairs.end());
  SparseStructure ss;
  ss.ptrs.assign(n + 1, 0);
  for (auto& p : pairs) {
    ss.ptrs[p.first + 1]++;
    ss.inds.push_back(p.second);
  }
  for (int64_t i = 0; i < n; i++) {
    ss.ptrs[i + 1] += ss.ptrs[i];
  }
  return ss;
}

Problem grid3d(int64_t n, int64_t bs) {
  auto id = [n](int64_t x, int64_t y, int64_t z) { return (x * n + y) * n + z; };
  vector<pair<int64_t, int64_t>> pairs;
  for (int64_t x = 0; x < n; x++)
    for (int64_t y = 0; y < n; y++)
      for (int64_t z = 0; z < n; z++)
        for (int64_t dx = 0; dx <= 1; dx++)
          for (int64_t dy = -1; dy <= 1; dy++)
            for (int64_t dz = -1; dz <= 1; dz++) {
              int64_t X = x + dx, Y = y + dy, Z = z + dz;
              if ((dx == 0 && (dy < 0 || (dy == 0 && dz <= 0))) || X >= n || Y < 0 || Y >= n ||
                  Z < 0 || Z >= n) {
                continue;
              }
              pairs.emplace_back(id(X, Y, Z), id(x, y, z));
            }
  return Problem{vector<int64_t>(n * n * n, bs), lowerCsr(n * n * n, pairs)};
}

// points (size 3) seen by 2..6 cameras, then cameras (odometry between consecutive ones)
Problem ba(int64_t nCams, int64_t nPts, int64_t camSize) {
  mt19937 gen(5);
  uniform_int_distribution<int64_t> camDist(0, nCams - 1), obsDist(2, 6);
  vector<pair<int64_t, int64_t>> pairs;
  for (int64_t p = 0; p < nPts; p++) {
    for (int64_t k = obsDist(gen); k > 0; k--) {
      pairs.emplace_back(nPts + camDist(gen), p);
    }
  }
  for (int64_t c = 0; c + 1 < nCams; c++) {
    pairs.emplace_back(nPts + c + 1, nPts + c);
  }
  vector<int64_t> sizes(nPts, 3);
  sizes.resize(nPts + nCams, camSize);
  return Problem{sizes, lowerCsr(nPts + nCams, pairs)};
}

// factors `prob` on Metal with the stats enabled, collecting the operation timings
void collect(const Problem& prob, ComputationModelSamples& samples, int reps,
             AnalysisMode analysis = AnalysisAuto) {
  Settings settings;
  settings.backend = BackendMetal;
  settings.analysis = analysis;
  SolverPtr solver = createSolver(settings, prob.paramSize, prob.ss);
  samples.collectFrom(*solver);
  solver->enableStats(true);
  // positive definite data: random off-diagonal blocks, dominant diagonal
  vector<float> data(solver->dataSize(), 0.0f);
  mt19937 gen(3);
  uniform_real_distribution<float> dist(-1.0f, 1.0f);
  auto acc = solver->accessor();
  int64_t n = prob.paramSize.size();
  for (int64_t i = 0; i < n; i++) {
    for (int64_t q = prob.ss.ptrs[i]; q < prob.ss.ptrs[i + 1]; q++) {
      int64_t j = prob.ss.inds[q];
      if (j != i) {
        auto b = acc.block(data.data(), i, j);
        for (int r = 0; r < b.rows(); r++)
          for (int c = 0; c < b.cols(); c++) b(r, c) = 0.1f * dist(gen);
      }
    }
    auto d = acc.diagBlock(data.data(), i);
    for (int r = 0; r < d.rows(); r++) d(r, r) = 100.0f;
  }
  for (int r = 0; r < reps; r++) {
    MetalMirror<float> dataGpu(data);
    solver->factor(dataGpu.ptr());
  }
}

double median(vector<double> v) {
  sort(v.begin(), v.end());
  return v.empty() ? 0 : v[v.size() / 2];
}

}  // namespace

TEST(MetalComputationModel, Fit) {
  // fixed cost of a command buffer: potrf of a 1x1 block, alone
  ComputationModelSamples trivial;
  collect(Problem{{1}, lowerCsr(1, {})}, trivial, 30);
  ASSERT_GE(trivial.potrf.size(), 30u);
  vector<double> trivialTimes;
  for (auto& p : trivial.potrf) trivialTimes.push_back(p[1]);
  double overhead = median(trivialTimes);

  // small and large operations: grids with both analyses (the exact one makes large nodes),
  // a larger grid (large separators), bundle adjustment
  ComputationModelSamples s;
  for (int64_t bs : {1, 2, 3, 6}) {
    for (AnalysisMode an : {AnalysisFast, AnalysisExact}) {
      collect(grid3d(bs <= 2 ? 12 : 9, bs), s, 1, an);
    }
  }
  collect(grid3d(18, 3), s, 1, AnalysisFast);
  collect(ba(80, 2000, 9), s, 1);
  printf(
      "metal model samples: potrf %zu, trsm %zu, syge %zu, asmbl %zu; command buffer "
      "overhead %.1f us\n",
      s.potrf.size(), s.trsm.size(), s.syge.size(), s.asmbl.size(), overhead * 1e6);
  ASSERT_GT(s.potrf.size(), 50u);
  ASSERT_GT(s.trsm.size(), 50u);
  ASSERT_GT(s.syge.size(), 50u);
  ASSERT_GT(s.asmbl.size(), 50u);

  // operation times without the command buffer overhead (at least 5% of the measured time)
  auto net = [&](double t) { return max(t - overhead, 0.05 * t); };
  for (auto& x : s.potrf) x.back() = net(x.back());
  for (auto& x : s.trsm) x.back() = net(x.back());
  for (auto& x : s.syge) x.back() = net(x.back());
  for (auto& x : s.asmbl) x.back() = net(x.back());

  ComputationModelFitStats stats[4];
  ComputationModel model = fitComputationModel(s, false, 4.0, 4, stats);
  printf("metal model (ComputationModel.cpp format):\n%s\n", formatComputationModel(model).c_str());
  printf("metal model parameters: %s\n", computationModelParams(model).c_str());
  const char* names[4] = {"potrf", "trsm", "syge", "asmbl"};
  for (int i = 0; i < 4; i++) {
    printf(
        "metal model fit %s: %lld samples, %lld kept, median relative error %.2f, "
        "estimated/measured total %.2f\n",
        names[i], (long long)stats[i].samples, (long long)stats[i].kept, stats[i].medianRelErr,
        stats[i].totalRatio);
    // loose bound: the paravirtual GPU of the CI runners is noisy
    EXPECT_LT(stats[i].medianRelErr, 1.0);
  }
  printf(
      "metal model GFlop/s syge(m=n=k): 128 %.1f, 512 %.1f, 2048 %.1f; potrf: 128 %.1f, "
      "512 %.1f, 2048 %.1f\n",
      2 * pow(128, 3) / model.sygeEst(128, 128, 128) * 1e-9,
      2 * pow(512, 3) / model.sygeEst(512, 512, 512) * 1e-9,
      2 * pow(2048, 3) / model.sygeEst(2048, 2048, 2048) * 1e-9,
      pow(128, 3) / 3 / model.potrfEst(128) * 1e-9, pow(512, 3) / 3 / model.potrfEst(512) * 1e-9,
      pow(2048, 3) / 3 / model.potrfEst(2048) * 1e-9);
  for (double p :
       {model.potrfParams.minCoeff(), model.trsmParams.minCoeff(), model.sygeParams.minCoeff(),
        model.asmblParams.minCoeff(), model.asmblElemParam}) {
    EXPECT_TRUE(std::isfinite(p));
    EXPECT_GE(p, 0.0);
  }
}
