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
// factorization the operations share command buffers, and the parameters of each model are
// fitted by least squares on the relative errors. The parameters are printed in the format of
// ComputationModel.cpp, followed by the error of the fit.

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>
#include "baspacho/baspacho/ComputationModel.h"
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

struct Samples {
  vector<pair<double, double>> potrf;                 // n, t
  vector<pair<Eigen::Vector2d, double>> trsm, asmbl;  // (n, k) / (br, bc), t
  vector<pair<Eigen::Vector3d, double>> syge;         // (m, n, k), t
};

// factors `prob` on Metal with the stats enabled, collecting the operation timings
void collect(const Problem& prob, Samples& samples, int reps) {
  Settings settings;
  settings.backend = BackendMetal;
  SolverPtr solver = createSolver(settings, prob.paramSize, prob.ss);
  SymbolicCtx& sym = solver->internalSymbolicContext();
  sym.potrfStat.callBack = [&](double t, int, int n) { samples.potrf.emplace_back(n, t); };
  sym.trsmStat.callBack = [&](double t, int, int n, int k) {
    samples.trsm.emplace_back(Eigen::Vector2d(n, k), t);
  };
  sym.sygeStat.callBack = [&](double t, int, int m, int n, int k) {
    samples.syge.emplace_back(Eigen::Vector3d(m, n, k), t);
  };
  sym.asmblStat.callBack = [&](double t, int, int br, int bc) {
    samples.asmbl.emplace_back(Eigen::Vector2d(br, bc), t);
  };
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

// least squares on the relative errors: minimizes sum_i ((f_i . x - t_i) / t_i)^2
template <int N>
Eigen::Matrix<double, N, 1> fitRelative(const vector<Eigen::Matrix<double, N, 1>>& features,
                                        const vector<double>& times) {
  Eigen::MatrixXd A(features.size(), N);
  Eigen::VectorXd b = Eigen::VectorXd::Ones(features.size());
  for (size_t i = 0; i < features.size(); i++) {
    A.row(i) = features[i].transpose() / times[i];
  }
  return A.colPivHouseholderQr().solve(b);
}

double median(vector<double> v) {
  sort(v.begin(), v.end());
  return v.empty() ? 0 : v[v.size() / 2];
}

template <typename Pred>
double medianRelErr(size_t n, Pred&& predAndTime) {
  vector<double> errs;
  for (size_t i = 0; i < n; i++) {
    auto [pred, t] = predAndTime(i);
    errs.push_back(fabs(pred - t) / t);
  }
  return median(errs);
}

void printVec(const char* what, const Eigen::VectorXd& v) {
  printf("    {");
  for (int i = 0; i < v.size(); i++) {
    printf("%s%.18e", i ? ", " : "", v[i]);
  }
  printf("},  // %s\n", what);
}

}  // namespace

TEST(MetalComputationModel, Fit) {
  // fixed cost of a command buffer: potrf of a 1x1 block, alone
  Samples trivial;
  collect(Problem{{1}, lowerCsr(1, {})}, trivial, 30);
  ASSERT_GE(trivial.potrf.size(), 30u);
  vector<double> trivialTimes;
  for (auto& [n, t] : trivial.potrf) trivialTimes.push_back(t);
  double overhead = median(trivialTimes);

  Samples s;
  for (int64_t bs : {1, 2, 3, 6}) {
    collect(grid3d(bs <= 2 ? 12 : 9, bs), s, 1);
  }
  collect(ba(80, 2000, 9), s, 1);
  printf("metal model samples: potrf %zu, trsm %zu, syge %zu, asmbl %zu; command buffer "
         "overhead %.1f us\n",
         s.potrf.size(), s.trsm.size(), s.syge.size(), s.asmbl.size(), overhead * 1e6);
  ASSERT_GT(s.potrf.size(), 50u);
  ASSERT_GT(s.trsm.size(), 50u);
  ASSERT_GT(s.syge.size(), 50u);
  ASSERT_GT(s.asmbl.size(), 50u);

  // operation times without the command buffer overhead (at least 5% of the measured time)
  auto net = [&](double t) { return max(t - overhead, 0.05 * t); };
  vector<Eigen::Vector4d> fp, fa;
  vector<Eigen::Matrix<double, 6, 1>> ft, fs;
  vector<double> tp, tt, ts, ta;
  for (auto& [n, t] : s.potrf) {
    fp.push_back(Eigen::Vector4d(1, n, n * n, n * n * n));
    tp.push_back(net(t));
  }
  for (auto& [nk, t] : s.trsm) {
    double n = nk[0], k = nk[1];
    Eigen::Matrix<double, 6, 1> f;
    f << 1, n, n * n, k, k * n, k * n * n;
    ft.push_back(f);
    tt.push_back(net(t));
  }
  for (auto& [mnk, t] : s.syge) {
    double m = mnk[0], n = mnk[1], k = mnk[2];
    Eigen::Matrix<double, 6, 1> f;
    f << 1, m + n, m * n, k, k * (m + n), k * m * n;
    fs.push_back(f);
    ts.push_back(net(t));
  }
  for (auto& [rc, t] : s.asmbl) {
    double br = rc[0], bc = rc[1];
    fa.push_back(Eigen::Vector4d(1, br, bc, br * bc));
    ta.push_back(net(t));
  }
  Eigen::Vector4d pPotrf = fitRelative<4>(fp, tp), pAsmbl = fitRelative<4>(fa, ta);
  Eigen::Matrix<double, 6, 1> pTrsm = fitRelative<6>(ft, tt), pSyge = fitRelative<6>(fs, ts);
  ComputationModel model(pPotrf, pTrsm, pSyge, pAsmbl);

  printf("metal model (ComputationModel.cpp format):\n");
  printVec("potrf", pPotrf);
  printVec("trsm", pTrsm);
  printVec("syge", pSyge);
  printVec("asmbl", pAsmbl);

  double ePotrf = medianRelErr(fp.size(), [&](size_t i) {
    return make_pair(model.potrfEst(s.potrf[i].first), tp[i]);
  });
  double eTrsm = medianRelErr(ft.size(), [&](size_t i) {
    return make_pair(model.trsmEst(s.trsm[i].first[0], s.trsm[i].first[1]), tt[i]);
  });
  double eSyge = medianRelErr(fs.size(), [&](size_t i) {
    auto& v = s.syge[i].first;
    return make_pair(model.sygeEst(v[0], v[1], v[2]), ts[i]);
  });
  double eAsmbl = medianRelErr(fa.size(), [&](size_t i) {
    return make_pair(model.asmblEst(s.asmbl[i].first[0], s.asmbl[i].first[1]), ta[i]);
  });
  printf("metal model fit, median relative error: potrf %.2f, trsm %.2f, syge %.2f, asmbl %.2f\n",
         ePotrf, eTrsm, eSyge, eAsmbl);

  for (double p : {pPotrf.norm(), pTrsm.norm(), pSyge.norm(), pAsmbl.norm()}) {
    EXPECT_TRUE(std::isfinite(p));
  }
  // loose bounds: the paravirtual GPU of the CI runners is noisy
  EXPECT_LT(ePotrf, 1.0);
  EXPECT_LT(eTrsm, 1.0);
  EXPECT_LT(eSyge, 1.0);
  EXPECT_LT(eAsmbl, 1.0);
}
