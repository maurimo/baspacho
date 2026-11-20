/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// End-to-end tests of the Metal backend through the public API (createSolver with
// BackendMetal): BA-like problems (sparse elimination of points + dense camera part) and
// 3D grids (large supernodes, so that the MetalPerformanceShaders gemm path is used).
// Factor and solve are compared with a double precision dense reference.
// The same checks are run with BackendFast, as baseline for the tolerances.

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <random>
#include <utility>
#include <vector>
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/SparseStructure.h"
#ifdef BASPACHO_USE_METAL
#include "baspacho/baspacho/MetalDefs.h"
#endif

using namespace BaSpaCho;
using namespace std;

namespace {

using MatD = Eigen::MatrixXd;

struct Problem {
  vector<int64_t> paramSize;
  SparseStructure ss;  // lower-half CSR of blocks, diagonal included
  vector<int64_t> elimRanges;
};

SparseStructure lowerCsrFromPairs(int64_t n, vector<pair<int64_t, int64_t>> pairs) {
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

// points (size 3) first, then cameras; each point is seen by 2..6 cameras, and consecutive
// cameras are linked (odometry)
Problem makeBa(int64_t nCams, int64_t nPts, int64_t camSize, int64_t seed) {
  mt19937 gen(seed);
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
  Problem prob;
  prob.paramSize.assign(nPts, 3);
  prob.paramSize.resize(nPts + nCams, camSize);
  prob.ss = lowerCsrFromPairs(nPts + nCams, pairs);
  prob.elimRanges = {0, nPts};
  return prob;
}

// n x n x n grid, 27-point stencil, blocks of size bs
Problem makeGrid3d(int64_t n, int64_t bs) {
  auto id = [n](int64_t x, int64_t y, int64_t z) { return (x * n + y) * n + z; };
  vector<pair<int64_t, int64_t>> pairs;
  for (int64_t x = 0; x < n; x++) {
    for (int64_t y = 0; y < n; y++) {
      for (int64_t z = 0; z < n; z++) {
        for (int64_t dx = -1; dx <= 1; dx++) {
          for (int64_t dy = -1; dy <= 1; dy++) {
            for (int64_t dz = -1; dz <= 1; dz++) {
              int64_t X = x + dx, Y = y + dy, Z = z + dz;
              if (X >= 0 && X < n && Y >= 0 && Y < n && Z >= 0 && Z < n) {
                pairs.emplace_back(id(X, Y, Z), id(x, y, z));
              }
            }
          }
        }
      }
    }
  }
  Problem prob;
  prob.paramSize.assign(n * n * n, bs);
  prob.ss = lowerCsrFromPairs(n * n * n, pairs);
  return prob;
}

// Random SPD matrix with the problem's block structure (fill blocks left to zero), made
// positive definite by moderate diagonal dominance (so that the off-diagonal blocks and
// hence the sparse elimination updates matter)
vector<float> randomSpdData(const Solver& solver, const Problem& prob, int64_t seed) {
  vector<float> data(solver.dataSize(), 0.0f);
  auto acc = solver.accessor();
  mt19937 gen(seed);
  uniform_real_distribution<double> dist(-1.0, 1.0);
  int64_t n = prob.paramSize.size();
  vector<vector<double>> rowAbs(n);
  for (int64_t i = 0; i < n; i++) {
    rowAbs[i].assign(prob.paramSize[i], 0.0);
  }
  for (int64_t i = 0; i < n; i++) {
    for (int64_t k = prob.ss.ptrs[i]; k < prob.ss.ptrs[i + 1]; k++) {
      int64_t j = prob.ss.inds[k];
      if (j == i) {
        continue;
      }
      auto b = acc.block(data.data(), i, j);
      for (int64_t r = 0; r < b.rows(); r++) {
        for (int64_t c = 0; c < b.cols(); c++) {
          double v = dist(gen);
          b(r, c) = v;
          rowAbs[i][r] += fabs(v);
          rowAbs[j][c] += fabs(v);
        }
      }
    }
  }
  for (int64_t i = 0; i < n; i++) {
    auto d = acc.diagBlock(data.data(), i);
    for (int64_t r = 0; r < d.rows(); r++) {
      double offDiag = rowAbs[i][r];
      for (int64_t c = 0; c < r; c++) {
        double v = 0.5 * dist(gen);
        d(r, c) = d(c, r) = v;
        offDiag += fabs(v);
        rowAbs[i][c] += fabs(v);
      }
    }
    for (int64_t r = 0; r < d.rows(); r++) {
      d(r, r) = 1.2 * (rowAbs[i][r] + 0.5 * (d.rows() - 1)) + 1.0;
    }
  }
  return data;
}

// Number of dense products (in factor) large enough to be run with MPS on Metal: one per
// off-diagonal board in each lump column (see Solver::eliminateBoard)
int64_t countLargeProducts(const CoalescedBlockMatrixSkel& skel, int64_t minOps) {
  int64_t count = 0;
  for (int64_t l = 0; l < skel.numLumps(); l++) {
    int64_t chainColBegin = skel.chainColPtr[l];
    int64_t boardColBegin = skel.boardColPtr[l];
    int64_t boardColEnd = skel.boardColPtr[l + 1];
    int64_t lumpSize = skel.lumpStart[l + 1] - skel.lumpStart[l];
    int64_t rowDataEnd1 = skel.boardChainColOrd[boardColEnd - 1];
    for (int64_t b = 1; b + 1 < boardColEnd - boardColBegin; b++) {
      int64_t belowDiag = skel.boardChainColOrd[boardColBegin + b];
      int64_t rowDataEnd0 = skel.boardChainColOrd[boardColBegin + b + 1];
      int64_t rectRowBegin = skel.chainRowsTillEnd[chainColBegin + belowDiag - 1];
      int64_t m = skel.chainRowsTillEnd[chainColBegin + rowDataEnd0 - 1] - rectRowBegin;
      int64_t n = skel.chainRowsTillEnd[chainColBegin + rowDataEnd1 - 1] - rectRowBegin;
      count += (m * n * lumpSize >= minOps);
    }
  }
  return count;
}

struct Errors {
  double factor;    // |L - Lref| / |Lref| (lower triangle)
  double solution;  // |x - xref| / |xref|
  double residual;  // |A x - b| / |b|
  double factorMs;  // time of factor (second run, after warm-up), informative only
  double solveMs;   // time of solve (second run)
};

double msSince(chrono::steady_clock::time_point t0) {
  return chrono::duration<double, milli>(chrono::steady_clock::now() - t0).count();
}

// factor + solve, either on host memory (CPU backends) or on Metal buffers
Errors factorSolveErrors(BackendType backend, const Problem& prob, int nRHS, int64_t seed,
                         SolverPtr* keep = nullptr) {
  Settings settings;
  settings.backend = backend;
  SolverPtr solver = createSolver(settings, prob.paramSize, prob.ss, prob.elimRanges);
  int64_t order = solver->order();

  vector<float> data = randomSpdData(*solver, prob, seed);
  MatD A = solver->skel().densify(data).cast<double>();
  A = MatD(A.selfadjointView<Eigen::Lower>());
  Eigen::LLT<MatD> llt(A);
  EXPECT_EQ(llt.info(), Eigen::Success);
  MatD Lref = llt.matrixL();

  mt19937 gen(seed + 1);
  uniform_real_distribution<float> dist(-1.0f, 1.0f);
  vector<float> rhs(order * nRHS);
  for (auto& v : rhs) {
    v = dist(gen);
  }
  MatD b = Eigen::Map<Eigen::MatrixXf>(rhs.data(), order, nRHS).cast<double>();
  MatD xRef = llt.solve(b);

  // run twice (the first run includes one-time costs such as the creation of the pipelines),
  // errors are computed on the result of the second run, which is timed
  const vector<float> data0 = data;
  vector<float> x;
  double factorMs = 0, solveMs = 0;
  for (int run = 0; run < 2; run++) {
    data = data0;
    x = rhs;
    if (backend == BackendMetal) {
#ifdef BASPACHO_USE_METAL
      MetalMirror<float> dataGpu(data), xGpu(x);
      auto t0 = chrono::steady_clock::now();
      solver->factor(dataGpu.ptr());
      factorMs = msSince(t0);
      t0 = chrono::steady_clock::now();
      solver->solve(dataGpu.ptr(), xGpu.ptr(), order, nRHS);
      solveMs = msSince(t0);
      dataGpu.get(data);
      xGpu.get(x);
#endif
    } else {
      auto t0 = chrono::steady_clock::now();
      solver->factor(data.data());
      factorMs = msSince(t0);
      t0 = chrono::steady_clock::now();
      solver->solve(data.data(), x.data(), order, nRHS);
      solveMs = msSince(t0);
    }
  }

  MatD L = MatD(solver->skel().densify(data).cast<double>()).triangularView<Eigen::Lower>();
  MatD xm = Eigen::Map<Eigen::MatrixXf>(x.data(), order, nRHS).cast<double>();
  Errors e;
  e.factor = (L - Lref).norm() / Lref.norm();
  e.solution = (xm - xRef).norm() / xRef.norm();
  e.residual = (A * xm - b).norm() / b.norm();
  e.factorMs = factorMs;
  e.solveMs = solveMs;
  if (keep) {
    *keep = std::move(solver);
  }
  return e;
}

// float tolerances, relative (CPU float errors are ~1e-7 on these problems)
constexpr double kFactorTol = 2e-5;
constexpr double kSolveTol = 2e-5;

void checkErrors(const Errors& e, const char* what) {
  printf("%s: factor %.2e, solution %.2e, residual %.2e (time: factor %.1f ms, solve %.1f ms)\n",
         what, e.factor, e.solution, e.residual, e.factorMs, e.solveMs);
  EXPECT_LT(e.factor, kFactorTol) << what;
  EXPECT_LT(e.solution, kSolveTol) << what;
  EXPECT_LT(e.residual, kSolveTol) << what;
}

}  // namespace

// baseline: same checks on the CPU backend
TEST(MetalSolver, Baseline_Fast_BA) {
  for (int s = 0; s < 3; s++) {
    checkErrors(factorSolveErrors(BackendFast, makeBa(40, 600, 9, 7 + s), 3, 11 + s), "fast BA");
  }
}

TEST(MetalSolver, Baseline_Fast_Grid3d) {
  checkErrors(factorSolveErrors(BackendFast, makeGrid3d(9, 3), 2, 5), "fast grid3d");
}

#ifdef BASPACHO_USE_METAL

// BA: sparse elimination of points (Metal kernels), dense camera part
TEST(MetalSolver, BA_FactorSolve) {
  for (int s = 0; s < 3; s++) {
    checkErrors(factorSolveErrors(BackendMetal, makeBa(40, 600, 9, 7 + s), 3, 11 + s), "metal BA");
  }
}

// BA with larger cameras: more and larger dense products
TEST(MetalSolver, BA_LargeCameras_FactorSolve) {
  checkErrors(factorSolveErrors(BackendMetal, makeBa(60, 800, 12, 3), 1, 17), "metal BA cam12");
}

// 3D grid: large supernodes, the dense products must go through MPS
TEST(MetalSolver, Grid3d_MpsGemm) {
  int64_t before = metalMpsGemmCount();
  SolverPtr solver;
  checkErrors(factorSolveErrors(BackendMetal, makeGrid3d(9, 3), 2, 5, &solver), "metal grid3d");
  int64_t expected = countLargeProducts(solver->skel(), kMetalMpsGemmMinOps);
  int64_t done = metalMpsGemmCount() - before;  // factorSolveErrors factors twice
  printf("MPS gemm calls: %lld in 2 factorizations (expected %lld each)\n", (long long)done,
         (long long)expected);
  EXPECT_GT(expected, 0) << "test problem too small to exercise the MPS path";
  EXPECT_EQ(done, 2 * expected);
}

// the Metal backend is float-only: double must be rejected with an exception
TEST(MetalSolver, DoubleRejected) {
  Problem prob = makeBa(5, 20, 6, 1);
  Settings settings;
  settings.backend = BackendMetal;
  SolverPtr solver = createSolver(settings, prob.paramSize, prob.ss, prob.elimRanges);
  MetalMirror<double> dataGpu(vector<double>(solver->dataSize(), 0.0));
  EXPECT_THROW(solver->factor(dataGpu.ptr()), std::runtime_error);
}

#endif  // BASPACHO_USE_METAL
