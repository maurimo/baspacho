/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Robustness of the Metal backend: bounds of MetalMirror::get, failures reported as exceptions
// (the process survives), independent solvers on several threads with overlapping readbacks
// and destruction, the README example, non positive definite matrices, backend detection.

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <vector>
#include "baspacho/baspacho/MetalDefs.h"
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/SparseStructure.h"
#include "baspacho/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace ::BaSpaCho::testing_utils;
using namespace std;

namespace {

struct TestProblem {
  vector<int64_t> paramSize;
  SparseStructure ss;
};

TestProblem makeProblem(int seed) {
  auto colBlocks = randomCols(120, 0.04, 37 + seed);
  colBlocks = makeIndependentElimSet(colBlocks, 0, 60);
  TestProblem p;
  p.ss = columnsToCscStruct(colBlocks).transpose();
  p.paramSize = randomVec(p.ss.ptrs.size() - 1, 2, 5, 47 + seed);
  return p;
}

// positive definite data for the solver
vector<float> makeData(const Solver& solver, int seed) {
  vector<float> data = randomData<float>(solver.dataSize(), -1.0, 1.0, seed);
  solver.skel().damp(data, 0.0f, float(solver.order() * 1.3));
  return data;
}

// |A x - b| / |b|, A the (permuted) matrix of `data`, in double
double residual(const Solver& solver, const vector<float>& data, const vector<float>& x,
                const vector<float>& b) {
  Eigen::MatrixXd A = solver.skel().densify(data).cast<double>();
  A = Eigen::MatrixXd(A.selfadjointView<Eigen::Lower>());
  Eigen::Map<const Eigen::VectorXf> xv(x.data(), x.size()), bv(b.data(), b.size());
  return (A * xv.cast<double>() - bv.cast<double>()).norm() / bv.cast<double>().norm();
}

}  // namespace

TEST(MetalRobustness, MirrorGetBounds) {
  vector<float> src = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  MetalMirror<float> m(src);
  vector<float> exact(10);
  m.get(exact);
  EXPECT_EQ(exact, src);
  vector<float> shorter(4);
  m.get(shorter);
  EXPECT_EQ(shorter, vector<float>(src.begin(), src.begin() + 4));
  vector<float> larger(11, -1.0f);
  EXPECT_THROW(m.get(larger), std::out_of_range);
  EXPECT_EQ(larger, vector<float>(11, -1.0f));  // untouched
}

TEST(MetalRobustness, AllocationFailureThrows) {
  MetalMirror<float> huge;
  EXPECT_THROW(huge.resizeToAtLeast(size_t(1) << 60), std::runtime_error);
  // the backend is still usable
  MetalMirror<float> ok(vector<float>{1, 2, 3});
  vector<float> back(3);
  ok.get(back);
  EXPECT_EQ(back, (vector<float>{1, 2, 3}));
}

// Independent solvers on several threads, each one factoring and solving repeatedly: the
// readbacks (MetalMirror::get, which flushes the pending work of all the contexts) and the
// destruction of the contexts of different threads overlap
TEST(MetalRobustness, ConcurrentSolvers) {
  const int numThreads = 4, numIters = 6;
  atomic<int> failures{0};
  vector<double> worst(numThreads, 0.0);
  vector<thread> threads;
  for (int t = 0; t < numThreads; t++) {
    threads.emplace_back([&, t] {
      try {
        TestProblem p = makeProblem(t);
        Settings settings;
        settings.backend = BackendMetal;
        for (int it = 0; it < numIters; it++) {
          SolverPtr solver = createSolver(settings, p.paramSize, p.ss);
          vector<float> data = makeData(*solver, 100 * t + it);
          vector<float> b = randomData<float>(solver->order(), -1.0, 1.0, 7 * t + it);
          vector<float> factor(data.size()), x(b.size());
          {
            MetalMirror<float> dataGpu(data), xGpu(b);
            solver->factor(dataGpu.ptr());
            solver->solve(dataGpu.ptr(), xGpu.ptr(), solver->order(), 1);
            xGpu.get(x);
            dataGpu.get(factor);
          }
          worst[t] = std::max(worst[t], residual(*solver, data, x, b));
        }
      } catch (const std::exception& e) {
        fprintf(stderr, "thread %d: %s\n", t, e.what());
        failures++;
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }
  EXPECT_EQ(failures.load(), 0);
  for (int t = 0; t < numThreads; t++) {
    EXPECT_LT(worst[t], 1e-4) << "thread " << t;
  }
}

// The example in README.md
TEST(MetalRobustness, ReadmeExample) {
  TestProblem p = makeProblem(5);
  const vector<int64_t>& paramSize = p.paramSize;
  const SparseStructure& structure = p.ss;

  Settings settings;
  settings.backend = BackendMetal;
  SolverPtr solver = createSolver(settings, paramSize, structure);

  std::vector<float> data(solver->dataSize());  // fill with the matrix (see solver->accessor())
  data = makeData(*solver, 3);
  MetalMirror<float> dataGpu(data);
  solver->factor(dataGpu.ptr());
  std::vector<float> rhs(solver->order());  // fill with the right hand side
  rhs = randomData<float>(solver->order(), -1.0, 1.0, 9);
  const vector<float> b = rhs;
  MetalMirror<float> rhsGpu(rhs);
  solver->solve(dataGpu.ptr(), rhsGpu.ptr(), solver->order(), /* nRHS = */ 1);
  rhsGpu.get(rhs);  // waits for the GPU and copies the solution back

  EXPECT_LT(residual(*solver, data, rhs, b), 1e-4);
}

// A matrix that isn't positive definite gives NaN in the factor (as on the other backends), and
// is not an error
TEST(MetalRobustness, NotPositiveDefiniteGivesNaN) {
  TestProblem p = makeProblem(2);
  Settings settings;
  settings.backend = BackendMetal;
  SolverPtr solver = createSolver(settings, p.paramSize, p.ss);
  const auto& skel = solver->skel();
  for (bool sparseElimLump : {true, false}) {
    vector<float> data = makeData(*solver, 4);
    int64_t lump = sparseElimLump ? 1 : skel.numLumps() - 1;
    data[skel.chainData[skel.chainColPtr[lump]]] = -1000.0f * skel.order();
    MetalMirror<float> dataGpu(data);
    EXPECT_NO_THROW(solver->factor(dataGpu.ptr()));
    dataGpu.get(data);
    EXPECT_FALSE(std::all_of(data.begin(), data.end(), [](float v) { return std::isfinite(v); }))
        << (sparseElimLump ? "sparse elimination lump" : "dense lump");
  }
}

TEST(MetalRobustness, DetectBestBackend) {
  EXPECT_TRUE(metalDeviceAvailable());
#ifndef BASPACHO_USE_CUBLAS
  EXPECT_EQ(detectBestBackend(), BackendMetal);
#endif
}
