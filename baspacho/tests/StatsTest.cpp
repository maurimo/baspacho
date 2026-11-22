/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Statistics are opt-in (Solver::enableStats): when disabled, factor must not record anything,
// including the counters and the largest node size; resetStats clears everything
#include <gtest/gtest.h>
#include <vector>
#include "baspacho/baspacho/MatOps.h"
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/SparseStructure.h"
#include "baspacho/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace ::BaSpaCho::testing_utils;
using namespace std;

static void expectNoStats(Solver& solver) {
  SymbolicCtx& sym = solver.internalSymbolicContext();
  EXPECT_EQ(sym.potrfBiggestN, 0);
  EXPECT_EQ(sym.syrkCalls, 0);
  EXPECT_EQ(sym.gemmCalls, 0);
  EXPECT_EQ(sym.potrfStat.numRuns, 0);
  EXPECT_EQ(sym.sygeStat.numRuns, 0);
}

static void testStats(BackendType backend) {
  auto colBlocks = randomCols(80, 0.1, 37);
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose();
  vector<int64_t> paramSize = randomVec(ss.ptrs.size() - 1, 2, 5, 47);
  Settings settings;
  settings.backend = backend;
  settings.findSparseEliminationRanges = false;
  SolverPtr solver = createSolver(settings, paramSize, ss);
  vector<double> data0 = randomData<double>(solver->dataSize(), -1.0, 1.0, 7);
  solver->skel().damp(data0, 0.0, double(solver->order() * 1.3));

  vector<double> data = data0;
  solver->factor(data.data());  // disabled by default
  expectNoStats(*solver);

  solver->enableStats(true);
  data = data0;
  solver->factor(data.data());
  SymbolicCtx& sym = solver->internalSymbolicContext();
  EXPECT_GT(sym.potrfBiggestN, 0);
  EXPECT_GT(sym.syrkCalls + sym.gemmCalls, 0);
  EXPECT_GT(sym.potrfStat.numRuns, 0);

  solver->resetStats();
  expectNoStats(*solver);

  solver->enableStats(false);
  data = data0;
  solver->factor(data.data());
  expectNoStats(*solver);
}

TEST(Stats, OptIn_Ref) { testStats(BackendRef); }

#ifdef BASPACHO_USE_BLAS
TEST(Stats, OptIn_Fast) { testStats(BackendFast); }
#endif
