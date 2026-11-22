/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <set>
#include <vector>
#include "baspacho/baspacho/CoalescedBlockMatrix.h"
#include "baspacho/baspacho/ComputationModel.h"
#include "baspacho/baspacho/EliminationTree.h"
#include "baspacho/baspacho/FactorTimeEstimate.h"
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/Utils.h"
#include "baspacho/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace ::BaSpaCho::testing_utils;
using namespace std;

namespace {

const ComputationModel* models[] = {&ComputationModel::model_OpenBlas_i7_1185g7_nonneg,
                                    &ComputationModel::model_Cuda117_2080Ti_bounded,
                                    &ComputationModel::model_Metal_CI_paravirtual};

struct Problem {
  vector<int64_t> paramSize;
  SparseStructure ss;  // lower half, by rows
};

Problem randomProblem(int64_t n, double fill, int64_t seed) {
  auto cols = randomCols(n, fill, seed);
  Problem p;
  p.ss = columnsToCscStruct(cols).transpose();
  p.paramSize = randomVec(n, 1, 6, seed + 1);
  return p;
}

// 3d grid with 26-neighbors (dense separators: large supernodes)
Problem grid3d(int64_t n, int64_t bs) {
  vector<set<int64_t>> cols(n * n * n);
  auto id = [n](int64_t x, int64_t y, int64_t z) { return (x * n + y) * n + z; };
  for (int64_t x = 0; x < n; x++)
    for (int64_t y = 0; y < n; y++)
      for (int64_t z = 0; z < n; z++) {
        int64_t i = id(x, y, z);
        cols[i].insert(i);
        for (int64_t dx = -1; dx <= 1; dx++)
          for (int64_t dy = -1; dy <= 1; dy++)
            for (int64_t dz = -1; dz <= 1; dz++) {
              int64_t X = x + dx, Y = y + dy, Z = z + dz;
              if (X < 0 || X >= n || Y < 0 || Y >= n || Z < 0 || Z >= n) continue;
              int64_t j = id(X, Y, Z);
              if (j > i) cols[i].insert(j);
            }
      }
  return Problem{vector<int64_t>(n * n * n, bs), columnsToCscStruct(cols).transpose()};
}

// the estimate of the optimizer must be the time of the factor operations (as issued by
// Solver::factor on the resulting skeleton) according to the model
void checkOptimizedCost(const Problem& prob, const ComputationModel& model, bool deep,
                        bool sparseElim) {
  vector<int64_t> perm = prob.ss.fillReducingPermutation();
  vector<int64_t> invPerm = inversePermutation(perm);
  SparseStructure ss = prob.ss.symmetricPermutation(invPerm, false);
  vector<int64_t> paramSize(perm.size());
  for (size_t i = 0; i < perm.size(); i++) {
    paramSize[invPerm[i]] = prob.paramSize[i];
  }
  EliminationTree et(paramSize, ss, &model);
  if (deep) {
    et.buildTree();
    et.mergeMode = EliminationTree::MergeDeep;
    et.deepIterations = 2;
  } else {
    et.buildTreeFast();
    et.mergeMode = EliminationTree::MergeOptimized;
  }
  et.processTree(sparseElim);
  et.computeAggregateStruct();
  CoalescedBlockMatrixSkel skel(et.computeSpanStart(), et.lumpToSpan, et.colStart, et.rowParam);
  FactorTimeEstimate est = estimateFactorTime(skel, et.sparseElimRanges, model);
  ASSERT_GT(et.optimizedCost, 0);
  EXPECT_NEAR(et.optimizedCost, est.supernodal, 1e-9 * est.supernodal);
}

}  // namespace

TEST(MergeOptimizer, CostIsFactorTime) {
  for (const ComputationModel* model : models) {
    for (int i = 0; i < 20; i++) {
      Problem p = randomProblem(60 + 7 * i, 0.03 + 0.005 * (i % 4), 31 + i);
      for (bool deep : {false, true}) {
        for (bool sparseElim : {false, true}) {
          checkOptimizedCost(p, *model, deep, sparseElim);
        }
      }
    }
    for (int64_t bs : {1, 3}) {
      Problem p = grid3d(7, bs);
      for (bool deep : {false, true}) {
        checkOptimizedCost(p, *model, deep, false);
      }
    }
  }
}

// the deep analysis starts (also) from the previous analysis' merges, and only improves
TEST(MergeOptimizer, DeepNotWorseThanExact) {
  for (const ComputationModel* model : models) {
    for (int i = 0; i < 10; i++) {
      Problem p = i < 8 ? randomProblem(80 + 11 * i, 0.04, 77 + i) : grid3d(6 + i - 8, 2);
      Settings settings;
      settings.backend = BackendRef;
      settings.computationModel = model;
      settings.analysis = AnalysisExact;
      double tExact =
          createSolver(settings, p.paramSize, p.ss)->estimatedFactorTime(*model).supernodal;
      settings.analysis = AnalysisDeep;
      double tDeep =
          createSolver(settings, p.paramSize, p.ss)->estimatedFactorTime(*model).supernodal;
      EXPECT_LE(tDeep, tExact * (1 + 1e-9));
    }
  }
}

// the per-op breakdown adds up, and counts one potrf per supernodal lump
TEST(MergeOptimizer, FactorTimeEstimate) {
  Problem p = grid3d(6, 2);
  Settings settings;
  settings.backend = BackendRef;
  auto solver = createSolver(settings, p.paramSize, p.ss);
  FactorTimeEstimate est =
      solver->estimatedFactorTime(ComputationModel::model_Metal_CI_paravirtual);
  double sum = est.opTime[0] + est.opTime[1] + est.opTime[2] + est.opTime[3];
  EXPECT_NEAR(sum, est.supernodal, 1e-12 * sum);
  int64_t denseLumps = solver->skel().numLumps() -
                       (solver->sparseEliminationRanges().empty()
                            ? 0
                            : solver->sparseEliminationRanges().back());
  EXPECT_EQ(est.opCount[0], denseLumps);
  EXPECT_EQ(est.opCount[2], est.opCount[3]);
}
