/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Tests of the level scheduling of the Metal solves (MetalSolveLevels.h is plain C++, so these
// tests run on every platform): the levels must respect the dependencies of the solves, and the
// dispatches issued by metalLevelSolve must process every lump once, after the lumps it depends
// on, in a different dispatch (threadgroups of a dispatch run in any order on the GPU).

#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "baspacho/baspacho/CoalescedBlockMatrix.h"
#include "baspacho/baspacho/EliminationTree.h"
#include "baspacho/baspacho/MetalSolveLevels.h"
#include "baspacho/baspacho/SparseStructure.h"
#include "baspacho/baspacho/Utils.h"
#include "baspacho/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace ::BaSpaCho::testing_utils;
using namespace std;

static unique_ptr<CoalescedBlockMatrixSkel> makeSkel(int seed, bool sparseElim) {
  int numParams = sparseElim ? 215 : 60;
  auto colBlocks = randomCols(numParams, sparseElim ? 0.03 : 0.15, 57 + seed);
  if (sparseElim) {
    colBlocks = makeIndependentElimSet(colBlocks, 0, 150);
  }
  SparseStructure sortedSs = columnsToCscStruct(colBlocks).transpose();
  vector<int64_t> paramSize =
      randomVec(sortedSs.ptrs.size() - 1, sparseElim ? 2 : 3, sparseElim ? 3 : 12, 47 + seed);
  EliminationTree et(paramSize, sortedSs);
  et.buildTree();
  et.processTree(sparseElim, {});
  et.computeAggregateStruct();
  return make_unique<CoalescedBlockMatrixSkel>(et.computeSpanStart(), et.lumpToSpan, et.colStart,
                                               et.rowParam);
}

// pairs (t, l), t > l, such that the block of rows of lump t and columns of lump l is not empty
static set<pair<int64_t, int64_t>> blockPairs(const CoalescedBlockMatrixSkel& skel) {
  set<pair<int64_t, int64_t>> retv;
  for (int64_t l = 0; l < skel.numLumps(); l++) {
    for (int64_t c = skel.chainColPtr[l]; c < skel.chainColPtr[l + 1]; c++) {
      int64_t t = skel.spanToLump[skel.chainRowSpan[c]];
      if (t > l) {
        retv.insert({t, l});
      }
    }
  }
  return retv;
}

static void checkRowChains(const CoalescedBlockMatrixSkel& skel) {
  MetalRowChains rc(skel);
  ASSERT_EQ((int64_t)rc.ptr.size(), skel.numSpans() + 1);
  int64_t numBelowDiag = 0;
  for (int64_t l = 0; l < skel.numLumps(); l++) {
    for (int64_t c = skel.chainColPtr[l]; c < skel.chainColPtr[l + 1]; c++) {
      if (skel.spanToLump[skel.chainRowSpan[c]] > l) {
        numBelowDiag++;
      }
    }
  }
  ASSERT_EQ(rc.ptr.back(), numBelowDiag);
  for (int64_t s = 0; s < skel.numSpans(); s++) {
    for (int64_t q = rc.ptr[s]; q < rc.ptr[s + 1]; q++) {
      int64_t c = rc.chain[q], l = rc.lump[q];
      ASSERT_EQ(skel.chainRowSpan[c], s);
      ASSERT_GE(c, skel.chainColPtr[l]);
      ASSERT_LT(c, skel.chainColPtr[l + 1]);
      ASSERT_LT(l, skel.spanToLump[s]);
      if (q > rc.ptr[s]) {
        ASSERT_LT(rc.lump[q - 1], l);  // sorted, one chain per column
      }
    }
  }
}

static void checkLevels(const CoalescedBlockMatrixSkel& skel, int64_t begin, bool transpose) {
  MetalSolveLevels lv = transpose ? MetalSolveLevels::backward(skel, begin)
                                  : MetalSolveLevels::forward(skel, begin);
  ASSERT_EQ((int64_t)lv.lumps.size(), skel.numLumps() - begin);
  map<int64_t, int64_t> levelOf;
  for (int64_t i = 0; i < lv.numLevels(); i++) {
    ASSERT_LT(lv.levelPtr[i], lv.levelPtr[i + 1]);  // no empty levels
    for (int64_t q = lv.levelPtr[i]; q < lv.levelPtr[i + 1]; q++) {
      ASSERT_GE(lv.lumps[q], begin);
      ASSERT_TRUE(levelOf.emplace(lv.lumps[q], i).second);
      if (q > lv.levelPtr[i]) {
        ASSERT_LT(lv.lumps[q - 1], lv.lumps[q]);
      }
    }
  }
  for (auto [t, l] : blockPairs(skel)) {
    if (l < begin) {
      continue;
    }
    if (transpose) {
      ASSERT_LT(levelOf[t], levelOf[l]);  // l needs t
    } else {
      ASSERT_LT(levelOf[l], levelOf[t]);  // t needs l
    }
  }
}

// Runs metalLevelSolve recording the dispatches, and checks their order
static void checkDispatches(const CoalescedBlockMatrixSkel& skel, int64_t begin, int64_t end,
                            bool transpose) {
  MetalSolveLevels lv = transpose ? MetalSolveLevels::backward(skel, begin)
                                  : MetalSolveLevels::forward(skel, begin);
  const int64_t numLumps = skel.numLumps();
  vector<int64_t> solvedAt(numLumps, -1), updatedAt(numLumps, -1);
  int64_t dispatch = 0;
  metalLevelSolve(skel, lv, transpose, begin, end, /* ldc = */ 100, /* nRHS = */ 3,
                  [&](MetalLevelKernel k, const MetalLevelSolveParams& p, int64_t numGroups,
                      int64_t nth) {
                    EXPECT_EQ(p.srcBegin, begin);
                    EXPECT_EQ(p.srcEnd, end);
                    EXPECT_EQ(p.transpose, transpose ? 1 : 0);
                    EXPECT_GE(nth, 32);
                    EXPECT_LE(nth, 256);
                    EXPECT_GT(numGroups, 0);
                    for (int64_t g = 0; g < numGroups; g++) {
                      int64_t l = p.useList ? lv.lumps[p.listOff + g] : p.listOff + g;
                      ASSERT_GE(l, 0);
                      ASSERT_LT(l, numLumps);
                      if (k == MetalLevelKernel::Update || p.update) {
                        EXPECT_EQ(updatedAt[l], -1);
                        updatedAt[l] = dispatch;
                      }
                      if (k == MetalLevelKernel::Solve) {
                        EXPECT_TRUE(p.useList);
                        EXPECT_EQ(solvedAt[l], -1);
                        solvedAt[l] = dispatch;
                      }
                    }
                    dispatch++;
                  });

  // every lump in [begin, end) is solved, the others aren't
  for (int64_t l = 0; l < numLumps; l++) {
    if (l >= begin && l < end) {
      EXPECT_GE(solvedAt[l], 0) << "lump " << l;
      if (updatedAt[l] >= 0) {
        EXPECT_LE(updatedAt[l], solvedAt[l]);  // in the same dispatch, the update comes first
      }
    } else {
      EXPECT_EQ(solvedAt[l], -1) << "lump " << l;
    }
  }

  // the update of a lump comes after the solve of the lumps it uses
  for (auto [t, l] : blockPairs(skel)) {
    if (transpose) {
      // l in [begin, end) uses t (if t >= end it's not solved, its value is given)
      if (l < begin || l >= end || t >= end) {
        continue;
      }
      EXPECT_GE(updatedAt[l], 0) << "lump " << l;
      EXPECT_LT(solvedAt[t], updatedAt[l]) << "lumps " << t << " " << l;
    } else {
      // t uses l in [begin, end), also if t >= end (rows below the range are updated)
      if (l < begin || l >= end) {
        continue;
      }
      EXPECT_GE(updatedAt[t], 0) << "lump " << t;
      EXPECT_LT(solvedAt[l], updatedAt[t]) << "lumps " << l << " " << t;
    }
  }
}

static void testSkel(const CoalescedBlockMatrixSkel& skel) {
  checkRowChains(skel);
  const int64_t numLumps = skel.numLumps();
  vector<int64_t> cuts = {0, numLumps / 3, numLumps / 2, numLumps - 1, numLumps};
  for (bool transpose : {false, true}) {
    for (int64_t begin : cuts) {
      checkLevels(skel, begin, transpose);
      for (int64_t end : cuts) {
        if (end >= begin) {
          checkDispatches(skel, begin, end, transpose);
        }
      }
    }
  }
}

TEST(MetalSolveLevels, SparseElim) {
  for (int i = 0; i < 10; i++) {
    testSkel(*makeSkel(i, true));
  }
}

TEST(MetalSolveLevels, LargeLumps) {
  for (int i = 0; i < 10; i++) {
    testSkel(*makeSkel(i, false));
  }
}

TEST(MetalSolveLevels, Fewer_Levels_Than_Lumps) {
  // the levels are useful only if there are many fewer of them than lumps
  int64_t numLumps = 0, numLevels = 0;
  for (int i = 0; i < 10; i++) {
    auto skel = makeSkel(i, true);
    numLumps += skel->numLumps();
    numLevels += MetalSolveLevels::forward(*skel, 0).numLevels();
  }
  EXPECT_LT(numLevels * 2, numLumps);
}
