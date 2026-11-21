/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file MetalSolveLevels.h
 * @brief Level scheduling of the triangular solves of the Metal backend (plain C++)
 *
 * Solving with L lump by lump needs a few GPU dispatches per lump, each one depending on the
 * previous one. Instead, the lumps are grouped in levels (of the elimination tree): the lumps of a
 * level only depend on the lumps of the previous levels, so a level is processed by a single
 * dispatch with one threadgroup per lump (updating and then solving it), whatever the number of
 * lumps in it.
 *
 * In the solve with L, sibling lumps of a level update the same rows (of their parent), so the
 * updates are "gathered" instead: when lump t is processed it subtracts from its rows the products
 * of the blocks in row t with the (already solved) lumps they are in the columns of. The blocks in
 * a row are found by MetalRowChains. In the solve with L^T the update of lump l comes from the
 * blocks in column l, and is gathered as well, so no two threadgroups write the same data.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "baspacho/baspacho/CoalescedBlockMatrix.h"
#include "baspacho/baspacho/DebugMacros.h"

namespace BaSpaCho {

// Must match LevelSolveParams in MetalKernels.metal
struct MetalLevelSolveParams {
  int64_t listOff;   // the lump of threadgroup tg is levelLumps[listOff + tg]...
  int64_t useList;   // ...if useList, else it's listOff + tg
  int64_t srcBegin;  // the update of a lump in the solve with L only uses the blocks in the
  int64_t srcEnd;    // columns of the lumps in [srcBegin, srcEnd)
  int64_t ldc;       // vectors are column-major (stride ldc, nRHS columns)
  int64_t nRHS;
  int64_t transpose;  // solve with L^T
  int64_t update;     // level_solve_float: update the lumps before solving them
};
static_assert(sizeof(MetalLevelSolveParams) == 8 * 8, "MetalLevelSolveParams layout");

// The blocks below the diagonal, by rows: for span s, the chains with row span s are
// chain[ptr[s]], ..., chain[ptr[s + 1] - 1], in the columns of lumps lump[...] (sorted)
struct MetalRowChains {
  std::vector<int64_t> ptr;    // per span, with end
  std::vector<int64_t> chain;  // index of the chain (in chainData, chainRowSpan)
  std::vector<int64_t> lump;   // lump of the column of the chain

  explicit MetalRowChains(const CoalescedBlockMatrixSkel& skel) {
    const int64_t numSpans = skel.numSpans(), numLumps = skel.numLumps();
    ptr.assign(numSpans + 1, 0);
    for (int64_t l = 0; l < numLumps; l++) {
      for (int64_t c = skel.chainColPtr[l]; c < skel.chainColPtr[l + 1]; c++) {
        int64_t s = skel.chainRowSpan[c];
        if (skel.spanToLump[s] > l) {
          ptr[s + 1]++;
        }
      }
    }
    for (int64_t s = 0; s < numSpans; s++) {
      ptr[s + 1] += ptr[s];
    }
    chain.resize(ptr[numSpans]);
    lump.resize(ptr[numSpans]);
    std::vector<int64_t> pos(ptr.begin(), ptr.end() - 1);
    for (int64_t l = 0; l < numLumps; l++) {  // by increasing column lump
      for (int64_t c = skel.chainColPtr[l]; c < skel.chainColPtr[l + 1]; c++) {
        int64_t s = skel.chainRowSpan[c];
        if (skel.spanToLump[s] > l) {
          chain[pos[s]] = c;
          lump[pos[s]] = l;
          pos[s]++;
        }
      }
    }
  }
};

// The lumps in [begin, numLumps) grouped in levels, such that the lumps of a level only depend on
// the lumps of previous levels, for the solve with L (forward) or L^T (backward)
struct MetalSolveLevels {
  std::vector<int64_t> levelPtr;  // with end
  std::vector<int64_t> lumps;     // the lumps of the levels, sorted within each level

  // Solve with L: lump t depends on the lumps l in [begin, t) having blocks in row t
  static MetalSolveLevels forward(const CoalescedBlockMatrixSkel& skel, int64_t begin) {
    const int64_t numLumps = skel.numLumps();
    std::vector<int64_t> level(numLumps, 0);
    for (int64_t l = begin; l < numLumps; l++) {  // level[l] is final when reached
      for (int64_t c = skel.chainColPtr[l]; c < skel.chainColPtr[l + 1]; c++) {
        int64_t t = skel.spanToLump[skel.chainRowSpan[c]];
        if (t > l) {
          level[t] = std::max(level[t], level[l] + 1);
        }
      }
    }
    return fromLevels(level, begin);
  }

  // Solve with L^T: lump l depends on the lumps t > l having blocks in column l
  static MetalSolveLevels backward(const CoalescedBlockMatrixSkel& skel, int64_t begin) {
    const int64_t numLumps = skel.numLumps();
    std::vector<int64_t> level(numLumps, 0);
    for (int64_t l = numLumps - 1; l >= begin; l--) {
      for (int64_t c = skel.chainColPtr[l]; c < skel.chainColPtr[l + 1]; c++) {
        int64_t t = skel.spanToLump[skel.chainRowSpan[c]];
        if (t > l) {
          level[l] = std::max(level[l], level[t] + 1);
        }
      }
    }
    return fromLevels(level, begin);
  }

  int64_t numLevels() const { return (int64_t)levelPtr.size() - 1; }

  // The lumps of level lv that are in [begin, end), as range of indices in `lumps`
  std::pair<int64_t, int64_t> levelRange(int64_t lv, int64_t begin, int64_t end) const {
    auto first = lumps.begin() + levelPtr[lv], last = lumps.begin() + levelPtr[lv + 1];
    auto a = std::lower_bound(first, last, begin);
    auto b = std::lower_bound(a, last, end);
    return {a - lumps.begin(), b - lumps.begin()};
  }

 private:
  static MetalSolveLevels fromLevels(const std::vector<int64_t>& level, int64_t begin) {
    const int64_t numLumps = level.size();
    MetalSolveLevels retv;
    int64_t numLevels = 0;
    for (int64_t l = begin; l < numLumps; l++) {
      numLevels = std::max(numLevels, level[l] + 1);
    }
    retv.levelPtr.assign(numLevels + 1, 0);
    for (int64_t l = begin; l < numLumps; l++) {
      retv.levelPtr[level[l] + 1]++;
    }
    for (int64_t lv = 0; lv < numLevels; lv++) {
      retv.levelPtr[lv + 1] += retv.levelPtr[lv];
    }
    retv.lumps.resize(numLumps - begin);
    std::vector<int64_t> pos(retv.levelPtr.begin(), retv.levelPtr.end() - 1);
    for (int64_t l = begin; l < numLumps; l++) {  // sorted within each level
      retv.lumps[pos[level[l]]++] = l;
    }
    return retv;
  }
};

// Kernels of the level solve (see MetalKernels.metal)
enum class MetalLevelKernel {
  Update,  // level_update_float: the update of the rows of each lump
  Solve,   // level_solve_float: the update (if params.update) and solve of each lump
};

// Threads per threadgroup for lumps up to maxLumpSize: a power of 2 in [32, 256], covering the
// diagonal tiles (up to 64 rows) for up to 4 right hand sides at once
inline int64_t metalLevelThreads(int64_t maxLumpSize, int64_t nRHS) {
  int64_t want = std::min<int64_t>(maxLumpSize, 64) * std::min<int64_t>(nRHS, 4);
  int64_t nth = 32;
  while (nth < want && nth < 256) {
    nth *= 2;
  }
  return nth;
}

// Runs the solve with L (or L^T if transpose) of the lumps in [lumpsBegin, lumpsEnd), as the
// per-lump loops in Solver::internalSolveLRange (internalSolveLtRange). The levels must have been
// computed from lumpsBegin. Calls launch(kernel, params, numThreadgroups, threadsPerThreadgroup)
// for each dispatch, in order: one dispatch per level (and one for the rows after lumpsEnd in the
// solve with L)
template <typename Launch>
void metalLevelSolve(const CoalescedBlockMatrixSkel& skel, const MetalSolveLevels& levels,
                     bool transpose, int64_t lumpsBegin, int64_t lumpsEnd, int64_t ldc,
                     int64_t nRHS, Launch&& launch) {
  BASPACHO_CHECK_LE(lumpsBegin, lumpsEnd);
  const int64_t numLumps = skel.numLumps();
  auto maxLumpSize = [&](auto lumpAt, int64_t n) {
    int64_t retv = 0;
    for (int64_t i = 0; i < n; i++) {
      int64_t l = lumpAt(i);
      retv = std::max(retv, skel.lumpStart[l + 1] - skel.lumpStart[l]);
    }
    return retv;
  };

  MetalLevelSolveParams p = {};
  p.useList = 1;
  p.srcBegin = lumpsBegin;
  p.srcEnd = lumpsEnd;
  p.ldc = ldc;
  p.nRHS = nRHS;
  p.transpose = transpose ? 1 : 0;
  for (int64_t lv = 0; lv < levels.numLevels(); lv++) {
    auto [a, b] = levels.levelRange(lv, lumpsBegin, lumpsEnd);
    if (a == b) {
      continue;
    }
    int64_t nth =
        metalLevelThreads(maxLumpSize([&](int64_t i) { return levels.lumps[a + i]; }, b - a), nRHS);
    p.listOff = a;
    p.update = lv > 0 ? 1 : 0;  // lumps in level 0 have no update (nothing before/below them)
    launch(MetalLevelKernel::Solve, p, b - a, nth);
  }

  // solve with L: the rows after lumpsEnd are updated with the solved lumps (not solved)
  if (!transpose && lumpsBegin < lumpsEnd && lumpsEnd < numLumps) {
    p.useList = 0;
    p.listOff = lumpsEnd;
    p.update = 1;
    int64_t nth = metalLevelThreads(
        maxLumpSize([&](int64_t i) { return lumpsEnd + i; }, numLumps - lumpsEnd), nRHS);
    launch(MetalLevelKernel::Update, p, numLumps - lumpsEnd, nth);
  }
}

}  // end namespace BaSpaCho
