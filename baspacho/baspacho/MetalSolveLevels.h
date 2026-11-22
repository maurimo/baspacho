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
 * level only depend on the lumps of the previous levels, so a level is processed by a few
 * dispatches with one threadgroup per lump, whatever the number of lumps in it.
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
  int64_t maxDiag;    // level_diag_solve_float skips the lumps larger than this
  int64_t rowChunk;   // level_update_float: rows of a lump per threadgroup (y = chunk index)
};
static_assert(sizeof(MetalLevelSolveParams) == 9 * 8, "MetalLevelSolveParams layout");

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
  // forward: the last lump l < t with blocks in row t, per lump t (-1 if none): the update of t
  // in a solve of the lumps from `begin` on is needed only if lastSource[t] >= begin
  std::vector<int64_t> lastSource;

  // Solve with L: lump t depends on the lumps l in [begin, t) having blocks in row t
  static MetalSolveLevels forward(const CoalescedBlockMatrixSkel& skel, int64_t begin) {
    const int64_t numLumps = skel.numLumps();
    std::vector<int64_t> level(numLumps, 0), lastSource(numLumps, -1);
    for (int64_t l = begin; l < numLumps; l++) {  // level[l] is final when reached
      for (int64_t c = skel.chainColPtr[l]; c < skel.chainColPtr[l + 1]; c++) {
        int64_t t = skel.spanToLump[skel.chainRowSpan[c]];
        if (t > l) {
          level[t] = std::max(level[t], level[l] + 1);
          lastSource[t] = l;  // l increasing
        }
      }
    }
    MetalSolveLevels retv = fromLevels(level, begin);
    retv.lastSource = std::move(lastSource);
    return retv;
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
  Update,          // level_update_float: the update of the rows of each lump
  DiagSolve,       // level_diag_solve_float: the solve with the diagonal block of each lump
  LargeDiagSolve,  // the solve with the diagonal block of a large lump, by the tiled kernels
};

// A dispatch of the level solve: threadgroups (groupsX, groupsY) of `threads` threads, or the
// tiled solve of `lump` for LargeDiagSolve
struct MetalLevelDispatch {
  MetalLevelKernel kernel;
  MetalLevelSolveParams params;
  int64_t groupsX, groupsY, threads;
  int64_t lump;
};

// Rows of a lump updated by a threadgroup of level_update_float (large lumps are split)
constexpr int64_t kMetalLevelRowChunk = 64;

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
// computed from a lump <= lumpsBegin (eg. 0, one schedule for all the ranges): restricted to
// [lumpsBegin, lumpsEnd) they still respect the dependencies. For each level:
// - the rows of its lumps are updated (one dispatch, large lumps split in chunks of rows)
// - the lumps up to maxDiag rows are solved with their diagonal block (one dispatch, one
//   threadgroup per lump), the larger ones by the tiled kernels (which use many threadgroups)
// Calls launch(const MetalLevelDispatch&) for each dispatch, in order
template <typename Launch>
void metalLevelSolve(const CoalescedBlockMatrixSkel& skel, const MetalSolveLevels& levels,
                     bool transpose, int64_t lumpsBegin, int64_t lumpsEnd, int64_t ldc,
                     int64_t nRHS, int64_t maxDiag, Launch&& launch) {
  BASPACHO_CHECK_LE(lumpsBegin, lumpsEnd);
  const int64_t numLumps = skel.numLumps();
  auto lumpSize = [&](int64_t l) { return skel.lumpStart[l + 1] - skel.lumpStart[l]; };

  MetalLevelDispatch d = {};
  MetalLevelSolveParams& p = d.params;
  p.useList = 1;
  p.srcBegin = lumpsBegin;
  p.srcEnd = lumpsEnd;
  p.ldc = ldc;
  p.nRHS = nRHS;
  p.transpose = transpose ? 1 : 0;
  p.maxDiag = maxDiag;
  p.rowChunk = kMetalLevelRowChunk;
  auto update = [&](int64_t numLumps, int64_t maxSize) {
    d.kernel = MetalLevelKernel::Update;
    d.groupsX = numLumps;
    d.groupsY = (maxSize + kMetalLevelRowChunk - 1) / kMetalLevelRowChunk;
    d.threads = metalLevelThreads(maxSize, nRHS);
    launch(d);
  };

  for (int64_t lv = 0; lv < levels.numLevels(); lv++) {
    auto [a, b] = levels.levelRange(lv, lumpsBegin, lumpsEnd);
    if (a == b) {
      continue;
    }
    int64_t maxSize = 0, maxSmallSize = 0;
    bool needUpdate = false;  // solve with L^T: lumps in level > 0 have blocks below them
    for (int64_t i = a; i < b; i++) {
      int64_t l = levels.lumps[i];
      needUpdate |= transpose ? lv > 0 : levels.lastSource[l] >= lumpsBegin;
      int64_t n = lumpSize(l);
      maxSize = std::max(maxSize, n);
      if (n <= maxDiag) {
        maxSmallSize = std::max(maxSmallSize, n);
      }
    }
    p.listOff = a;
    if (needUpdate) {  // eg. the lumps of the first levels, or only fed by lumps < lumpsBegin
      update(b - a, maxSize);
    }
    if (maxSmallSize > 0) {
      d.kernel = MetalLevelKernel::DiagSolve;
      d.groupsX = b - a;
      d.groupsY = 1;
      d.threads = metalLevelThreads(maxSmallSize, nRHS);
      launch(d);
    }
    for (int64_t i = a; i < b; i++) {
      if (lumpSize(levels.lumps[i]) > maxDiag) {
        d.kernel = MetalLevelKernel::LargeDiagSolve;
        d.lump = levels.lumps[i];
        d.groupsX = d.groupsY = d.threads = 0;
        launch(d);
      }
    }
  }

  // solve with L: the rows after lumpsEnd are updated with the solved lumps (not solved)
  if (!transpose && lumpsBegin < lumpsEnd && lumpsEnd < numLumps) {
    p.useList = 0;
    p.listOff = lumpsEnd;
    int64_t maxSize = 0;
    for (int64_t l = lumpsEnd; l < numLumps; l++) {
      maxSize = std::max(maxSize, lumpSize(l));
    }
    update(numLumps - lumpsEnd, maxSize);
  }
}

}  // end namespace BaSpaCho
