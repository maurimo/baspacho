/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "baspacho/baspacho/AnalysisParallel.h"
#include "baspacho/baspacho/SparseStructure.h"

#ifdef BASPACHO_USE_SUITESPARSE_AMD
#include <amd.h>
#else
#include <Eigen/OrderingMethods>
#endif

#include <dispenso/parallel_for.h>
#include <algorithm>
#include <cstdint>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#include "baspacho/baspacho/DebugMacros.h"
#include "baspacho/baspacho/Utils.h"

namespace BaSpaCho {

using namespace std;

void SparseStructure::sortIndices() {
  int64_t ord = order();
  for (int64_t i = 0; i < ord; i++) {
    int64_t start = ptrs[i];
    int64_t end = ptrs[i + 1];
    std::sort(inds.begin() + start, inds.begin() + end);
  }
}

// assumed square matrix
SparseStructure SparseStructure::transpose() const {
  int64_t ord = order();
  SparseStructure retv;
  retv.ptrs.assign(ord + 1, 0);

  for (int64_t i = 0; i < ord; i++) {
    int64_t start = ptrs[i];
    int64_t end = ptrs[i + 1];
    for (int64_t k = start; k < end; k++) {
      int64_t j = inds[k];
      BASPACHO_CHECK_LT(j, ord);
      retv.ptrs[j]++;
    }
  }

  int64_t tot = cumSumVec(retv.ptrs);
  retv.inds.resize(tot);

  for (int64_t i = 0; i < ord; i++) {
    int64_t start = ptrs[i];
    int64_t end = ptrs[i + 1];
    for (int64_t k = start; k < end; k++) {
      int64_t j = inds[k];
      BASPACHO_CHECK_LT(j, ord);
      BASPACHO_CHECK_LT(retv.ptrs[j], (int64_t)retv.inds.size());
      retv.inds[retv.ptrs[j]++] = i;
    }
  }

  rewindVec(retv.ptrs);

  return retv;
}

// assumed square matrix
SparseStructure SparseStructure::clear(bool lowerHalf) const {
  int64_t ord = order();
  SparseStructure retv;
  retv.ptrs.assign(ord + 1, 0);

  for (int64_t i = 0; i < ord; i++) {
    int64_t start = ptrs[i];
    int64_t end = ptrs[i + 1];
    for (int64_t k = start; k < end; k++) {
      int64_t j = inds[k];
      BASPACHO_CHECK_LT(j, ord);
      if (i != j && (j > i) == lowerHalf) {
        continue;
      }
      retv.ptrs[i]++;
    }
  }

  int64_t tot = cumSumVec(retv.ptrs);
  retv.inds.resize(tot);

  for (int64_t i = 0; i < ord; i++) {
    int64_t start = ptrs[i];
    int64_t end = ptrs[i + 1];
    for (int64_t k = start; k < end; k++) {
      int64_t j = inds[k];
      BASPACHO_CHECK_LT(j, ord);
      if (i != j && (j > i) == lowerHalf) {
        continue;
      }
      BASPACHO_CHECK_LT(retv.ptrs[i], (int64_t)retv.inds.size());
      retv.inds[retv.ptrs[i]++] = j;
    }
  }

  rewindVec(retv.ptrs);

  return retv;
}

SparseStructure SparseStructure::symmetricPermutation(const std::vector<int64_t>& mapPerm,
                                                      bool lowerHalf, bool sortIndices) const {
  int64_t ord = order();
  BASPACHO_CHECK_EQ(ord, (int64_t)mapPerm.size());

  SparseStructure retv;
  retv.ptrs.assign(ord + 1, 0);

  for (int64_t i = 0; i < ord; i++) {
    int64_t start = ptrs[i];
    int64_t end = ptrs[i + 1];
    int64_t newI = mapPerm[i];
    BASPACHO_CHECK_LT(newI, ord);
    for (int64_t k = start; k < end; k++) {
      int64_t j = inds[k];
      BASPACHO_CHECK_LT(j, ord);
      int64_t newJ = mapPerm[j];
      BASPACHO_CHECK_LT(newJ, ord);
      int64_t col = lowerHalf ? min(newI, newJ) : max(newI, newJ);
      retv.ptrs[col]++;
    }
  }

  int64_t tot = cumSumVec(retv.ptrs);
  retv.inds.resize(tot);

  for (int64_t i = 0; i < ord; i++) {
    int64_t start = ptrs[i];
    int64_t end = ptrs[i + 1];
    int64_t newI = mapPerm[i];
    BASPACHO_CHECK_LT(newI, ord);
    for (int64_t k = start; k < end; k++) {
      int64_t j = inds[k];
      BASPACHO_CHECK_LT(j, ord);
      int64_t newJ = mapPerm[j];
      BASPACHO_CHECK_LT(newJ, ord);
      int64_t col = lowerHalf ? min(newI, newJ) : max(newI, newJ);
      int64_t row = lowerHalf ? max(newI, newJ) : min(newI, newJ);
      BASPACHO_CHECK_LT(retv.ptrs[col], (int64_t)retv.inds.size());
      retv.inds[retv.ptrs[col]++] = row;
    }
  }

  rewindVec(retv.ptrs);

  if (sortIndices) {
    retv.sortIndices();
  }

  return retv;
}

static inline int64_t countTrailingZeros(uint64_t x) {  // x != 0
#if defined(_MSC_VER)
  unsigned long i;
  _BitScanForward64(&i, x);
  return i;
#else
  return __builtin_ctzll(x);
#endif
}

SparseStructure SparseStructure::addIndependentEliminationFill(int64_t elimStart, int64_t elimEnd,
                                                               bool sortIdx) const {
  int64_t ord = order();
  BASPACHO_CHECK_LE(0, elimStart);
  BASPACHO_CHECK_LE(elimStart, elimEnd);
  BASPACHO_CHECK_LE(elimEnd, ord);

  // nothing to do, no entries are added
  if (elimEnd == ord) {
    return *this;
  }

  // transpose of the columns in elim range, restricted to rows >= elimEnd:
  // elimRows[elimPtrs[i - elimStart]...] are the rows having an entry in column i,
  // sorted because rows are visited in increasing order.
  int64_t numElim = elimEnd - elimStart;
  vector<int64_t> elimPtrs(numElim + 1, 0), lastRow(numElim, -1);  // lastRow: skip duplicates
  for (int64_t k = elimEnd; k < ord; k++) {
    for (int64_t q = ptrs[k], qEnd = ptrs[k + 1]; q < qEnd; q++) {
      int64_t i = inds[q];
      if (i >= elimStart && i < elimEnd && lastRow[i - elimStart] != k) {
        lastRow[i - elimStart] = k;
        elimPtrs[i - elimStart]++;
      }
    }
  }
  int64_t numElimEntries = cumSumVec(elimPtrs);
  vector<int64_t> elimRows(numElimEntries);
  lastRow.assign(numElim, -1);
  for (int64_t k = elimEnd; k < ord; k++) {
    for (int64_t q = ptrs[k], qEnd = ptrs[k + 1]; q < qEnd; q++) {
      int64_t i = inds[q];
      if (i >= elimStart && i < elimEnd && lastRow[i - elimStart] != k) {
        lastRow[i - elimStart] = k;
        elimRows[elimPtrs[i - elimStart]++] = k;
      }
    }
  }
  rewindVec(elimPtrs);

  // The fill: pairs (k, w), elimEnd <= w < k, of rows having an entry in the same eliminated
  // column. If they fit in memory they are generated column by column (sequential access) into
  // per-row buckets, else each row walks the columns of its eliminated entries (random access)
  int64_t numRows0 = ord - elimEnd, numPairs = 0;
  for (int64_t c = 0; c < numElim; c++) {
    int64_t d = elimPtrs[c + 1] - elimPtrs[c];
    numPairs += d * (d - 1) / 2;
  }
  // With few rows (eg. the cameras of a BA problem) the pairs are set in a bitmap of rows x rows
  // (no duplicates, fits in cache), else they are collected in per-row buckets if they fit in
  // memory, else each row walks the columns of its eliminated entries
  constexpr int64_t kMaxBitmapRows = 16384;              // up to 32MB
  constexpr int64_t kMaxBucketPairs = int64_t(1) << 26;  // 256MB of int32 entries
  const bool bitmap = numRows0 <= kMaxBitmapRows;
  const bool buckets = !bitmap && numPairs <= kMaxBucketPairs && numRows0 < (int64_t(1) << 31);
  const int64_t bitmapWords = (numRows0 + 63) / 64;
  vector<uint64_t> pairBits;
  if (bitmap) {
    pairBits.assign(numRows0 * bitmapWords, 0);
    for (int64_t c = 0; c < numElim; c++) {
      for (int64_t b = elimPtrs[c] + 1; b < elimPtrs[c + 1]; b++) {
        uint64_t* row = pairBits.data() + (elimRows[b] - elimEnd) * bitmapWords;
        for (int64_t a = elimPtrs[c]; a < b; a++) {
          int64_t w = elimRows[a] - elimEnd;
          row[w >> 6] |= uint64_t(1) << (w & 63);
        }
      }
    }
  }
  vector<int64_t> fillPtr;
  vector<int32_t> fillRows;  // relative to elimEnd
  if (buckets) {
    fillPtr.assign(numRows0 + 1, 0);
    for (int64_t c = 0; c < numElim; c++) {
      for (int64_t b = elimPtrs[c] + 1; b < elimPtrs[c + 1]; b++) {
        fillPtr[elimRows[b] - elimEnd] += b - elimPtrs[c];  // all the rows before it
      }
    }
    cumSumVec(fillPtr);
    fillRows.resize(fillPtr[numRows0]);
    for (int64_t c = 0; c < numElim; c++) {
      for (int64_t b = elimPtrs[c] + 1; b < elimPtrs[c + 1]; b++) {
        int64_t& pos = fillPtr[elimRows[b] - elimEnd];
        for (int64_t a = elimPtrs[c]; a < b; a++) {
          fillRows[pos++] = (int32_t)(elimRows[a] - elimEnd);
        }
      }
    }
    rewindVec(fillPtr);
  }

  // compute rows >= elimEnd in parallel, in chunks. Row k contains the original
  // entries with column < elimEnd, followed by the union (sorted) of the original
  // entries in [elimEnd, k] and of the fill L(k,w) for each w having a common
  // neighbor in the elimination range.
  int64_t numRows = ord - elimEnd;
  int64_t numChunks = std::min<int64_t>(numRows, 256);
  vector<vector<int64_t>> chunkInds(numChunks);
  vector<int64_t> rowSize(numRows);
  // each call processes a few chunks, the work arrays are allocated once per call
  int64_t chunksPerCall = std::max<int64_t>(1, numChunks / (4 * analysisThreads()));
  analysisParallelFor(0L, numChunks, chunksPerCall, [&](int64_t c0, int64_t c1) {
    vector<int64_t> tags(numRows, -1);
    vector<int64_t> high, low;
    for (int64_t c = c0; c < c1; c++) {
      int64_t kBegin = elimEnd + numRows * c / numChunks;
      int64_t kEnd = elimEnd + numRows * (c + 1) / numChunks;
      auto& out = chunkInds[c];
      for (int64_t k = kBegin; k < kEnd; k++) {
        size_t rowBegin = out.size();
        high.clear();
        tags[k - elimEnd] = k;
        high.push_back(k);
        // entries before elimEnd, deduplicated (sorted if needed, ie if not increasing)
        low.clear();
        bool increasing = true;
        for (int64_t q = ptrs[k], qEnd = ptrs[k + 1]; q < qEnd; q++) {
          int64_t i = inds[q];
          if (i < elimEnd) {
            increasing = increasing && (low.empty() || low.back() < i);
            low.push_back(i);
          }
        }
        if (!increasing) {
          std::sort(low.begin(), low.end());
          low.erase(std::unique(low.begin(), low.end()), low.end());
        }
        if (bitmap) {
          out.insert(out.end(), low.begin(), low.end());
          const uint64_t* row = pairBits.data() + (k - elimEnd) * bitmapWords;
          for (int64_t wd = 0; wd < bitmapWords; wd++) {
            for (uint64_t bits = row[wd]; bits; bits &= bits - 1) {
              int64_t w = elimEnd + wd * 64 + countTrailingZeros(bits);
              tags[w - elimEnd] = k;
              high.push_back(w);
            }
          }
        } else if (buckets) {
          out.insert(out.end(), low.begin(), low.end());
          for (int64_t q = fillPtr[k - elimEnd], qEnd = fillPtr[k - elimEnd + 1]; q < qEnd; q++) {
            int64_t w = fillRows[q] + elimEnd;
            if (tags[w - elimEnd] != k) {
              tags[w - elimEnd] = k;
              high.push_back(w);
            }
          }
        }
        for (int64_t i : (bitmap || buckets) ? std::vector<int64_t>() : low) {
          out.push_back(i);
          if (i >= elimStart) {  // walk rows of elim column i, below k
            for (int64_t t = elimPtrs[i - elimStart], tEnd = elimPtrs[i - elimStart + 1]; t < tEnd;
                 t++) {
              int64_t w = elimRows[t];
              if (w >= k) {
                break;  // sorted
              }
              if (tags[w - elimEnd] != k) {
                tags[w - elimEnd] = k;
                high.push_back(w);
              }
            }
          }
        }
        for (int64_t q = ptrs[k], qEnd = ptrs[k + 1]; q < qEnd; q++) {
          int64_t i = inds[q];
          if (i > k) {
            continue;  // upper half entries are ignored (as the diagonal, added above)
          }
          if (i >= elimEnd && tags[i - elimEnd] != k) {
            tags[i - elimEnd] = k;
            high.push_back(i);
          }
        }
        if (sortIdx) {
          std::sort(out.begin() + rowBegin, out.end());
          out.erase(std::unique(out.begin() + rowBegin, out.end()), out.end());
          std::sort(high.begin(), high.end());
        }
        out.insert(out.end(), high.begin(), high.end());
        rowSize[k - elimEnd] = out.size() - rowBegin;
      }
    }
  });

  SparseStructure retv;
  retv.ptrs.resize(ord + 1);
  std::copy(ptrs.begin(), ptrs.begin() + elimEnd + 1, retv.ptrs.begin());
  for (int64_t k = elimEnd; k < ord; k++) {
    retv.ptrs[k + 1] = retv.ptrs[k] + rowSize[k - elimEnd];
  }
  retv.inds.resize(retv.ptrs[ord]);
  std::copy(inds.begin(), inds.begin() + ptrs[elimEnd], retv.inds.begin());
  if (sortIdx) {
    for (int64_t k = 0; k < elimEnd; k++) {
      std::sort(retv.inds.begin() + retv.ptrs[k], retv.inds.begin() + retv.ptrs[k + 1]);
    }
  }
  analysisParallelFor(0L, numChunks, 1L,
                         [&](int64_t c0, int64_t c1) {
                           for (int64_t c = c0; c < c1; c++) {
                             int64_t kBegin = elimEnd + numRows * c / numChunks;
                             std::copy(chunkInds[c].begin(), chunkInds[c].end(),
                                       retv.inds.begin() + retv.ptrs[kBegin]);
                           }
                         });

  return retv;
}

SparseStructure SparseStructure::addFullEliminationFill() const {
  int64_t ord = order();
  vector<int64_t> tags(ord), parent(ord, -1);

  // skeleton of the algo to iterate on fillup's nodes is from Eigen's
  // `SimplicialCholesky_impl.h` (by Gael Guennebaud),
  // in turn from LDL by Timothy A. Davis.
  SparseStructure retv;
  retv.ptrs.assign(ord + 1, 1);  // write sizes here initially
  for (int64_t k = 0; k < ord; ++k) {
    /* L(k,:) pattern: all nodes reachable in etree from nz in A(0:k-1,k) */
    parent[k] = -1; /* parent of k is not yet known */
    tags[k] = k;    /* mark node k as visited, L(k,k) is nonzero */

    int64_t start = ptrs[k];
    int64_t end = ptrs[k + 1];
    for (int64_t q = start; q < end; q++) {
      int64_t i = inds[q];
      if (i >= k) {
        continue;
      }
      /* follow path from i to root of etree, stop at flagged node */
      for (; tags[i] != k; i = parent[i]) {
        /* find parent of i if not yet determined */
        if (parent[i] == -1) {
          parent[i] = k;
        }

        retv.ptrs[k]++; /* L(k,i) is nonzero */
        tags[i] = k;
      }
    }
  }

  // cumulate-sum ptrs: sizes -> pointers
  int64_t tot = cumSumVec(retv.ptrs);
  retv.inds.resize(tot);

  // walk again, saving entries in rows
  for (int64_t k = 0; k < ord; ++k) {
    /* L(k,:) pattern: all nodes reachable in etree from nz in A(0:k-1,k) */
    parent[k] = -1;                /* parent of k is not yet known */
    tags[k] = k;                   /* mark node k as visited */
    retv.inds[retv.ptrs[k]++] = k; /* L(k,k) is nonzero */

    int64_t start = ptrs[k];
    int64_t end = ptrs[k + 1];
    for (int64_t q = start; q < end; q++) {
      int64_t i = inds[q];
      if (i < k) {
        /* follow path from i to root of etree, stop at flagged node
         */
        for (; tags[i] != k; i = parent[i]) {
          /* find parent of i if not yet determined */
          if (parent[i] == -1) {
            parent[i] = k;
          }
          retv.inds[retv.ptrs[k]++] = i; /* L(k,i) is nonzero */
          tags[i] = k;                   /* mark i as visited */
        }
      }
    }
  }

  rewindVec(retv.ptrs);

  retv.sortIndices();

  return retv;
}

#ifdef BASPACHO_USE_SUITESPARSE_AMD

std::vector<int64_t> SparseStructure::fillReducingPermutation() const {
  std::vector<int64_t> colPtr(ptrs.begin(), ptrs.end()), rowInd(inds.begin(), inds.end());
  std::vector<int64_t> P(colPtr.size() - 1);
  double Control[AMD_CONTROL], Info[AMD_INFO];

  amd_l_defaults(Control);
  // amd_l_control(Control); // print verbose messages

  int result = amd_l_order(P.size(), colPtr.data(), rowInd.data(), P.data(), Control, Info);
  BASPACHO_CHECK_EQ(result, AMD_OK);

  return std::vector<int64_t>(P.begin(), P.end());
}

#else

std::vector<int64_t> SparseStructure::fillReducingPermutation() const {
  using INT = int;

  std::vector<INT> colPtr(ptrs.begin(), ptrs.end()), rowInd(inds.begin(), inds.end());

  using namespace Eigen;

  // NOTE: feeding directly Map<...> references to AMDOrdering doesn't work,
  // this is probably a bug/limitation of Eigen
  vector<int8_t> fakeData(rowInd.size());
  Map<SparseMatrix<int8_t, ColMajor, INT> > matMap(colPtr.size() - 1, colPtr.size() - 1,
                                                   rowInd.size(), colPtr.data(), rowInd.data(),
                                                   (int8_t*)fakeData.data());
  SparseMatrix<int8_t, ColMajor, INT> mat = matMap;
  PermutationMatrix<Dynamic, Dynamic, INT> perm;
  AMDOrdering<INT>()(mat, perm);

  return std::vector<int64_t>(perm.indices().data(), perm.indices().data() + perm.size());
}

#endif

SparseStructure SparseStructure::extractRightBottom(int64_t startRow) const {
  int64_t ord = order();
  BASPACHO_CHECK_LE(startRow, ord);
  BASPACHO_CHECK_GE(startRow, 0);
  int64_t newOrd = ord - startRow;

  SparseStructure retv;
  retv.ptrs.assign(newOrd + 1, 0);

  for (int64_t i = startRow; i < ord; i++) {
    int64_t start = ptrs[i];
    int64_t end = ptrs[i + 1];
    for (int64_t k = start; k < end; k++) {
      int64_t j = inds[k];
      BASPACHO_CHECK_LT(j, ord);
      if (j >= startRow) {
        retv.ptrs[i - startRow]++;
      }
    }
  }

  int64_t tot = cumSumVec(retv.ptrs);
  retv.inds.resize(tot);

  for (int64_t i = startRow; i < ord; i++) {
    int64_t start = ptrs[i];
    int64_t end = ptrs[i + 1];
    for (int64_t k = start; k < end; k++) {
      int64_t j = inds[k];
      BASPACHO_CHECK_LT(j, ord);
      if (j >= startRow) {
        BASPACHO_CHECK_LT(retv.ptrs[i - startRow], (int64_t)retv.inds.size());
        retv.inds[retv.ptrs[i - startRow]++] = j - startRow;
      }
    }
  }

  rewindVec(retv.ptrs);
  return retv;
}

}  // end namespace BaSpaCho
