/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "baspacho/baspacho/CoalescedBlockMatrix.h"
#include <dispenso/parallel_for.h>
#include <iostream>
#include "baspacho/baspacho/DebugMacros.h"
#include "baspacho/baspacho/Utils.h"

namespace BaSpaCho {

using namespace std;

CoalescedBlockMatrixSkel::CoalescedBlockMatrixSkel(const vector<int64_t>& spanStart,
                                                   const vector<int64_t>& lumpToSpan,
                                                   const vector<int64_t>& colPtr,
                                                   const vector<int64_t>& rowInd)
    : spanStart(spanStart), lumpToSpan(lumpToSpan) {
  BASPACHO_CHECK_GE(spanStart.size(), lumpToSpan.size());
  BASPACHO_CHECK_GE((int64_t)lumpToSpan.size(), 1);
  BASPACHO_CHECK_EQ((int64_t)spanStart.size() - 1, lumpToSpan[lumpToSpan.size() - 1]);
  BASPACHO_CHECK_EQ(colPtr.size(), lumpToSpan.size());
  BASPACHO_CHECK(isStrictlyIncreasing(spanStart, 0, spanStart.size()));
  BASPACHO_CHECK(isStrictlyIncreasing(lumpToSpan, 0, lumpToSpan.size()));

  int64_t totSize = spanStart[spanStart.size() - 1];
  int64_t numSpans = spanStart.size() - 1;
  int64_t numLumps = lumpToSpan.size() - 1;

  spanToLump.resize(numSpans + 1);
  lumpStart.resize(numLumps + 1);
  for (int64_t l = 0; l < numLumps; l++) {
    int64_t sBegin = lumpToSpan[l];
    int64_t sEnd = lumpToSpan[l + 1];
    lumpStart[l] = spanStart[sBegin];
    for (int64_t s = sBegin; s < sEnd; s++) {
      spanToLump[s] = l;
    }
  }
  spanToLump[numSpans] = numLumps;
  lumpStart[numLumps] = totSize;
  spanOffsetInLump.resize(numSpans + 1);
  for (int64_t s = 0; s < numSpans; s++) {
    spanOffsetInLump[s] = spanStart[s] - lumpStart[spanToLump[s]];
  }
  spanOffsetInLump[numSpans] = 0;

  // Two passes, parallel on lumps: first compute per-lump sizes (number of chains,
  // data size, number of boards), then after prefix sums fill all arrays.
  BASPACHO_CHECK_EQ(colPtr[0], 0);
  int64_t numChains = colPtr[numLumps];
  BASPACHO_CHECK_EQ(numChains, (int64_t)rowInd.size());
  chainColPtr.assign(colPtr.begin(), colPtr.end());
  chainRowSpan.assign(rowInd.begin(), rowInd.end());
  chainData.resize(numChains + 1);
  chainRowsTillEnd.resize(numChains);
  boardColPtr.assign(numLumps + 1, 0);
  vector<int64_t> lumpDataSize(numLumps + 1, 0);

  constexpr int64_t kLumpChunk = 256;
  dispenso::parallel_for(
      dispenso::makeChunkedRange(0L, numLumps, kLumpChunk), [&](int64_t l0, int64_t l1) {
        for (int64_t l = l0; l < l1; l++) {
          int64_t colStart = colPtr[l];
          int64_t colEnd = colPtr[l + 1];
          BASPACHO_CHECK(isStrictlyIncreasing(rowInd, colStart, colEnd));
          int64_t lSpanBegin = lumpToSpan[l];
          int64_t lSpanEnd = lumpToSpan[l + 1];
          int64_t lSpanSize = lSpanEnd - lSpanBegin;

          // check the initial section is the set of params from `a`, and
          // therefore the full diagonal block is contained in the matrix
          // Column must contain full diagonal block:
          BASPACHO_CHECK_GE(colEnd - colStart, lSpanSize);
          // Column data must start at diagonal block:
          BASPACHO_CHECK_EQ(rowInd[colStart], lSpanBegin);
          // Column must contain full diagonal block:
          BASPACHO_CHECK_EQ(rowInd[colStart + lSpanSize - 1], lSpanEnd - 1);

          int64_t numRows = 0, numBoards = 0, currentRowLump = kInvalid;
          for (int64_t i = colStart; i < colEnd; i++) {
            int64_t p = rowInd[i];
            numRows += spanStart[p + 1] - spanStart[p];
            chainRowsTillEnd[i] = numRows;
            int64_t rowLump = spanToLump[p];
            if (rowLump != currentRowLump) {
              currentRowLump = rowLump;
              numBoards++;
            }
          }
          lumpDataSize[l] = numRows * (lumpStart[l + 1] - lumpStart[l]);
          boardColPtr[l] = numBoards + 1;  // + end marker
        }
      });
  cumSumVec(lumpDataSize);
  int64_t totNumBoards = cumSumVec(boardColPtr);
  boardRowLump.resize(totNumBoards);
  boardChainColOrd.resize(totNumBoards);

  dispenso::parallel_for(
      dispenso::makeChunkedRange(0L, numLumps, kLumpChunk), [&](int64_t l0, int64_t l1) {
        for (int64_t l = l0; l < l1; l++) {
          int64_t colStart = colPtr[l];
          int64_t colEnd = colPtr[l + 1];
          int64_t lDataSize = lumpStart[l + 1] - lumpStart[l];
          int64_t dataPtr = lumpDataSize[l];
          int64_t b = boardColPtr[l];
          int64_t currentRowLump = kInvalid;
          for (int64_t i = colStart; i < colEnd; i++) {
            int64_t p = rowInd[i];
            chainData[i] = dataPtr;
            dataPtr += lDataSize * (spanStart[p + 1] - spanStart[p]);
            int64_t rowLump = spanToLump[p];
            if (rowLump != currentRowLump) {
              currentRowLump = rowLump;
              boardRowLump[b] = rowLump;
              boardChainColOrd[b] = i - colStart;
              b++;
            }
          }
          boardRowLump[b] = kInvalid;
          boardChainColOrd[b] = colEnd - colStart;
          BASPACHO_CHECK_EQ(b + 1, boardColPtr[l + 1]);
        }
      });
  chainData[numChains] = lumpDataSize[numLumps];

  boardRowPtr.assign(numLumps + 1, 0);
  for (int64_t l = 0; l < numLumps; l++) {
    for (int64_t i = boardColPtr[l]; i < boardColPtr[l + 1] - 1; i++) {
      int64_t rowLump = boardRowLump[i];
      boardRowPtr[rowLump]++;
    }
  }
  int64_t numBoards = cumSumVec(boardRowPtr);
  boardColLump.resize(numBoards);
  boardColOrd.resize(numBoards);
  for (int64_t l = 0; l < numLumps; l++) {
    for (int64_t i = boardColPtr[l]; i < boardColPtr[l + 1] - 1; i++) {
      int64_t rowLump = boardRowLump[i];
      boardColLump[boardRowPtr[rowLump]] = l;
      boardColOrd[boardRowPtr[rowLump]] = i - boardColPtr[l];
      boardRowPtr[rowLump]++;
    }
  }
  rewindVec(boardRowPtr);
}

template <typename T>
void CoalescedBlockMatrixSkel::densify(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& dense,
                                       const T* data, bool fillUpperHalf,
                                       int64_t startSpanIndex) const {
  BASPACHO_CHECK_GE(startSpanIndex, 0);
  BASPACHO_CHECK_LT(startSpanIndex, (int64_t)spanOffsetInLump.size());
  BASPACHO_CHECK_EQ(spanOffsetInLump[startSpanIndex], 0);
  int64_t startLump = spanToLump[startSpanIndex];

  int64_t offset = spanStart[startSpanIndex];
  int64_t totSize = spanStart[spanStart.size() - 1] - offset;
  dense.resize(totSize, totSize);
  dense.setZero();

  for (size_t a = startLump; a < chainColPtr.size() - 1; a++) {
    int64_t lBegin = lumpStart[a];
    int64_t lSize = lumpStart[a + 1] - lBegin;
    int64_t colStart = chainColPtr[a];
    int64_t colEnd = chainColPtr[a + 1];
    for (int64_t i = colStart; i < colEnd; i++) {
      int64_t p = chainRowSpan[i];
      int64_t pStart = spanStart[p];
      int64_t pSize = spanStart[p + 1] - pStart;
      int64_t dataPtr = chainData[i];

      dense.block(pStart - offset, lBegin - offset, pSize, lSize) =
          Eigen::Map<const MatRMaj<T>>(data + dataPtr, pSize, lSize);
    }
  }

  if (fillUpperHalf) {
    dense.template triangularView<Eigen::StrictlyUpper>() =
        dense.template triangularView<Eigen::StrictlyLower>().adjoint();
  }
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> CoalescedBlockMatrixSkel::densify(
    const std::vector<T>& data, bool fillUpperHalf) const {
  int64_t totData = chainData[chainData.size() - 1];
  BASPACHO_CHECK_EQ(totData, (int64_t)data.size());

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> dense;
  densify(dense, data.data(), fillUpperHalf);

  return dense;
}

template <typename T>
void CoalescedBlockMatrixSkel::damp(std::vector<T>& data, T alpha, T beta) const {
  int64_t totData = chainData[chainData.size() - 1];
  BASPACHO_CHECK_EQ(totData, (int64_t)data.size());

  for (size_t a = 0; a < chainColPtr.size() - 1; a++) {
    int64_t aStart = lumpStart[a];
    int64_t aSize = lumpStart[a + 1] - aStart;
    int64_t colStart = chainColPtr[a];
    int64_t dataPtr = chainData[colStart];

    Eigen::Map<MatRMaj<T>> block(data.data() + dataPtr, aSize, aSize);
    block.diagonal() *= (1 + alpha);
    block.diagonal().array() += beta;
  }
}

template void CoalescedBlockMatrixSkel::densify<double>(
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>&, const double*, bool, int64_t) const;
template void CoalescedBlockMatrixSkel::densify<float>(
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>&, const float*, bool, int64_t) const;
template Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
CoalescedBlockMatrixSkel::densify<double>(const std::vector<double>&, bool) const;
template Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>
CoalescedBlockMatrixSkel::densify<float>(const std::vector<float>&, bool) const;
template void CoalescedBlockMatrixSkel::damp<double>(std::vector<double>& data, double alpha,
                                                     double beta) const;
template void CoalescedBlockMatrixSkel::damp<float>(std::vector<float>& data, float alpha,
                                                    float beta) const;

}  // end namespace BaSpaCho
