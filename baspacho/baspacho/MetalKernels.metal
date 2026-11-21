/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// ============================================================================
// Helper functions (equivalent to MathUtils.h device functions)
// ============================================================================

// Convert linear index to ordered pair (x, y) where 0 <= x <= y < n
// p varies in 0 <= p < n*(n+1)/2
inline int2 toOrderedPair(int64_t n, int64_t p) {
    int64_t odd = n & 1;
    int64_t m = n + 1 - odd;
    int64_t x = p % m;
    int64_t y = n - 1 - (p / m);
    if (x > y) {
        x = x - y - 1;
        y = n - 1 - odd - y;
    }
    return int2(int(x), int(y));
}

// Binary search: find largest i such that array[i] <= needle
inline int64_t bisect(constant int64_t* array, int64_t size, int64_t needle) {
    int64_t a = 0, b = size;
    while (b - a > 1) {
        int64_t mid = (a + b) / 2;
        if (needle >= array[mid]) {
            a = mid;
        } else {
            b = mid;
        }
    }
    return a;
}

// In-place Cholesky decomposition for small blocks
// A is row-major with stride lda
template <typename T>
inline void cholesky(device T* A, int lda, int n) {
    device T* b_ii = A;

    for (int i = 0; i < n; i++) {
        T d = sqrt(*b_ii);
        *b_ii = d;

        device T* b_ji = b_ii + lda;
        for (int j = i + 1; j < n; j++) {
            T c = *b_ji / d;
            *b_ji = c;

            device T* b_ki = b_ii + lda;
            device T* b_jk = b_ji + 1;
            for (int k = i + 1; k <= j; k++) {
                *b_jk -= c * (*b_ki);
                b_ki += lda;
                b_jk += 1;
            }

            b_ji += lda;
        }

        b_ii += lda + 1;
    }
}

// In-place solver for A^T (A built upper-diagonal col-major)
template <typename T>
inline void solveUpperT(constant T* A, int lda, int n, device T* v) {
    constant T* b_ii = A;
    for (int i = 0; i < n; i++) {
        T x = v[i];

        for (int j = 0; j < i; j++) {
            x -= b_ii[j] * v[j];
        }

        v[i] = x / b_ii[i];
        b_ii += lda;
    }
}

// In-place solver for A^T with device pointer for A
template <typename T>
inline void solveUpperT_dev(device T* A, int lda, int n, device T* v) {
    device T* b_ii = A;
    for (int i = 0; i < n; i++) {
        T x = v[i];

        for (int j = 0; j < i; j++) {
            x -= b_ii[j] * v[j];
        }

        v[i] = x / b_ii[i];
        b_ii += lda;
    }
}

// In-place solver for A (A built upper-diagonal col-major)
template <typename T>
inline void solveUpper(constant T* A, int lda, int n, device T* v) {
    constant T* b_ii = A + (lda + 1) * (n - 1);
    for (int i = n - 1; i >= 0; i--) {
        T x = v[i];

        constant T* b_ij = b_ii;
        for (int j = i + 1; j < n; j++) {
            b_ij += lda;
            x -= (*b_ij) * v[j];
        }

        v[i] = x / (*b_ii);
        b_ii -= lda + 1;
    }
}

// In-place solver for A with device pointer
template <typename T>
inline void solveUpper_dev(device T* A, int lda, int n, device T* v) {
    device T* b_ii = A + (lda + 1) * (n - 1);
    for (int i = n - 1; i >= 0; i--) {
        T x = v[i];

        device T* b_ij = b_ii;
        for (int j = i + 1; j < n; j++) {
            b_ij += lda;
            x -= (*b_ij) * v[j];
        }

        v[i] = x / (*b_ii);
        b_ii -= lda + 1;
    }
}

// ============================================================================
// Atomic operations for sparse elimination
// Metal 2.4 lacks native atomic_float, so we use CAS-based emulation
// ============================================================================

// Atomic subtract for float using compare-and-swap
inline void atomicSubFloat(device atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    uint desired;
    do {
        float current = as_type<float>(expected);
        float newVal = current - val;
        desired = as_type<uint>(newVal);
    } while (!atomic_compare_exchange_weak_explicit(addr, &expected, desired,
                                                     memory_order_relaxed,
                                                     memory_order_relaxed));
}

// Atomic add for float using compare-and-swap
inline void atomicAddFloat(device atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    uint desired;
    do {
        float current = as_type<float>(expected);
        float newVal = current + val;
        desired = as_type<uint>(newVal);
    } while (!atomic_compare_exchange_weak_explicit(addr, &expected, desired,
                                                     memory_order_relaxed,
                                                     memory_order_relaxed));
}

// Strided matrix subtraction with atomics: A -= B * C^T
// A is bRows x cRows (stride aStride), B is bRows x k (stride bStride),
// C is cRows x k (stride cStride). Same as locked_sub_product in CudaAtomic.cuh
inline void locked_sub_product_float(device float* aMat, int64_t aStride,
                                     device const float* bMat, int64_t bRows, int64_t bStride,
                                     device const float* cMat, int64_t cRows, int64_t cStride,
                                     int64_t k) {
    for (int64_t i = 0; i < bRows; i++) {
        for (int64_t j = 0; j < cRows; j++) {
            float val = 0.0f;
            for (int64_t q = 0; q < k; q++) {
                val += bMat[i * bStride + q] * cMat[j * cStride + q];
            }
            atomicSubFloat((device atomic_uint*)&aMat[i * aStride + j], val);
        }
    }
}

// ============================================================================
// Kernel 1: factor_lumps_kernel (Cholesky on diagonal blocks)
// One thread per lump
// ============================================================================
kernel void factor_lumps_kernel_float(
    constant int64_t* lumpStart [[buffer(0)]],
    constant int64_t* chainColPtr [[buffer(1)]],
    constant int64_t* chainData [[buffer(2)]],
    constant int64_t* boardColPtr [[buffer(3)]],
    constant int64_t* boardChainColOrd [[buffer(4)]],
    constant int64_t* chainRowsTillEnd [[buffer(5)]],
    device float* data [[buffer(6)]],
    constant int64_t& lumpIndexStart [[buffer(7)]],
    constant int64_t& lumpIndexEnd [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    int64_t lump = lumpIndexStart + tid;
    if (lump >= lumpIndexEnd) {
        return;
    }

    int64_t lumpSize = lumpStart[lump + 1] - lumpStart[lump];
    int64_t colStart = chainColPtr[lump];
    int64_t dataPtr = chainData[colStart];

    // In-place lower diag Cholesky on diagonal block
    device float* diagBlockPtr = data + dataPtr;
    cholesky(diagBlockPtr, int(lumpSize), int(lumpSize));

    // Below-diagonal solve
    int64_t gatheredStart = boardColPtr[lump];
    int64_t gatheredEnd = boardColPtr[lump + 1];
    int64_t rowDataStart = boardChainColOrd[gatheredStart + 1];
    int64_t rowDataEnd = boardChainColOrd[gatheredEnd - 1];
    int64_t belowDiagStart = chainData[colStart + rowDataStart];
    int64_t numRows = chainRowsTillEnd[colStart + rowDataEnd - 1]
                    - chainRowsTillEnd[colStart + rowDataStart - 1];

    device float* belowDiagBlockPtr = data + belowDiagStart;
    for (int64_t i = 0; i < numRows; i++) {
        solveUpperT_dev(diagBlockPtr, int(lumpSize), int(lumpSize), belowDiagBlockPtr);
        belowDiagBlockPtr += lumpSize;
    }
}

// ============================================================================
// Kernel 2: factor_spans_kernel (Partial block Cholesky)
// One thread per span
// ============================================================================
kernel void factor_spans_kernel_float(
    constant int64_t* spanToLump [[buffer(0)]],
    constant int64_t* spanOffsetInLump [[buffer(1)]],
    constant int64_t* lumpToSpan [[buffer(2)]],
    constant int64_t* spanStart [[buffer(3)]],
    constant int64_t* lumpStart [[buffer(4)]],
    constant int64_t* chainColPtr [[buffer(5)]],
    constant int64_t* chainData [[buffer(6)]],
    constant int64_t* boardColPtr [[buffer(7)]],
    constant int64_t* boardChainColOrd [[buffer(8)]],
    constant int64_t* chainRowsTillEnd [[buffer(9)]],
    device float* data [[buffer(10)]],
    constant int64_t& spanIndexStart [[buffer(11)]],
    constant int64_t& spanIndexEnd [[buffer(12)]],
    uint tid [[thread_position_in_grid]])
{
    int64_t span = spanIndexStart + tid;
    if (span >= spanIndexEnd) {
        return;
    }

    int64_t lump = spanToLump[span];
    int64_t spanOffInLump = spanOffsetInLump[span];
    int64_t spanIndexInLump = span - lumpToSpan[lump];
    int64_t lumpSize = lumpStart[lump + 1] - lumpStart[lump];
    int64_t spanSize = spanStart[span + 1] - spanStart[span];
    int64_t colStart = chainColPtr[lump];
    int64_t dataPtr = chainData[colStart + spanIndexInLump] + spanOffInLump;

    // In-place Cholesky of the span's diagonal block
    device float* spanDiagPtr = data + dataPtr;
    cholesky(spanDiagPtr, int(lumpSize), int(spanSize));

    // Solve all rows below this span's diagonal block (the following spans of the same
    // lump included), as in factor_spans_kernel in MatOpsCuda.cu
    int64_t gatheredEnd = boardColPtr[lump + 1];
    int64_t rowDataEnd = boardChainColOrd[gatheredEnd - 1];
    int64_t belowDiagPtr = chainData[colStart + spanIndexInLump + 1] + spanOffInLump;
    int64_t numRows = chainRowsTillEnd[colStart + rowDataEnd - 1]
                    - chainRowsTillEnd[colStart + spanIndexInLump];

    device float* belowDiagBlockPtr = data + belowDiagPtr;
    for (int64_t i = 0; i < numRows; i++) {
        solveUpperT_dev(spanDiagPtr, int(lumpSize), int(spanSize), belowDiagBlockPtr);
        belowDiagBlockPtr += lumpSize;
    }
}

// ============================================================================
// Kernel 3: sparse_elim_straight_kernel (Sparse elimination)
// One thread per block pair
// ============================================================================
kernel void sparse_elim_straight_kernel_float(
    constant int64_t* chainColPtr [[buffer(0)]],
    constant int64_t* lumpStart [[buffer(1)]],
    constant int64_t* chainRowSpan [[buffer(2)]],
    constant int64_t* spanStart [[buffer(3)]],
    constant int64_t* chainData [[buffer(4)]],
    constant int64_t* spanToLump [[buffer(5)]],
    constant int64_t* spanOffsetInLump [[buffer(6)]],
    device float* data [[buffer(7)]],
    constant int64_t& lumpIndexStart [[buffer(8)]],
    constant int64_t& lumpIndexEnd [[buffer(9)]],
    constant int64_t* makeBlockPairEnumStraight [[buffer(10)]],
    constant int64_t& numBlockPairs [[buffer(11)]],
    uint tid [[thread_position_in_grid]])
{
    if (int64_t(tid) >= numBlockPairs) {
        return;
    }

    // Find which lump this block pair belongs to
    int64_t pos = bisect(makeBlockPairEnumStraight, lumpIndexEnd - lumpIndexStart, int64_t(tid));
    int64_t l = lumpIndexStart + pos;

    int64_t startPtr = chainColPtr[l] + 1;  // skip diagonal block
    int64_t n = chainColPtr[l + 1] - startPtr;
    int2 di_dj = toOrderedPair(n, int64_t(tid) - makeBlockPairEnumStraight[pos]);
    int64_t lColSize = lumpStart[l + 1] - lumpStart[l];

    // block (si, l): siSize x lColSize, row-major
    int64_t i = startPtr + di_dj.x;
    int64_t si = chainRowSpan[i];
    int64_t siSize = spanStart[si + 1] - spanStart[si];
    device const float* ilBlock = data + chainData[i];

    // block (sj, l): sjSize x lColSize, row-major (sj >= si)
    int64_t j = startPtr + di_dj.y;
    int64_t sj = chainRowSpan[j];
    int64_t sjSize = spanStart[sj + 1] - spanStart[sj];
    device const float* jlBlock = data + chainData[j];

    // target block (sj, si) lives in the column of the lump containing si
    int64_t targetLump = spanToLump[si];
    int64_t targetSpanOffsetInLump = spanOffsetInLump[si];
    int64_t targetStartPtr = chainColPtr[targetLump];
    int64_t targetEndPtr = chainColPtr[targetLump + 1];
    int64_t targetLumpSize = lumpStart[targetLump + 1] - lumpStart[targetLump];
    int64_t tPos = bisect(chainRowSpan + targetStartPtr, targetEndPtr - targetStartPtr, sj);
    device float* jiBlock = data + chainData[targetStartPtr + tPos] + targetSpanOffsetInLump;

    // jiBlock -= jlBlock * ilBlock^T (atomically, several pairs hit the same target)
    locked_sub_product_float(jiBlock, targetLumpSize, jlBlock, sjSize, lColSize,
                             ilBlock, siSize, lColSize, lColSize);
}

// ============================================================================
// Kernel 4: assemble_kernel (Assemble rectangular sections)
// ============================================================================
kernel void assemble_kernel_float(
    constant int64_t& numBlockRows [[buffer(0)]],
    constant int64_t& numBlockCols [[buffer(1)]],
    constant int64_t& startRow [[buffer(2)]],
    constant int64_t& srcRectWidth [[buffer(3)]],
    constant int64_t& dstStride [[buffer(4)]],
    constant int64_t* pChainRowsTillEnd [[buffer(5)]],
    constant int64_t* pToSpan [[buffer(6)]],
    constant int64_t* pSpanToChainOffset [[buffer(7)]],
    constant int64_t* pSpanOffsetInLump [[buffer(8)]],
    constant float* matRectPtr [[buffer(9)]],
    device float* data [[buffer(10)]],
    uint tid [[thread_position_in_grid]])
{
    if (int64_t(tid) >= numBlockRows * numBlockCols) {
        return;
    }

    int64_t r = tid % numBlockRows;
    int64_t c = tid / numBlockRows;

    // Only process lower triangle
    if (c > r) {
        return;
    }

    // CPU ref: rBegin = pChainRowsTillEnd[r - 1] - startRow
    //          rSize = pChainRowsTillEnd[r] - startRow - rBegin
    // where startRow = pChainRowsTillEnd[-1] (element before chain start)
    // Note: When r=0, pChainRowsTillEnd[-1] - startRow = startRow - startRow = 0
    // Handle r=0 explicitly to avoid negative indexing
    int64_t rBegin = (r > 0) ? (pChainRowsTillEnd[r - 1] - startRow) : 0;
    int64_t rEnd = pChainRowsTillEnd[r] - startRow;
    int64_t rSize = rEnd - rBegin;
    int64_t rParam = pToSpan[r];
    int64_t rOffset = pSpanToChainOffset[rParam];

    // Handle c=0 explicitly to avoid negative indexing
    int64_t cStart = (c > 0) ? (pChainRowsTillEnd[c - 1] - startRow) : 0;
    int64_t cEnd = pChainRowsTillEnd[c] - startRow;
    int64_t cSize = cEnd - cStart;
    int64_t offset = rOffset + pSpanOffsetInLump[pToSpan[c]];

    // Source pointer in temporary rectangle (row-major layout)
    constant float* matRowPtr = matRectPtr + rBegin * srcRectWidth;
    constant float* src = matRowPtr + cStart;

    // Destination pointer in block matrix (row-major layout)
    device float* dst = data + offset;

    // Subtract source from destination (stridedMatSub)
    for (int64_t i = 0; i < rSize; i++) {
        for (int64_t j = 0; j < cSize; j++) {
            device atomic_uint* addr = (device atomic_uint*)&dst[i * dstStride + j];
            atomicSubFloat(addr, src[i * srcRectWidth + j]);
        }
    }
}

// ============================================================================
// Kernel 5: assembleVec_kernel (Vector assembly during solve)
// ============================================================================
kernel void assembleVec_kernel_float(
    constant int64_t* chainRowsTillEnd [[buffer(0)]],
    constant int64_t* toSpan [[buffer(1)]],
    constant int64_t* spanStarts [[buffer(2)]],
    constant float* A [[buffer(3)]],
    constant int64_t& numColItems [[buffer(4)]],
    device float* C [[buffer(5)]],
    constant int64_t& ldc [[buffer(6)]],
    constant int64_t& nRHS [[buffer(7)]],
    constant int64_t& startRow [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    if (int64_t(tid) >= numColItems) {
        return;
    }

    // CPU ref: rowOffset = chainRowsTillEnd[i - 1] - startRow
    // where startRow = chainRowsTillEnd[-1] (element before chain start)
    int64_t rowsBefore = (tid > 0) ? (chainRowsTillEnd[tid - 1] - startRow) : 0;
    int64_t rowsAfter = chainRowsTillEnd[tid] - startRow;
    int64_t blockRows = rowsAfter - rowsBefore;

    int64_t span = toSpan[tid];
    int64_t spanStart = spanStarts[span];

    // A (temp buffer) is row-major with stride nRHS
    // C (output vector) is column-major with stride ldc
    // stridedTransAdd: dst += src (transposes from row-major to col-major)
    constant float* srcPtr = A + rowsBefore * nRHS;
    device float* dstPtr = C + spanStart;

    for (int64_t rhs = 0; rhs < nRHS; rhs++) {
        for (int64_t i = 0; i < blockRows; i++) {
            dstPtr[i + rhs * ldc] += srcPtr[i * nRHS + rhs];
        }
    }
}

// ============================================================================
// Kernel 6: assembleVecT_kernel (Transposed vector assembly)
// ============================================================================
kernel void assembleVecT_kernel_float(
    constant int64_t* chainRowsTillEnd [[buffer(0)]],
    constant int64_t* toSpan [[buffer(1)]],
    constant int64_t* spanStarts [[buffer(2)]],
    constant float* C [[buffer(3)]],
    constant int64_t& numColItems [[buffer(4)]],
    device float* A [[buffer(5)]],
    constant int64_t& ldc [[buffer(6)]],
    constant int64_t& nRHS [[buffer(7)]],
    constant int64_t& startRow [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    if (int64_t(tid) >= numColItems) {
        return;
    }

    // CPU ref: rowOffset = chainRowsTillEnd[i - 1] - startRow
    // where startRow = chainRowsTillEnd[-1] (element before chain start)
    int64_t rowsBefore = (tid > 0) ? (chainRowsTillEnd[tid - 1] - startRow) : 0;
    int64_t rowsAfter = chainRowsTillEnd[tid] - startRow;
    int64_t blockRows = rowsAfter - rowsBefore;

    int64_t span = toSpan[tid];
    int64_t spanStart = spanStarts[span];

    // A (temp buffer) is row-major with stride nRHS
    // C (input vector) is column-major with stride ldc
    device float* dstPtr = A + rowsBefore * nRHS;
    constant float* srcPtr = C + spanStart;

    for (int64_t rhs = 0; rhs < nRHS; rhs++) {
        for (int64_t i = 0; i < blockRows; i++) {
            dstPtr[i * nRHS + rhs] = srcPtr[i + rhs * ldc];
        }
    }
}

// ============================================================================
// Solve kernels: sparseElim_diagSolveL
// ============================================================================
kernel void sparseElim_diagSolveL_float(
    constant int64_t* lumpStarts [[buffer(0)]],
    constant int64_t* chainColPtr [[buffer(1)]],
    constant int64_t* chainData [[buffer(2)]],
    device float* data [[buffer(3)]],
    device float* v [[buffer(4)]],
    constant int64_t& ldc [[buffer(5)]],
    constant int64_t& nRHS [[buffer(6)]],
    constant int64_t& lumpIndexStart [[buffer(7)]],
    constant int64_t& lumpIndexEnd [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    int64_t lump = lumpIndexStart + tid;
    if (lump >= lumpIndexEnd) {
        return;
    }

    int64_t lumpStart = lumpStarts[lump];
    int64_t lumpSize = lumpStarts[lump + 1] - lumpStart;
    int64_t colStart = chainColPtr[lump];
    int64_t diagDataPtr = chainData[colStart];

    device float* diagBlock = data + diagDataPtr;

    for (int64_t rhs = 0; rhs < nRHS; rhs++) {
        solveUpperT_dev(diagBlock, int(lumpSize), int(lumpSize), v + lumpStart + ldc * rhs);
    }
}

// ============================================================================
// Solve kernels: sparseElim_diagSolveLt
// ============================================================================
kernel void sparseElim_diagSolveLt_float(
    constant int64_t* lumpStarts [[buffer(0)]],
    constant int64_t* chainColPtr [[buffer(1)]],
    constant int64_t* chainData [[buffer(2)]],
    device float* data [[buffer(3)]],
    device float* v [[buffer(4)]],
    constant int64_t& ldc [[buffer(5)]],
    constant int64_t& nRHS [[buffer(6)]],
    constant int64_t& lumpIndexStart [[buffer(7)]],
    constant int64_t& lumpIndexEnd [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    int64_t lump = lumpIndexStart + tid;
    if (lump >= lumpIndexEnd) {
        return;
    }

    int64_t lumpStart = lumpStarts[lump];
    int64_t lumpSize = lumpStarts[lump + 1] - lumpStart;
    int64_t colStart = chainColPtr[lump];
    int64_t diagDataPtr = chainData[colStart];

    device float* diagBlock = data + diagDataPtr;

    for (int64_t rhs = 0; rhs < nRHS; rhs++) {
        solveUpper_dev(diagBlock, int(lumpSize), int(lumpSize), v + lumpStart + ldc * rhs);
    }
}

// ============================================================================
// Solve kernels: sparseElim_subDiagMult (after diagSolveL)
// v[rows of sub-diagonal blocks] -= block * v[lump rows], one thread per lump
// ============================================================================
kernel void sparseElim_subDiagMult_float(
    constant int64_t* lumpStarts [[buffer(0)]],
    constant int64_t* spanStarts [[buffer(1)]],
    constant int64_t* chainColPtr [[buffer(2)]],
    constant int64_t* chainRowSpan [[buffer(3)]],
    constant int64_t* chainData [[buffer(4)]],
    device const float* data [[buffer(5)]],
    device float* v [[buffer(6)]],
    constant int64_t& ldc [[buffer(7)]],
    constant int64_t& nRHS [[buffer(8)]],
    constant int64_t& lumpIndexStart [[buffer(9)]],
    constant int64_t& lumpIndexEnd [[buffer(10)]],
    uint tid [[thread_position_in_grid]])
{
    int64_t lump = lumpIndexStart + tid;
    if (lump >= lumpIndexEnd) {
        return;
    }

    int64_t lumpStart = lumpStarts[lump];
    int64_t lumpSize = lumpStarts[lump + 1] - lumpStart;
    int64_t colStart = chainColPtr[lump];
    int64_t colEnd = chainColPtr[lump + 1];

    for (int64_t colPtr = colStart + 1; colPtr < colEnd; colPtr++) {
        int64_t rowSpan = chainRowSpan[colPtr];
        int64_t rowSpanStart = spanStarts[rowSpan];
        int64_t rowSpanSize = spanStarts[rowSpan + 1] - rowSpanStart;
        device const float* block = data + chainData[colPtr];  // rowSpanSize x lumpSize
        for (int64_t c = 0; c < nRHS; c++) {
            device const float* x = v + lumpStart + c * ldc;
            for (int64_t r = 0; r < rowSpanSize; r++) {
                float val = 0.0f;
                for (int64_t q = 0; q < lumpSize; q++) {
                    val += block[r * lumpSize + q] * x[q];
                }
                atomicSubFloat((device atomic_uint*)&v[rowSpanStart + r + c * ldc], val);
            }
        }
    }
}

// ============================================================================
// Solve kernels: sparseElim_subDiagMultT (before diagSolveLt)
// v[lump rows] -= block^T * v[rows of sub-diagonal blocks], one thread per lump
// (each thread only writes the rows of its own lump, no atomics needed)
// ============================================================================
kernel void sparseElim_subDiagMultT_float(
    constant int64_t* lumpStarts [[buffer(0)]],
    constant int64_t* spanStarts [[buffer(1)]],
    constant int64_t* chainColPtr [[buffer(2)]],
    constant int64_t* chainRowSpan [[buffer(3)]],
    constant int64_t* chainData [[buffer(4)]],
    device const float* data [[buffer(5)]],
    device float* v [[buffer(6)]],
    constant int64_t& ldc [[buffer(7)]],
    constant int64_t& nRHS [[buffer(8)]],
    constant int64_t& lumpIndexStart [[buffer(9)]],
    constant int64_t& lumpIndexEnd [[buffer(10)]],
    uint tid [[thread_position_in_grid]])
{
    int64_t lump = lumpIndexStart + tid;
    if (lump >= lumpIndexEnd) {
        return;
    }

    int64_t lumpStart = lumpStarts[lump];
    int64_t lumpSize = lumpStarts[lump + 1] - lumpStart;
    int64_t colStart = chainColPtr[lump];
    int64_t colEnd = chainColPtr[lump + 1];

    for (int64_t colPtr = colStart + 1; colPtr < colEnd; colPtr++) {
        int64_t rowSpan = chainRowSpan[colPtr];
        int64_t rowSpanStart = spanStarts[rowSpan];
        int64_t rowSpanSize = spanStarts[rowSpan + 1] - rowSpanStart;
        device const float* block = data + chainData[colPtr];  // rowSpanSize x lumpSize
        for (int64_t c = 0; c < nRHS; c++) {
            device const float* q = v + rowSpanStart + c * ldc;
            device float* x = v + lumpStart + c * ldc;
            for (int64_t k = 0; k < lumpSize; k++) {
                float val = 0.0f;
                for (int64_t r = 0; r < rowSpanSize; r++) {
                    val += block[r * lumpSize + k] * q[r];
                }
                x[k] -= val;
            }
        }
    }
}

// ============================================================================
// Dense kernels: potrf / trsm / gemm on the blocks of the factor, and the triangular solves,
// gemv and symm of the solve. Blocks of the factor are row-major, vectors are column-major
// (leading dimension ldc, nRHS columns). Larger blocks are processed by the host in tiles of
// kDenseTile, combining the *_tile kernels with dense_gemm_float.
// Matrix offsets are passed in elements (not as buffer offsets) to avoid alignment constraints.
// ============================================================================

#define kDenseTile 64  // must match MatOpsMetal.mm
#define kGemmTile 16   // dense_gemm_float uses kGemmTile x kGemmTile threads (and MatOpsMetal.mm)

// Must match DenseGemmParams in MatOpsMetal.mm
struct DenseGemmParams {
    int64_t M, N, K;
    int64_t aOff, aRowStride, aColStride;  // A(i,p) = A[aOff + i * aRowStride + p * aColStride]
    int64_t bOff, bRowStride, bColStride;  // B(j,p) = B[bOff + j * bRowStride + p * bColStride]
    int64_t cOff, cRowStride, cColStride;  // C(i,j) = C[cOff + i * cRowStride + j * cColStride]
    int64_t lowerOnly;  // only compute C(i,j) with j <= i
    int64_t aSymLower;  // A is symmetric (M == K), only its lower half is read
    float alpha, beta;  // C = beta * C + alpha * A * B^T  (if beta == 0, C is not read)
};

// C = beta * C + alpha * A * B^T, each threadgroup of kGemmTile x kGemmTile threads computes a
// tile of C, staging tiles of A and B in threadgroup memory
kernel void dense_gemm_float(device const float* A [[buffer(0)]],
                             device const float* B [[buffer(1)]],
                             device float* C [[buffer(2)]],
                             constant DenseGemmParams& p [[buffer(3)]],
                             uint2 tg [[threadgroup_position_in_grid]],
                             uint2 lid [[thread_position_in_threadgroup]]) {
    threadgroup float tileA[kGemmTile][kGemmTile + 1];
    threadgroup float tileB[kGemmTile][kGemmTile + 1];

    const int64_t i0 = int64_t(tg.y) * kGemmTile;  // first row of the C tile
    const int64_t j0 = int64_t(tg.x) * kGemmTile;  // first column of the C tile
    if (p.lowerOnly && j0 > i0 + kGemmTile - 1) {
        return;  // tile entirely above the diagonal (uniform in the threadgroup)
    }

    const int64_t ty = lid.y, tx = lid.x;
    float acc = 0.0f;
    for (int64_t k0 = 0; k0 < p.K; k0 += kGemmTile) {
        int64_t kk = k0 + tx;
        int64_t ai = i0 + ty;
        float a = 0.0f;
        if (ai < p.M && kk < p.K) {
            bool upper = p.aSymLower && kk > ai;
            int64_t r = upper ? kk : ai, c = upper ? ai : kk;
            a = A[p.aOff + r * p.aRowStride + c * p.aColStride];
        }
        tileA[ty][tx] = a;
        int64_t bj = j0 + ty;
        tileB[ty][tx] = (bj < p.N && kk < p.K) ? B[p.bOff + bj * p.bRowStride + kk * p.bColStride]
                                               : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int q = 0; q < kGemmTile; q++) {
            acc += tileA[ty][q] * tileB[tx][q];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int64_t i = i0 + ty, j = j0 + tx;
    if (i < p.M && j < p.N && (!p.lowerOnly || j <= i)) {
        device float* c = C + p.cOff + i * p.cRowStride + j * p.cColStride;
        *c = (p.beta == 0.0f ? 0.0f : p.beta * *c) + p.alpha * acc;
    }
}

// In-place lower Cholesky of the n x n (n <= kDenseTile) row-major block A[aOff], stride lda,
// in a single threadgroup. Only the lower triangle is read and written. Sets *status to 1 if
// the block is not positive definite.
kernel void dense_potrf_tile_float(device float* A [[buffer(0)]],
                                   constant int64_t& aOff [[buffer(1)]],
                                   constant int64_t& lda [[buffer(2)]],
                                   constant int64_t& n64 [[buffer(3)]],
                                   device atomic_int* status [[buffer(4)]],
                                   uint lid [[thread_position_in_threadgroup]],
                                   uint nth [[threads_per_threadgroup]]) {
    threadgroup float a[kDenseTile][kDenseTile + 1];
    const int n = int(n64);
    device float* M = A + aOff;

    for (int e = int(lid); e < n * n; e += int(nth)) {
        int r = e / n, c = e % n;
        if (c <= r) {
            a[r][c] = M[r * lda + c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int j = 0; j < n; j++) {
        float d = a[j][j];
        if (lid == 0 && !(d > 0.0f)) {
            atomic_store_explicit(status, 1, memory_order_relaxed);
        }
        d = sqrt(d);
        for (int i = j + 1 + int(lid); i < n; i += int(nth)) {
            a[i][j] /= d;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            a[j][j] = d;
        }
        // trailing update of the lower triangle: a[r][c] -= a[r][j] * a[c][j], j < c <= r
        const int m = n - j - 1;
        for (int e = int(lid); e < m * m; e += int(nth)) {
            int r = j + 1 + e / m, c = j + 1 + e % m;
            if (c <= r) {
                a[r][c] -= a[r][j] * a[c][j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int e = int(lid); e < n * n; e += int(nth)) {
        int r = e / n, c = e % n;
        if (c <= r) {
            M[r * lda + c] = a[r][c];
        }
    }
}

// B := B * L^-T, for L the n x n (n <= kDenseTile) lower triangular row-major block L[lOff]
// (stride ldl) and B the numRows x n row-major block B[bOff] (stride ldb). One thread per
// row of B, L is staged in threadgroup memory.
kernel void dense_trsm_tile_float(device const float* L [[buffer(0)]],
                                  constant int64_t& lOff [[buffer(1)]],
                                  constant int64_t& ldl [[buffer(2)]],
                                  device float* B [[buffer(3)]],
                                  constant int64_t& bOff [[buffer(4)]],
                                  constant int64_t& ldb [[buffer(5)]],
                                  constant int64_t& n64 [[buffer(6)]],
                                  constant int64_t& numRows [[buffer(7)]],
                                  uint lid [[thread_position_in_threadgroup]],
                                  uint nth [[threads_per_threadgroup]],
                                  uint tgid [[threadgroup_position_in_grid]]) {
    threadgroup float l[kDenseTile][kDenseTile + 1];
    const int n = int(n64);

    for (int e = int(lid); e < n * n; e += int(nth)) {
        int r = e / n, c = e % n;
        if (c <= r) {
            l[r][c] = L[lOff + r * ldl + c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int64_t row = int64_t(tgid) * nth + lid;
    if (row < numRows) {
        device float* b = B + bOff + row * ldb;
        for (int j = 0; j < n; j++) {
            float s = b[j];
            for (int q = 0; q < j; q++) {
                s -= l[j][q] * b[q];
            }
            b[j] = s / l[j][j];
        }
    }
}

// Solves L * x = c (transpose == 0) or L^T * x = c (transpose != 0) in place, for L the
// n x n (n <= kDenseTile) lower triangular row-major block L[lOff] (stride ldl), and c the
// n x nRHS column-major block C[cOff] (stride ldc). One threadgroup per right hand side.
kernel void dense_trsv_tile_float(device const float* L [[buffer(0)]],
                                  constant int64_t& lOff [[buffer(1)]],
                                  constant int64_t& ldl [[buffer(2)]],
                                  device float* C [[buffer(3)]],
                                  constant int64_t& cOff [[buffer(4)]],
                                  constant int64_t& ldc [[buffer(5)]],
                                  constant int64_t& n64 [[buffer(6)]],
                                  constant int64_t& transpose [[buffer(7)]],
                                  uint lid [[thread_position_in_threadgroup]],
                                  uint nth [[threads_per_threadgroup]],
                                  uint tgid [[threadgroup_position_in_grid]]) {
    threadgroup float l[kDenseTile][kDenseTile + 1];
    threadgroup float x[kDenseTile];
    const int n = int(n64);
    device float* c = C + cOff + int64_t(tgid) * ldc;

    for (int e = int(lid); e < n * n; e += int(nth)) {
        int r = e / n, q = e % n;
        if (q <= r) {
            l[r][q] = L[lOff + r * ldl + q];
        }
    }
    for (int i = int(lid); i < n; i += int(nth)) {
        x[i] = c[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (transpose == 0) {
        for (int j = 0; j < n; j++) {
            if (lid == 0) {
                x[j] /= l[j][j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int i = j + 1 + int(lid); i < n; i += int(nth)) {
                x[i] -= l[i][j] * x[j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (int j = n - 1; j >= 0; j--) {
            if (lid == 0) {
                x[j] /= l[j][j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int i = int(lid); i < j; i += int(nth)) {
                x[i] -= l[j][i] * x[j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (int i = int(lid); i < n; i += int(nth)) {
        c[i] = x[i];
    }
}

// spanToChainOffset[chainRowSpan[i]] = chainData[i] for i in [begin, end), ie the map used by
// assemble_kernel_float for the target lump (written on the GPU, in order with the other work)
kernel void set_span_to_chain_offset_float(device const int64_t* chainRowSpan [[buffer(0)]],
                                           device const int64_t* chainData [[buffer(1)]],
                                           device int64_t* spanToChainOffset [[buffer(2)]],
                                           constant int64_t& begin [[buffer(3)]],
                                           constant int64_t& end [[buffer(4)]],
                                           uint tid [[thread_position_in_grid]]) {
    int64_t i = begin + int64_t(tid);
    if (i < end) {
        spanToChainOffset[chainRowSpan[i]] = chainData[i];
    }
}

// ============================================================================
// Level solve kernels (see MetalSolveLevels.h): the lumps of a level of the elimination tree are
// processed together, one threadgroup per lump. Vectors are column-major (stride ldc).
// ============================================================================

// Must match MetalLevelSolveParams in MetalSolveLevels.h
struct LevelSolveParams {
    int64_t listOff;   // the lump of threadgroup tg is levelLumps[listOff + tg]...
    int64_t useList;   // ...if useList, else it's listOff + tg
    int64_t srcBegin;  // the update of a lump in the solve with L only uses the blocks in the
    int64_t srcEnd;    // columns of the lumps in [srcBegin, srcEnd)
    int64_t ldc;
    int64_t nRHS;
    int64_t transpose;  // solve with L^T
    int64_t maxDiag;    // level_diag_solve_float skips the lumps larger than this
    int64_t rowChunk;   // level_update_float: rows of a lump per threadgroup (y = chunk index)
};

#define LEVEL_SOLVE_ARGS                                                  \
    device const int64_t* lumpStart [[buffer(0)]],                        \
    device const int64_t* spanStart [[buffer(1)]],                        \
    device const int64_t* lumpToSpan [[buffer(2)]],                       \
    device const int64_t* spanToLump [[buffer(3)]],                       \
    device const int64_t* chainColPtr [[buffer(4)]],                      \
    device const int64_t* chainRowSpan [[buffer(5)]],                     \
    device const int64_t* chainData [[buffer(6)]],                        \
    device const int64_t* rowChainPtr [[buffer(7)]],                      \
    device const int64_t* rowChainIdx [[buffer(8)]],                      \
    device const int64_t* rowChainLump [[buffer(9)]],                     \
    device const int64_t* levelLumps [[buffer(10)]],                      \
    device const float* data [[buffer(11)]],                              \
    device float* x [[buffer(12)]],                                       \
    constant LevelSolveParams& p [[buffer(13)]],                          \
    uint lid [[thread_position_in_threadgroup]],                          \
    uint nth [[threads_per_threadgroup]],                                 \
    uint2 tg [[threadgroup_position_in_grid]]

// Update of the rows [rowBegin, rowEnd) of lump `lump` with the solved lumps:
// - solve with L: x[t] -= L[t, l] * x[l], for the blocks in row t (in the columns of the lumps
//   in [srcBegin, srcEnd))
// - solve with L^T: x[l] -= L[t, l]^T * x[t], for the blocks in column l (below the diagonal)
// Each thread computes some elements of x[lump], no synchronization needed
inline void levelUpdate(device const int64_t* lumpStart, device const int64_t* spanStart,
                        device const int64_t* lumpToSpan, device const int64_t* spanToLump,
                        device const int64_t* chainColPtr, device const int64_t* chainRowSpan,
                        device const int64_t* chainData, device const int64_t* rowChainPtr,
                        device const int64_t* rowChainIdx, device const int64_t* rowChainLump,
                        device const float* data, device float* x, constant LevelSolveParams& p,
                        int64_t lump, int rowBegin, int rowEnd, uint lid, uint nth) {
    const int64_t start = lumpStart[lump];
    const int n = int(lumpStart[lump + 1] - start);
    const int nr = rowEnd - rowBegin, nRHS = int(p.nRHS);
    const int64_t ldc = p.ldc;

    if (p.transpose == 0) {
        const int64_t span0 = lumpToSpan[lump], span1 = lumpToSpan[lump + 1];
        for (int e = int(lid); e < nr * nRHS; e += int(nth)) {
            const int i = rowBegin + e % nr, k = e / nr;
            // span of row start + i (bisection on the spans of the lump)
            int64_t a = span0, b = span1;
            while (b - a > 1) {
                int64_t m = (a + b) / 2;
                if (spanStart[m] <= start + i) {
                    a = m;
                } else {
                    b = m;
                }
            }
            const int r = int(start + i - spanStart[a]);  // row in the span
            float acc = 0.0f;
            for (int64_t q = rowChainPtr[a]; q < rowChainPtr[a + 1]; q++) {
                const int64_t l = rowChainLump[q];
                if (l < p.srcBegin) {
                    continue;
                }
                if (l >= p.srcEnd) {
                    break;
                }
                const int64_t lStart = lumpStart[l];
                const int w = int(lumpStart[l + 1] - lStart);
                device const float* blk = data + chainData[rowChainIdx[q]] + r * w;
                device const float* xl = x + lStart + k * ldc;
                for (int j = 0; j < w; j++) {
                    acc += blk[j] * xl[j];
                }
            }
            x[start + i + k * ldc] -= acc;
        }
    } else {
        for (int e = int(lid); e < nr * nRHS; e += int(nth)) {
            const int i = rowBegin + e % nr, k = e / nr;
            float acc = 0.0f;
            for (int64_t c = chainColPtr[lump]; c < chainColPtr[lump + 1]; c++) {
                const int64_t s = chainRowSpan[c];
                if (spanToLump[s] <= lump) {
                    continue;  // diagonal block
                }
                const int64_t sStart = spanStart[s];
                const int ns = int(spanStart[s + 1] - sStart);
                device const float* blk = data + chainData[c] + i;  // ns x n block, column i
                device const float* xs = x + sStart + k * ldc;
                for (int r = 0; r < ns; r++) {
                    acc += blk[r * n] * xs[r];
                }
            }
            x[start + i + k * ldc] -= acc;
        }
    }
}

inline int64_t levelLump(constant LevelSolveParams& p, device const int64_t* levelLumps, uint tg) {
    return p.useList ? levelLumps[p.listOff + int64_t(tg)] : p.listOff + int64_t(tg);
}

// Update of the rows of the lumps: threadgroup (x, y) updates the y-th chunk of rowChunk rows of
// the x-th lump (see levelUpdate)
kernel void level_update_float(LEVEL_SOLVE_ARGS) {
    const int64_t lump = levelLump(p, levelLumps, tg.x);
    const int n = int(lumpStart[lump + 1] - lumpStart[lump]);
    const int rowBegin = int(tg.y) * int(p.rowChunk);
    if (rowBegin >= n) {
        return;
    }
    const int rowEnd = (n - rowBegin) < int(p.rowChunk) ? n : rowBegin + int(p.rowChunk);
    levelUpdate(lumpStart, spanStart, lumpToSpan, spanToLump, chainColPtr, chainRowSpan, chainData,
                rowChainPtr, rowChainIdx, rowChainLump, data, x, p, lump, rowBegin, rowEnd, lid,
                nth);
}

#define kLevelMaxChunk 8  // max right hand sides per pass of the diagonal solve (256 / 32)

// Solve with the diagonal block of each lump (one threadgroup per lump), of any size: L * x = x
// or L^T * x = x in place. The diagonal block is processed in tiles of up to kDenseTile rows
// (staged in threadgroup memory), and the right hand sides in chunks. In a tile each thread
// owns an element of x (in a register) and the solved values are exchanged in threadgroup memory,
// one barrier per row. The rest of the lump is then updated with the solved tile (kept in
// threadgroup memory): in the solve with L the blocks of L below the tile are staged in
// threadgroup memory (so that they are read from device memory by consecutive threads), in the
// solve with L^T the blocks on the left of the tile are read directly (by consecutive threads).
// Sizes are 32 bit integers, offsets in the vectors are 64 bit. Lumps larger than p.maxDiag
// are skipped (solved by the tiled kernels, which use many threadgroups).
kernel void level_diag_solve_float(LEVEL_SOLVE_ARGS) {
    threadgroup float lt[kDenseTile][kDenseTile + 1];  // diagonal tile, then the blocks below
    threadgroup float xt[kLevelMaxChunk][kDenseTile];  // solved tile of x
    threadgroup float piv[2][kLevelMaxChunk];

    const int64_t lump = levelLump(p, levelLumps, tg.x);
    const int64_t start = lumpStart[lump];
    const int n = int(lumpStart[lump + 1] - start);
    if (n > p.maxDiag) {
        return;  // large lump, solved by the tiled kernels
    }
    const int64_t ldc = p.ldc;
    const int nRHS = int(p.nRHS);
    const bool transpose = p.transpose != 0;
    device const float* L = data + chainData[chainColPtr[lump]];  // n x n, row-major
    device float* xl = x + start;

    const int t = int(lid), nt = int(nth);
    const int tile = nt < kDenseTile ? nt : kDenseTile;
    const int chunk = nt / tile;  // <= kLevelMaxChunk
    const int myI = t % tile, myK = t / tile;
    const int numTiles = (n + tile - 1) / tile;

    for (int k0 = 0; k0 < nRHS; k0 += chunk) {
        const int kc = (nRHS - k0) < chunk ? (nRHS - k0) : chunk;
        for (int tt = 0; tt < numTiles; tt++) {
            const int jb = (transpose ? numTiles - 1 - tt : tt) * tile;
            const int bs = (n - jb) < tile ? (n - jb) : tile;

            for (int e = t; e < bs * bs; e += nt) {
                const int r = e / bs, c = e % bs;
                if (c <= r) {
                    lt[r][c] = L[int64_t(jb + r) * n + jb + c];
                }
            }
            const bool owner = myI < bs && myK < kc;
            device float* xp = xl + (jb + myI) + int64_t(k0 + myK) * ldc;
            float v = owner ? *xp : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // piv is double buffered: the values of step j are read before the barrier of step j+1
            if (!transpose) {
                for (int j = 0; j < bs; j++) {
                    if (owner && myI == j) {
                        v /= lt[j][j];
                        piv[j & 1][myK] = v;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (owner && myI > j) {
                        v -= lt[myI][j] * piv[j & 1][myK];
                    }
                }
            } else {
                for (int j = bs - 1; j >= 0; j--) {
                    if (owner && myI == j) {
                        v /= lt[j][j];
                        piv[j & 1][myK] = v;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (owner && myI < j) {
                        v -= lt[j][myI] * piv[j & 1][myK];
                    }
                }
            }
            if (owner) {
                *xp = v;
                xt[myK][myI] = v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);  // xt is complete, lt can be reused

            // update the rest of the lump with the solved tile
            if (!transpose) {
                // x[i] -= L[i, jb:jb+bs] * x[jb:jb+bs], for i in [jb + bs, n), by blocks of rows
                for (int i0 = jb + bs; i0 < n; i0 += tile) {
                    const int rb = (n - i0) < tile ? (n - i0) : tile;
                    for (int e = t; e < rb * bs; e += nt) {
                        const int r = e / bs, c = e % bs;
                        lt[r][c] = L[int64_t(i0 + r) * n + jb + c];
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (int e = t; e < rb * kc; e += nt) {
                        const int r = e % rb, k = e / rb;
                        float acc = 0.0f;
                        for (int c = 0; c < bs; c++) {
                            acc += lt[r][c] * xt[k][c];
                        }
                        xl[i0 + r + int64_t(k0 + k) * ldc] -= acc;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);  // before restaging lt
                }
            } else {
                // x[i] -= L[jb:jb+bs, i]^T * x[jb:jb+bs], for i in [0, jb)
                for (int e = t; e < jb * kc; e += nt) {
                    const int i = e % jb, k = e / jb;
                    float acc = 0.0f;
                    for (int j = 0; j < bs; j++) {
                        acc += L[int64_t(jb + j) * n + i] * xt[k][j];
                    }
                    xl[i + int64_t(k0 + k) * ldc] -= acc;
                }
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
    }
}
