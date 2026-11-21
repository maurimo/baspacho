/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <array>
#include <atomic>
#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <typeindex>

#include "baspacho/baspacho/CoalescedBlockMatrix.h"
#include "baspacho/baspacho/DebugMacros.h"
#include "baspacho/baspacho/MatOps.h"
#include "baspacho/baspacho/MetalDefs.h"
#include "baspacho/baspacho/MetalSolveLevels.h"
#include "baspacho/baspacho/Utils.h"

namespace BaSpaCho {

using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

// Synchronization ops for Metal
struct MetalSyncOps {
  static void sync() { MetalContext::instance().synchronize(); }
};

// Symbolic elimination context for Metal
struct MetalSymElimCtx : SymElimCtx {
  MetalSymElimCtx() {}
  virtual ~MetalSymElimCtx() override {}

  int64_t numColumns;
  int64_t numBlockPairs;
  MetalMirror<int64_t> makeBlockPairEnumStraight;
};

// Forward declarations
struct MetalSymbolicCtx;

template <typename T>
struct MetalNumericCtx;

template <typename T>
struct MetalSolveCtx;

// Symbolic context for Metal operations
struct MetalSymbolicCtx : SymbolicCtx {
  MetalSymbolicCtx(const CoalescedBlockMatrixSkel& skel_, const std::vector<int64_t>& permutation)
      : skel(skel_) {
    @autoreleasepool {
      device = (__bridge id<MTLDevice>)MetalContext::instance().device();
      commandQueue = (__bridge id<MTLCommandQueue>)MetalContext::instance().commandQueue();

      // Load all skeleton data to GPU buffers
      devLumpToSpan.load(skel.lumpToSpan);
      devChainRowsTillEnd.load(skel.chainRowsTillEnd);
      devChainRowSpan.load(skel.chainRowSpan);
      devSpanOffsetInLump.load(skel.spanOffsetInLump);
      devLumpStart.load(skel.lumpStart);
      devChainColPtr.load(skel.chainColPtr);
      devChainData.load(skel.chainData);
      devBoardColPtr.load(skel.boardColPtr);
      devBoardChainColOrd.load(skel.boardChainColOrd);
      devSpanStart.load(skel.spanStart);
      devSpanToLump.load(skel.spanToLump);
      devPermutation.load(permutation);

      // blocks by rows, for the level solve (with a dummy entry if empty, buffers can't be empty)
      MetalRowChains rowChains(skel);
      if (rowChains.chain.empty()) {
        rowChains.chain.push_back(0);
        rowChains.lump.push_back(0);
      }
      devRowChainPtr.load(rowChains.ptr);
      devRowChainIdx.load(rowChains.chain);
      devRowChainLump.load(rowChains.lump);

      NSLog(@"MetalSymbolicCtx initialized");
    }
  }

  virtual ~MetalSymbolicCtx() override {}

  virtual PermutedCoalescedAccessor deviceAccessor() override {
    PermutedCoalescedAccessor retv;
    retv.init(devSpanStart.ptr(), devSpanToLump.ptr(), devLumpStart.ptr(), devSpanOffsetInLump.ptr(),
              devChainColPtr.ptr(), devChainRowSpan.ptr(), devChainData.ptr(), devPermutation.ptr());
    return retv;
  }

  virtual SymElimCtxPtr prepareElimination(int64_t lumpsBegin, int64_t lumpsEnd) override {
    MetalSymElimCtx* elim = new MetalSymElimCtx;

    vector<int64_t> makeStraight(lumpsEnd - lumpsBegin + 1);

    // For each lump, compute number of pairs contributing to elimination
    for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
      int64_t startPtr = skel.chainColPtr[l] + 1;  // skip diag block
      int64_t endPtr = skel.chainColPtr[l + 1];
      int64_t n = endPtr - startPtr;
      makeStraight[l - lumpsBegin] = n * (n + 1) / 2;
    }
    cumSumVec(makeStraight);

    elim->numColumns = lumpsEnd - lumpsBegin;
    elim->numBlockPairs = makeStraight[makeStraight.size() - 1];
    elim->makeBlockPairEnumStraight.load(makeStraight);

    return SymElimCtxPtr(elim);
  }

  virtual NumericCtxBase* createNumericCtxForType(type_index tIdx, int64_t tempBufSize,
                                                  int batchSize) override;

  virtual SolveCtxBase* createSolveCtxForType(type_index tIdx, int nRHS, int batchSize) override;

  // Levels of the solves of the lumps from `begin` on, and their lists of lumps on the GPU
  struct SolveLevels {
    MetalSolveLevels fwd, bwd;
    MetalMirror<int64_t> devFwdLumps, devBwdLumps;
  };

  // computed on first use (and kept, typically begin is 0, or the start of the dense lumps)
  const SolveLevels& solveLevels(int64_t begin) {
    std::lock_guard<std::mutex> lock(solveLevelsMutex);
    std::unique_ptr<SolveLevels>& lv = solveLevelsFrom[begin];
    if (!lv) {
      lv = std::make_unique<SolveLevels>();
      lv->fwd = MetalSolveLevels::forward(skel, begin);
      lv->bwd = MetalSolveLevels::backward(skel, begin);
      std::vector<int64_t> fwdLumps = lv->fwd.lumps, bwdLumps = lv->bwd.lumps;
      fwdLumps.push_back(0);  // not empty
      bwdLumps.push_back(0);
      lv->devFwdLumps.load(fwdLumps);
      lv->devBwdLumps.load(bwdLumps);
    }
    return *lv;
  }

  const CoalescedBlockMatrixSkel& skel;

  id<MTLDevice> device;
  id<MTLCommandQueue> commandQueue;

  // Device buffers (mirrors of skeleton data)
  MetalMirror<int64_t> devLumpToSpan;
  MetalMirror<int64_t> devChainRowsTillEnd;
  MetalMirror<int64_t> devChainRowSpan;
  MetalMirror<int64_t> devSpanOffsetInLump;
  MetalMirror<int64_t> devLumpStart;
  MetalMirror<int64_t> devChainColPtr;
  MetalMirror<int64_t> devChainData;
  MetalMirror<int64_t> devBoardColPtr;
  MetalMirror<int64_t> devBoardChainColOrd;
  MetalMirror<int64_t> devSpanStart;
  MetalMirror<int64_t> devSpanToLump;
  MetalMirror<int64_t> devPermutation;
  MetalMirror<int64_t> devRowChainPtr;
  MetalMirror<int64_t> devRowChainIdx;
  MetalMirror<int64_t> devRowChainLump;

  std::mutex solveLevelsMutex;
  std::map<int64_t, std::unique_ptr<SolveLevels>> solveLevelsFrom;
};

// Metal operations factory
struct MetalOps : Ops {
  virtual SymbolicCtxPtr createSymbolicCtx(const CoalescedBlockMatrixSkel& skel,
                                           const std::vector<int64_t>& permutation) override {
    return SymbolicCtxPtr(new MetalSymbolicCtx(skel, permutation));
  }
};

// Throw if a command buffer failed (eg. page fault or timeout in a kernel)
static void checkCommandBuffer(id<MTLCommandBuffer> cmdBuf, const char* what) {
  if (cmdBuf.status == MTLCommandBufferStatusError) {
    NSString* desc = cmdBuf.error ? cmdBuf.error.localizedDescription : @"unknown error";
    throw std::runtime_error(std::string("Metal command buffer failed (") + what +
                             "): " + desc.UTF8String);
  }
}

static std::atomic<int64_t> mpsGemmCount{0};
int64_t metalMpsGemmCount() { return mpsGemmCount.load(); }

static std::atomic<int64_t> statDispatches{0};
static std::atomic<int64_t> statCommandBuffers{0};
static std::atomic<int64_t> statGpuNanoseconds{0};
MetalStats metalStats() {
  return MetalStats{statDispatches.load(), statCommandBuffers.load(),
                    statGpuNanoseconds.load() * 1e-9};
}

// Records the GPU work of a numeric/solve context: dispatches are encoded in order in a serial
// compute encoder (so each one sees the results of the previous ones), and command buffers are
// committed in batches without waiting. The CPU waits only in flush(), called when the results
// are needed on the host: by MetalMirror::get/load (via MetalContext::synchronize, possibly from
// another thread), by finish() at the end of each Solver operation, and when the context is
// destroyed. All the methods lock the stream, so a flush from another thread never interleaves
// with the encoding. A failed command buffer is remembered: finish() reports it to the owner even
// if the flush that found it was run by another thread.
class MetalStream {
 public:
  static constexpr int kMaxDispatchesPerCommandBuffer = 256;

  explicit MetalStream(id<MTLCommandQueue> queue_) : queue(queue_) {
    MetalContext::instance().addPendingWorkFlusher(this, [this] { flush(); });
  }

  ~MetalStream() {
    // after this no other thread can flush the stream (synchronize holds the same lock while
    // flushing), so it can be destroyed
    MetalContext::instance().removePendingWorkFlusher(this);
    try {
      flush();
    } catch (const std::exception& e) {
      fprintf(stderr, "Metal: %s\n", e.what());  // destructors can't throw, see finish()
    }
  }

  MetalStream(const MetalStream&) = delete;
  MetalStream& operator=(const MetalStream&) = delete;

  // 1D grid of numThreads threads (in threadgroups of up to 256)
  void dispatch(id<MTLComputePipelineState> pipeline,
                void (^encodeBlock)(id<MTLComputeCommandEncoder>), NSUInteger numThreads) {
    if (numThreads == 0) {
      return;  // empty grid: nothing to do (and a 0-sized threadgroup is invalid)
    }
    NSUInteger groupSize = MIN(MIN(pipeline.maxTotalThreadsPerThreadgroup, 256), numThreads);
    dispatchGroups(pipeline, encodeBlock,
                   MTLSizeMake((numThreads + groupSize - 1) / groupSize, 1, 1),
                   MTLSizeMake(groupSize, 1, 1));
  }

  void dispatchGroups(id<MTLComputePipelineState> pipeline,
                      void (^encodeBlock)(id<MTLComputeCommandEncoder>), MTLSize numGroups,
                      MTLSize groupSize) {
    if (numGroups.width * numGroups.height * numGroups.depth == 0) {
      return;
    }
    NSUInteger groupThreads = groupSize.width * groupSize.height * groupSize.depth;
    if (groupThreads > pipeline.maxTotalThreadsPerThreadgroup) {
      throw std::runtime_error("Metal: threadgroup of " + std::to_string(groupThreads) +
                               " threads exceeds the pipeline limit of " +
                               std::to_string(pipeline.maxTotalThreadsPerThreadgroup));
    }
    std::lock_guard<std::recursive_mutex> lock(mutex);
    @autoreleasepool {
      if (!encoder) {
        encoder = [commandBuffer() computeCommandEncoder];
      }
      [encoder setComputePipelineState:pipeline];
      encodeBlock(encoder);
      [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:groupSize];
    }
    countDispatchLocked();
  }

  // Encodes MetalPerformanceShaders kernels with encodeBlock(commandBuffer), after the work
  // recorded so far (with the stream locked: the command buffer can't be committed meanwhile)
  void encodeMps(void (^encodeBlock)(id<MTLCommandBuffer>)) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    endEncoding();
    @autoreleasepool {
      encodeBlock(commandBuffer());
    }
    countDispatchLocked();
  }

  // Commits the recorded work (without waiting)
  void commit() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    endEncoding();
    if (cmdBuf) {
      [cmdBuf commit];
      committed.push_back(cmdBuf);
      cmdBuf = nil;
      numDispatches = 0;
    }
  }

  // Commits the recorded work and waits for its completion, throws on errors
  void flush() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    @autoreleasepool {
      commit();
      std::vector<id<MTLCommandBuffer>> toWait;
      toWait.swap(committed);
      for (id<MTLCommandBuffer> buf : toWait) {
        [buf waitUntilCompleted];
      }
      for (id<MTLCommandBuffer> buf : toWait) {
        statCommandBuffers++;
        statGpuNanoseconds += (int64_t)((buf.GPUEndTime - buf.GPUStartTime) * 1e9);
      }
      for (id<MTLCommandBuffer> buf : toWait) {
        try {
          checkCommandBuffer(buf, "MetalStream");
        } catch (const std::exception& e) {
          if (error.empty()) {
            error = e.what();
          }
          throw;
        }
      }
    }
  }

  // Completes the work and throws if any of it failed (also if found by another thread's flush)
  void finish() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    flush();
    if (!error.empty()) {
      std::string what = error;
      error.clear();
      throw std::runtime_error(what);
    }
  }

 private:
  void countDispatchLocked() {
    statDispatches++;
    if (++numDispatches >= kMaxDispatchesPerCommandBuffer) {
      commit();
    }
  }

  id<MTLCommandBuffer> commandBuffer() {
    if (!cmdBuf) {
      cmdBuf = [queue commandBuffer];
    }
    return cmdBuf;
  }

  void endEncoding() {
    if (encoder) {
      [encoder endEncoding];
      encoder = nil;
    }
  }

  std::recursive_mutex mutex;
  id<MTLCommandQueue> queue;
  id<MTLCommandBuffer> cmdBuf = nil;
  id<MTLComputeCommandEncoder> encoder = nil;
  std::vector<id<MTLCommandBuffer>> committed;
  int numDispatches = 0;
  std::string error;  // first command buffer error, reported by finish()
};

static id<MTLComputePipelineState> metalPipeline(const char* name) {
  return (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(name);
}

// A pointer registered with MetalBufferRegistry, as (buffer, byte offset)
struct MetalBufRef {
  id<MTLBuffer> buf;
  size_t off;
};

static MetalBufRef findMetalBuf(const void* ptr, const char* what) {
  auto info = MetalBufferRegistry::instance().findBuffer(ptr);
  if (!info.first) {
    throw std::runtime_error(std::string(what) + ": buffer not found (data must be allocated " +
                             "with MetalMirror)");
  }
  return MetalBufRef{(__bridge id<MTLBuffer>)info.first, info.second};
}

// Dense kernels (see MetalKernels.metal). Blocks of the factor are row-major, vectors are
// column-major. Blocks larger than kMetalDenseTile are processed in tiles.
static constexpr int64_t kMetalDenseTile = 64;  // kDenseTile in MetalKernels.metal
static constexpr int64_t kMetalGemmTile = 16;   // kGemmTile in MetalKernels.metal

// Must match DenseGemmParams in MetalKernels.metal
struct DenseGemmParams {
  int64_t M, N, K;
  int64_t aOff, aRowStride, aColStride;  // A(i,p) = A[aOff + i * aRowStride + p * aColStride]
  int64_t bOff, bRowStride, bColStride;  // B(j,p) = B[bOff + j * bRowStride + p * bColStride]
  int64_t cOff, cRowStride, cColStride;  // C(i,j) = C[cOff + i * cRowStride + j * cColStride]
  int64_t lowerOnly;  // only compute C(i,j) with j <= i
  int64_t aSymLower;  // A is symmetric (M == K), only its lower half is read
  float alpha, beta;  // C = beta * C + alpha * A * B^T  (if beta == 0, C is not read)
};
static_assert(sizeof(DenseGemmParams) == 15 * 8, "DenseGemmParams layout");

struct MetalDense {
  // C = beta * C + alpha * A * B^T (see DenseGemmParams)
  static void gemm(MetalStream& stream, const MetalBufRef& a, const MetalBufRef& b,
                   const MetalBufRef& c, const DenseGemmParams& prm) {
    if (prm.M <= 0 || prm.N <= 0) {
      return;
    }
    BASPACHO_CHECK_GT(prm.K, 0);
    const DenseGemmParams p = prm;
    stream.dispatchGroups(
        metalPipeline("dense_gemm_float"),
        ^(id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:a.buf offset:a.off atIndex:0];
          [encoder setBuffer:b.buf offset:b.off atIndex:1];
          [encoder setBuffer:c.buf offset:c.off atIndex:2];
          [encoder setBytes:&p length:sizeof(p) atIndex:3];
        },
        MTLSizeMake((p.N + kMetalGemmTile - 1) / kMetalGemmTile,
                    (p.M + kMetalGemmTile - 1) / kMetalGemmTile, 1),
        MTLSizeMake(kMetalGemmTile, kMetalGemmTile, 1));
  }

  static void potrfTile(MetalStream& stream, const MetalBufRef& a, int64_t aOff, int64_t lda,
                        int64_t n, const MetalBufRef& status) {
    id<MTLComputePipelineState> pipeline = metalPipeline("dense_potrf_tile_float");
    NSUInteger threads = MIN(256, MAX(32, ((n * n + 31) / 32) * 32));
    stream.dispatchGroups(
        pipeline,
        ^(id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:a.buf offset:a.off atIndex:0];
          [encoder setBytes:&aOff length:sizeof(int64_t) atIndex:1];
          [encoder setBytes:&lda length:sizeof(int64_t) atIndex:2];
          [encoder setBytes:&n length:sizeof(int64_t) atIndex:3];
          [encoder setBuffer:status.buf offset:status.off atIndex:4];
        },
        MTLSizeMake(1, 1, 1), MTLSizeMake(threads, 1, 1));
  }

  // B := B * L^-T, L (n x n, n <= tile) lower at lOff, B (numRows x n) at bOff
  static void trsmTile(MetalStream& stream, const MetalBufRef& l, int64_t lOff, int64_t ldl,
                       const MetalBufRef& b, int64_t bOff, int64_t ldb, int64_t n,
                       int64_t numRows) {
    if (numRows <= 0) {
      return;
    }
    const NSUInteger threads = 256;
    stream.dispatchGroups(
        metalPipeline("dense_trsm_tile_float"),
        ^(id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:l.buf offset:l.off atIndex:0];
          [encoder setBytes:&lOff length:sizeof(int64_t) atIndex:1];
          [encoder setBytes:&ldl length:sizeof(int64_t) atIndex:2];
          [encoder setBuffer:b.buf offset:b.off atIndex:3];
          [encoder setBytes:&bOff length:sizeof(int64_t) atIndex:4];
          [encoder setBytes:&ldb length:sizeof(int64_t) atIndex:5];
          [encoder setBytes:&n length:sizeof(int64_t) atIndex:6];
          [encoder setBytes:&numRows length:sizeof(int64_t) atIndex:7];
        },
        MTLSizeMake((numRows + threads - 1) / threads, 1, 1), MTLSizeMake(threads, 1, 1));
  }

  // solves L * x = c (or L^T * x = c), L (n x n, n <= tile) lower at lOff, c at cOff (n x nRHS)
  static void trsvTile(MetalStream& stream, const MetalBufRef& l, int64_t lOff, int64_t ldl,
                       const MetalBufRef& c, int64_t cOff, int64_t ldc, int64_t n, int nRHS,
                       bool transpose) {
    const int64_t transp = transpose ? 1 : 0;
    stream.dispatchGroups(
        metalPipeline("dense_trsv_tile_float"),
        ^(id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:l.buf offset:l.off atIndex:0];
          [encoder setBytes:&lOff length:sizeof(int64_t) atIndex:1];
          [encoder setBytes:&ldl length:sizeof(int64_t) atIndex:2];
          [encoder setBuffer:c.buf offset:c.off atIndex:3];
          [encoder setBytes:&cOff length:sizeof(int64_t) atIndex:4];
          [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:5];
          [encoder setBytes:&n length:sizeof(int64_t) atIndex:6];
          [encoder setBytes:&transp length:sizeof(int64_t) atIndex:7];
        },
        MTLSizeMake(nRHS, 1, 1), MTLSizeMake(kMetalDenseTile, 1, 1));
  }

  // ---- Algorithms on top of the kernels above (plain C++, until the end of MetalDense) ----

  // params for C = A * B^T, to be completed with offsets and strides
  static DenseGemmParams params(int64_t M, int64_t N, int64_t K) {
    DenseGemmParams p = {};
    p.M = M;
    p.N = N;
    p.K = K;
    p.alpha = 1.0f;
    p.beta = 0.0f;
    return p;
  }

  // temp (n x m, row-major) = B * A^T, A (m x k) and B (n x k) both row-major at offset
  static void syrkGemm(MetalStream& stream, const MetalBufRef& d, int64_t m, int64_t n, int64_t k,
                       int64_t offset, const MetalBufRef& temp) {
    DenseGemmParams p = params(n, m, k);
    p.aOff = p.bOff = offset;
    p.aRowStride = p.bRowStride = k;
    p.aColStride = p.bColStride = 1;
    p.cOff = 0;
    p.cRowStride = m;
    p.cColStride = 1;
    gemm(stream, d, d, temp, p);
  }

  // D += alpha * A * C, A the symmetric n x n block at offM (lower half stored), C and D
  // column-major n x nRHS at offC (strides ldc and ldd)
  static void symm(MetalStream& stream, const MetalBufRef& d, int64_t offM, int64_t n,
                   const MetalBufRef& c, int64_t offC, int64_t ldc, const MetalBufRef& dst,
                   int64_t ldd, int nRHS, float alpha) {
    DenseGemmParams p = params(n, nRHS, n);
    p.aOff = offM;
    p.aRowStride = n;
    p.aColStride = 1;
    p.aSymLower = 1;
    p.bOff = offC;
    p.bRowStride = ldc;
    p.bColStride = 1;
    p.cOff = offC;
    p.cRowStride = 1;
    p.cColStride = ldd;
    p.alpha = alpha;
    p.beta = 1.0f;
    gemm(stream, d, c, dst, p);
  }

  // temp (nRows x nRHS, row-major) = alpha * M * A, M the nRows x nCols block at offM, A
  // column-major nCols x nRHS at offA (stride lda)
  static void gemv(MetalStream& stream, const MetalBufRef& d, int64_t offM, int64_t nRows,
                   int64_t nCols, const MetalBufRef& a, int64_t offA, int64_t lda, int nRHS,
                   float alpha, const MetalBufRef& temp) {
    DenseGemmParams p = params(nRows, nRHS, nCols);
    p.aOff = offM;
    p.aRowStride = nCols;
    p.aColStride = 1;
    p.bOff = offA;
    p.bRowStride = lda;
    p.bColStride = 1;
    p.cOff = 0;
    p.cRowStride = nRHS;
    p.cColStride = 1;
    p.alpha = alpha;
    gemm(stream, d, a, temp, p);
  }

  // A += alpha * M^T * temp (see gemv)
  static void gemvT(MetalStream& stream, const MetalBufRef& d, int64_t offM, int64_t nRows,
                    int64_t nCols, const MetalBufRef& a, int64_t offA, int64_t lda, int nRHS,
                    float alpha, const MetalBufRef& temp) {
    DenseGemmParams p = params(nCols, nRHS, nRows);
    p.aOff = offM;
    p.aRowStride = 1;
    p.aColStride = nCols;
    p.bOff = 0;
    p.bRowStride = 1;
    p.bColStride = nRHS;
    p.cOff = offA;
    p.cRowStride = 1;
    p.cColStride = lda;
    p.alpha = alpha;
    p.beta = 1.0f;
    gemm(stream, d, temp, a, p);
  }

  // In-place lower Cholesky of the n x n row-major block at offA (right-looking, by tiles)
  static void potrf(MetalStream& stream, const MetalBufRef& d, int64_t n, int64_t offA,
                    const MetalBufRef& status) {
    for (int64_t jb = 0; jb < n; jb += kMetalDenseTile) {
      int64_t bs = std::min(kMetalDenseTile, n - jb);
      int64_t diag = offA + jb * n + jb;
      potrfTile(stream, d, diag, n, bs, status);
      int64_t rem = n - jb - bs;
      if (rem == 0) {
        break;
      }
      int64_t below = diag + bs * n;
      trsmTile(stream, d, diag, n, d, below, n, bs, rem);
      // A22 -= L21 * L21^T (lower half)
      DenseGemmParams p = params(rem, rem, bs);
      p.aOff = p.bOff = below;
      p.aRowStride = p.bRowStride = n;
      p.aColStride = p.bColStride = 1;
      p.cOff = below + bs;
      p.cRowStride = n;
      p.cColStride = 1;
      p.lowerOnly = 1;
      p.alpha = -1.0f;
      p.beta = 1.0f;
      gemm(stream, d, d, d, p);
    }
  }

  // B := B * L^-T, L the n x n diag block at offA, B the k x n block at offB (stride n)
  static void trsm(MetalStream& stream, const MetalBufRef& d, int64_t n, int64_t k, int64_t offA,
                   int64_t offB) {
    for (int64_t jb = 0; jb < n; jb += kMetalDenseTile) {
      int64_t bs = std::min(kMetalDenseTile, n - jb);
      trsmTile(stream, d, offA + jb * n + jb, n, d, offB + jb, n, bs, k);
      int64_t rem = n - jb - bs;
      if (rem == 0) {
        break;
      }
      // B2 -= X1 * L21^T
      DenseGemmParams p = params(k, rem, bs);
      p.aOff = offB + jb;
      p.aRowStride = n;
      p.aColStride = 1;
      p.bOff = offA + (jb + bs) * n + jb;
      p.bRowStride = n;
      p.bColStride = 1;
      p.cOff = offB + jb + bs;
      p.cRowStride = n;
      p.cColStride = 1;
      p.alpha = -1.0f;
      p.beta = 1.0f;
      gemm(stream, d, d, d, p);
    }
  }

  // solves L * x = c in place, L the n x n diag block at offM, c at offC (n x nRHS, stride ldc)
  static void solveL(MetalStream& stream, const MetalBufRef& d, int64_t offM, int64_t n,
                     const MetalBufRef& c, int64_t offC, int64_t ldc, int nRHS) {
    for (int64_t jb = 0; jb < n; jb += kMetalDenseTile) {
      int64_t bs = std::min(kMetalDenseTile, n - jb);
      trsvTile(stream, d, offM + jb * n + jb, n, c, offC + jb, ldc, bs, nRHS, false);
      int64_t rem = n - jb - bs;
      if (rem == 0) {
        break;
      }
      // c2 -= L21 * x1
      DenseGemmParams p = params(rem, nRHS, bs);
      p.aOff = offM + (jb + bs) * n + jb;
      p.aRowStride = n;
      p.aColStride = 1;
      p.bOff = offC + jb;
      p.bRowStride = ldc;
      p.bColStride = 1;
      p.cOff = offC + jb + bs;
      p.cRowStride = 1;
      p.cColStride = ldc;
      p.alpha = -1.0f;
      p.beta = 1.0f;
      gemm(stream, d, c, c, p);
    }
  }

  // solves L^T * x = c in place (see solveL)
  static void solveLt(MetalStream& stream, const MetalBufRef& d, int64_t offM, int64_t n,
                      const MetalBufRef& c, int64_t offC, int64_t ldc, int nRHS) {
    int64_t numTiles = (n + kMetalDenseTile - 1) / kMetalDenseTile;
    for (int64_t t = numTiles - 1; t >= 0; t--) {
      int64_t jb = t * kMetalDenseTile;
      int64_t bs = std::min(kMetalDenseTile, n - jb);
      trsvTile(stream, d, offM + jb * n + jb, n, c, offC + jb, ldc, bs, nRHS, true);
      if (jb == 0) {
        break;
      }
      // c1 -= L21^T * x2, where L21 = L[jb:jb+bs, 0:jb]
      DenseGemmParams p = params(jb, nRHS, bs);
      p.aOff = offM + jb * n;
      p.aRowStride = 1;
      p.aColStride = n;
      p.bOff = offC + jb;
      p.bRowStride = ldc;
      p.bColStride = 1;
      p.cOff = offC;
      p.cRowStride = 1;
      p.cColStride = ldc;
      p.alpha = -1.0f;
      p.beta = 1.0f;
      gemm(stream, d, c, c, p);
    }
  }
};

// Numeric context for float - Metal implementation
template <>
struct MetalNumericCtx<float> : NumericCtx<float> {
  MetalNumericCtx(MetalSymbolicCtx& sym_, int64_t tempBufSize, int64_t numSpans)
      : sym(sym_), numSpans_(numSpans), stream(sym_.commandQueue) {
    tempBuffer.resizeToAtLeast(tempBufSize);
    devSpanToChainOffset.resizeToAtLeast(numSpans);
    potrfStatus.load(std::vector<int32_t>{0});
  }

  // the stream is flushed by its destructor (errors are only logged there, see finish()). A block
  // that is not positive definite gives NaN in the factor (as on the other backends), potrfStatus
  // is only informative
  virtual ~MetalNumericCtx() override {}

  virtual void finish() override { stream.finish(); }

  virtual void pseudoFactorSpans(float* data, int64_t spanBegin, int64_t spanEnd) override {
    @autoreleasepool {
      // Find the MTLBuffer for data
      auto bufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      if (!bufferInfo.first) {
        throw std::runtime_error("MetalNumericCtx<float>::pseudoFactorSpans: data buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)bufferInfo.first;
      size_t dataBaseOffset = bufferInfo.second;

      // Get pipeline state for factor_spans_kernel_float
      id<MTLComputePipelineState> pipeline =
          (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
              "factor_spans_kernel_float");

      int64_t numSpans = spanEnd - spanBegin;
      if (numSpans <= 0) return;

      stream.dispatch(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanToLump.buffer() offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanOffsetInLump.buffer()
                        offset:0
                       atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpToSpan.buffer() offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer() offset:0 atIndex:3];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer() offset:0 atIndex:4];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                        offset:0
                       atIndex:5];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer() offset:0 atIndex:6];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devBoardColPtr.buffer()
                        offset:0
                       atIndex:7];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devBoardChainColOrd.buffer()
                        offset:0
                       atIndex:8];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowsTillEnd.buffer()
                        offset:0
                       atIndex:9];
            [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:10];
            [encoder setBytes:&spanBegin length:sizeof(int64_t) atIndex:11];
            [encoder setBytes:&spanEnd length:sizeof(int64_t) atIndex:12];
          },
          (NSUInteger)numSpans);
    }
  }

  virtual void doElimination(const SymElimCtx& elimData, float* data, int64_t lumpsBegin,
                             int64_t lumpsEnd) override {
    @autoreleasepool {
      const MetalSymElimCtx* pElim = dynamic_cast<const MetalSymElimCtx*>(&elimData);
      BASPACHO_CHECK_NOTNULL(pElim);
      const MetalSymElimCtx& elim = *pElim;

      // Find the MTLBuffer for data
      auto bufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      if (!bufferInfo.first) {
        throw std::runtime_error("MetalNumericCtx<float>::doElimination: data buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)bufferInfo.first;
      size_t dataBaseOffset = bufferInfo.second;

      int64_t numLumps = lumpsEnd - lumpsBegin;
      if (numLumps <= 0) return;

      // Step 1: Factor lumps (Cholesky on diagonal blocks + below-diagonal solve)
      {
        id<MTLComputePipelineState> pipeline =
            (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
                "factor_lumps_kernel_float");

        stream.dispatch(
            pipeline,
            ^(id<MTLComputeCommandEncoder> encoder) {
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer()
                          offset:0
                         atIndex:0];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                          offset:0
                         atIndex:1];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer()
                          offset:0
                         atIndex:2];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devBoardColPtr.buffer()
                          offset:0
                         atIndex:3];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devBoardChainColOrd.buffer()
                          offset:0
                         atIndex:4];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowsTillEnd.buffer()
                          offset:0
                         atIndex:5];
              [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:6];
              [encoder setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:7];
              [encoder setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:8];
            },
            (NSUInteger)numLumps);
      }

      // Step 2: Sparse elimination
      if (elim.numBlockPairs > 0) {
        id<MTLComputePipelineState> pipeline =
            (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
                "sparse_elim_straight_kernel_float");

        stream.dispatch(
            pipeline,
            ^(id<MTLComputeCommandEncoder> encoder) {
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                          offset:0
                         atIndex:0];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer()
                          offset:0
                         atIndex:1];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer()
                          offset:0
                         atIndex:2];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer()
                          offset:0
                         atIndex:3];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer()
                          offset:0
                         atIndex:4];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanToLump.buffer()
                          offset:0
                         atIndex:5];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanOffsetInLump.buffer()
                          offset:0
                         atIndex:6];
              [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:7];
              [encoder setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:8];
              [encoder setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:9];
              [encoder setBuffer:(__bridge id<MTLBuffer>)elim.makeBlockPairEnumStraight.buffer()
                          offset:0
                         atIndex:10];
              [encoder setBytes:&elim.numBlockPairs length:sizeof(int64_t) atIndex:11];
            },
            (NSUInteger)elim.numBlockPairs);
      }
    }
  }

  virtual void potrf(int64_t n, float* data, int64_t offA) override {
    if (n <= 0) return;
    MetalBufRef d = findMetalBuf(data, "MetalNumericCtx<float>::potrf");
    MetalBufRef status{(__bridge id<MTLBuffer>)potrfStatus.buffer(), 0};
    MetalDense::potrf(stream, d, n, offA, status);
  }

  virtual void trsm(int64_t n, int64_t k, float* data, int64_t offA, int64_t offB) override {
    if (n <= 0 || k <= 0) return;
    MetalBufRef d = findMetalBuf(data, "MetalNumericCtx<float>::trsm");
    MetalDense::trsm(stream, d, n, k, offA, offB);
  }

  // tempBuffer (n x m, row-major) = B * A^T, A (m x k) and B (n x k) both starting at offset
  virtual void saveSyrkGemm(int64_t m, int64_t n, int64_t k, const float* data,
                            int64_t offset) override {
    @autoreleasepool {
      if (m <= 0 || n <= 0 || k <= 0) return;

      // (no need to wait: a reallocated buffer is retained by the command buffers using it,
      // and the contents of tempBuffer are only used by the following assemble)
      tempBuffer.resizeToAtLeast(m * n);
      MetalBufRef d = findMetalBuf(data, "MetalNumericCtx<float>::saveSyrkGemm");
      MetalBufRef temp{(__bridge id<MTLBuffer>)tempBuffer.buffer(), 0};

      // MetalPerformanceShaders for larger matrices, as its dispatch overhead makes it slower
      // than the simple kernel for small matrices
      if (m * n * k >= kMetalMpsGemmMinOps) {
        // MPS uses row-major storage, and computes
        //   result = alpha * op(left) * op(right) + beta * result
        // We want C(n,m) = B(n,k) * A^T(k,m), so: left=B, right=A with transposeRight=YES
        MPSMatrixDescriptor* descA =
            [MPSMatrixDescriptor matrixDescriptorWithRows:m
                                                  columns:k
                                                 rowBytes:k * sizeof(float)
                                                 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* descB =
            [MPSMatrixDescriptor matrixDescriptorWithRows:n
                                                  columns:k
                                                 rowBytes:k * sizeof(float)
                                                 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* descC =
            [MPSMatrixDescriptor matrixDescriptorWithRows:n
                                                  columns:m
                                                 rowBytes:m * sizeof(float)
                                                 dataType:MPSDataTypeFloat32];

        MPSMatrix* mpsA = [[MPSMatrix alloc] initWithBuffer:d.buf
                                                     offset:d.off + offset * sizeof(float)
                                                 descriptor:descA];
        MPSMatrix* mpsB = [[MPSMatrix alloc] initWithBuffer:d.buf
                                                     offset:d.off + offset * sizeof(float)
                                                 descriptor:descB];
        MPSMatrix* mpsC = [[MPSMatrix alloc] initWithBuffer:temp.buf offset:0 descriptor:descC];

        MPSMatrixMultiplication* gemm =
            [[MPSMatrixMultiplication alloc] initWithDevice:sym.device
                                              transposeLeft:NO
                                             transposeRight:YES
                                                 resultRows:n
                                              resultColumns:m
                                            interiorColumns:k
                                                      alpha:1.0
                                                       beta:0.0];

        stream.encodeMps(^(id<MTLCommandBuffer> cmdBuf) {
          [gemm encodeToCommandBuffer:cmdBuf leftMatrix:mpsB rightMatrix:mpsA resultMatrix:mpsC];
        });
        mpsGemmCount++;
      } else {
        MetalDense::syrkGemm(stream, d, m, n, k, offset, temp);
      }
    }
  }

  // Sets the span -> chain data offset map of the target lump, used by assemble. It is written
  // by a kernel, so that it is ordered with the assemble calls of the previous target lumps
  virtual void prepareAssemble(int64_t targetLump) override {
    @autoreleasepool {
      int64_t begin = sym.skel.chainColPtr[targetLump];
      int64_t end = sym.skel.chainColPtr[targetLump + 1];
      stream.dispatch(
          metalPipeline("set_span_to_chain_offset_float"),
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer()
                        offset:0
                       atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer()
                        offset:0
                       atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)devSpanToChainOffset.buffer()
                        offset:0
                       atIndex:2];
            [encoder setBytes:&begin length:sizeof(int64_t) atIndex:3];
            [encoder setBytes:&end length:sizeof(int64_t) atIndex:4];
          },
          (NSUInteger)(end - begin));
    }
  }

  virtual void assemble(float* data, int64_t rectRowBegin, int64_t dstStride,
                        int64_t srcColDataOffset, int64_t srcRectWidth, int64_t numBlockRows,
                        int64_t numBlockCols) override {
    @autoreleasepool {
      if (numBlockRows <= 0 || numBlockCols <= 0) return;

      // Find the MTLBuffer for data
      auto bufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      if (!bufferInfo.first) {
        throw std::runtime_error("MetalNumericCtx<float>::assemble: data buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)bufferInfo.first;
      size_t dataBaseOffset = bufferInfo.second;

      // Get pipeline state
      id<MTLComputePipelineState> pipeline =
          (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
              "assemble_kernel_float");

      int64_t numThreads = numBlockRows * numBlockCols;

      // Compute startRow = chainRowsTillEnd[srcColDataOffset - 1] (element before chain start)
      // This matches CPU ref where startRow = pChainRowsTillEnd[-1] after offsetting the pointer
      int64_t startRow = (srcColDataOffset > 0) ? sym.skel.chainRowsTillEnd[srcColDataOffset - 1] : 0;

      stream.dispatch(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBytes:&numBlockRows length:sizeof(int64_t) atIndex:0];
            [encoder setBytes:&numBlockCols length:sizeof(int64_t) atIndex:1];
            [encoder setBytes:&startRow length:sizeof(int64_t) atIndex:2];
            [encoder setBytes:&srcRectWidth length:sizeof(int64_t) atIndex:3];
            [encoder setBytes:&dstStride length:sizeof(int64_t) atIndex:4];
            // pChainRowsTillEnd offset by srcColDataOffset
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowsTillEnd.buffer()
                        offset:srcColDataOffset * sizeof(int64_t)
                       atIndex:5];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer()
                        offset:srcColDataOffset * sizeof(int64_t)
                       atIndex:6];
            [encoder setBuffer:(__bridge id<MTLBuffer>)devSpanToChainOffset.buffer()
                        offset:0
                       atIndex:7];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanOffsetInLump.buffer()
                        offset:0
                       atIndex:8];
            [encoder setBuffer:(__bridge id<MTLBuffer>)tempBuffer.buffer() offset:0 atIndex:9];
            [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:10];
          },
          (NSUInteger)numThreads);
    }
  }

  MetalSymbolicCtx& sym;
  int64_t numSpans_;
  MetalMirror<float> tempBuffer;
  MetalMirror<int64_t> devSpanToChainOffset;
  MetalMirror<int32_t> potrfStatus;  // set to 1 by potrf on non positive definite blocks
  MetalStream stream;                // last: destroyed (and flushed) before the buffers
};

// Solve context for float - Metal implementation
template <>
struct MetalSolveCtx<float> : SolveCtx<float> {
  MetalSolveCtx(MetalSymbolicCtx& sym_, int nRHS_)
      : sym(sym_), nRHS(nRHS_), stream(sym_.commandQueue) {
    // Allocate temp buffer for vector assembly
    tempVecBuffer.resizeToAtLeast(sym.skel.order() * nRHS);
  }

  virtual ~MetalSolveCtx() override {}

  virtual void finish() override { stream.finish(); }

  virtual void sparseElimSolveL(const SymElimCtx& elimData, const float* data, int64_t lumpsBegin,
                                int64_t lumpsEnd, float* C, int64_t ldc) override {
    @autoreleasepool {
      const MetalSymElimCtx* pElim = dynamic_cast<const MetalSymElimCtx*>(&elimData);
      BASPACHO_CHECK_NOTNULL(pElim);

      // Find buffers
      auto dataBufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      auto cBufferInfo = MetalBufferRegistry::instance().findBuffer(C);
      if (!dataBufferInfo.first || !cBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::sparseElimSolveL: buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)dataBufferInfo.first;
      id<MTLBuffer> cBuffer = (__bridge id<MTLBuffer>)cBufferInfo.first;
      size_t dataOffset = dataBufferInfo.second;
      size_t cOffset = cBufferInfo.second;

      int64_t numLumps = lumpsEnd - lumpsBegin;
      if (numLumps <= 0) return;

      // Dispatch diagonal solve kernel
      id<MTLComputePipelineState> pipeline =
          (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
              "sparseElim_diagSolveL_float");

      int64_t nRHS64 = nRHS;
      stream.dispatch(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer() offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                        offset:0
                       atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer() offset:0 atIndex:2];
            [encoder setBuffer:dataBuffer offset:dataOffset atIndex:3];
            [encoder setBuffer:cBuffer offset:cOffset atIndex:4];
            [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:5];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:6];
            [encoder setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:7];
            [encoder setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:8];
          },
          (NSUInteger)numLumps);

      // Sub-diagonal update: v[below] -= L[below, lump] * v[lump]
      {
        id<MTLComputePipelineState> subPipeline =
            (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
                "sparseElim_subDiagMult_float");
        stream.dispatch(
            subPipeline,
            ^(id<MTLComputeCommandEncoder> encoder) {
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer()
                          offset:0
                         atIndex:0];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer()
                          offset:0
                         atIndex:1];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                          offset:0
                         atIndex:2];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer()
                          offset:0
                         atIndex:3];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer()
                          offset:0
                         atIndex:4];
              [encoder setBuffer:dataBuffer offset:dataOffset atIndex:5];
              [encoder setBuffer:cBuffer offset:cOffset atIndex:6];
              [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:7];
              [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:8];
              [encoder setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:9];
              [encoder setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:10];
            },
            (NSUInteger)numLumps);
      }
    }
  }

  virtual void sparseElimSolveLt(const SymElimCtx& elimData, const float* data, int64_t lumpsBegin,
                                 int64_t lumpsEnd, float* C, int64_t ldc) override {
    @autoreleasepool {
      const MetalSymElimCtx* pElim = dynamic_cast<const MetalSymElimCtx*>(&elimData);
      BASPACHO_CHECK_NOTNULL(pElim);

      // Find buffers
      auto dataBufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      auto cBufferInfo = MetalBufferRegistry::instance().findBuffer(C);
      if (!dataBufferInfo.first || !cBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::sparseElimSolveLt: buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)dataBufferInfo.first;
      id<MTLBuffer> cBuffer = (__bridge id<MTLBuffer>)cBufferInfo.first;
      size_t dataOffset = dataBufferInfo.second;
      size_t cOffset = cBufferInfo.second;

      int64_t numLumps = lumpsEnd - lumpsBegin;
      if (numLumps <= 0) return;

      int64_t nRHS64 = nRHS;

      // Sub-diagonal update: v[lump] -= L[below, lump]^T * v[below]
      {
        id<MTLComputePipelineState> subPipeline =
            (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
                "sparseElim_subDiagMultT_float");
        stream.dispatch(
            subPipeline,
            ^(id<MTLComputeCommandEncoder> encoder) {
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer()
                          offset:0
                         atIndex:0];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer()
                          offset:0
                         atIndex:1];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                          offset:0
                         atIndex:2];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer()
                          offset:0
                         atIndex:3];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer()
                          offset:0
                         atIndex:4];
              [encoder setBuffer:dataBuffer offset:dataOffset atIndex:5];
              [encoder setBuffer:cBuffer offset:cOffset atIndex:6];
              [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:7];
              [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:8];
              [encoder setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:9];
              [encoder setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:10];
            },
            (NSUInteger)numLumps);
      }

      // Dispatch diagonal solve kernel
      id<MTLComputePipelineState> pipeline =
          (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
              "sparseElim_diagSolveLt_float");

      stream.dispatch(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer() offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                        offset:0
                       atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer() offset:0 atIndex:2];
            [encoder setBuffer:dataBuffer offset:dataOffset atIndex:3];
            [encoder setBuffer:cBuffer offset:cOffset atIndex:4];
            [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:5];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:6];
            [encoder setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:7];
            [encoder setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:8];
          },
          (NSUInteger)numLumps);
    }
  }

  // Solves with L (or L^T) the lumps in [lumpsBegin, lumpsEnd) level by level, see
  // MetalSolveLevels.h: a couple of dispatches per level, instead of a few per lump
  bool levelSolve(const float* data, int64_t lumpsBegin, int64_t lumpsEnd, float* C, int64_t ldc,
                  bool transpose) {
    if (lumpsBegin >= lumpsEnd || nRHS <= 0) {
      return true;
    }
    const MetalSymbolicCtx::SolveLevels& lv = sym.solveLevels(lumpsBegin);
    MetalBufRef d = findMetalBuf(data, "MetalSolveCtx<float>::levelSolve");
    MetalBufRef c = findMetalBuf(C, "MetalSolveCtx<float>::levelSolve");
    id<MTLBuffer> lumpsBuf =
        (__bridge id<MTLBuffer>)(transpose ? lv.devBwdLumps.buffer() : lv.devFwdLumps.buffer());
    const std::array<id<MTLBuffer>, 10> idx = {
        (__bridge id<MTLBuffer>)sym.devLumpStart.buffer(),
        (__bridge id<MTLBuffer>)sym.devSpanStart.buffer(),
        (__bridge id<MTLBuffer>)sym.devLumpToSpan.buffer(),
        (__bridge id<MTLBuffer>)sym.devSpanToLump.buffer(),
        (__bridge id<MTLBuffer>)sym.devChainColPtr.buffer(),
        (__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer(),
        (__bridge id<MTLBuffer>)sym.devChainData.buffer(),
        (__bridge id<MTLBuffer>)sym.devRowChainPtr.buffer(),
        (__bridge id<MTLBuffer>)sym.devRowChainIdx.buffer(),
        (__bridge id<MTLBuffer>)sym.devRowChainLump.buffer(),
    };
    id<MTLComputePipelineState> updatePipeline = metalPipeline("level_update_float");
    id<MTLComputePipelineState> diagPipeline = metalPipeline("level_diag_solve_float");
    metalLevelSolve(
        sym.skel, transpose ? lv.bwd : lv.fwd, transpose, lumpsBegin, lumpsEnd, ldc, nRHS,
        [&](MetalLevelKernel kernel, const MetalLevelSolveParams& params, int64_t numGroups,
            int64_t numThreads) {
          const MetalLevelSolveParams prm = params;
          stream.dispatchGroups(
              kernel == MetalLevelKernel::Update ? updatePipeline : diagPipeline,
              ^(id<MTLComputeCommandEncoder> encoder) {
                for (size_t i = 0; i < idx.size(); i++) {
                  [encoder setBuffer:idx[i] offset:0 atIndex:i];
                }
                [encoder setBuffer:lumpsBuf offset:0 atIndex:10];
                [encoder setBuffer:d.buf offset:d.off atIndex:11];
                [encoder setBuffer:c.buf offset:c.off atIndex:12];
                [encoder setBytes:&prm length:sizeof(prm) atIndex:13];
              },
              MTLSizeMake(numGroups, 1, 1), MTLSizeMake(numThreads, 1, 1));
        });
    return true;
  }

  virtual bool solveLRange(const float* data, int64_t lumpsBegin, int64_t lumpsEnd, float* C,
                           int64_t ldc) override {
    return levelSolve(data, lumpsBegin, lumpsEnd, C, ldc, /* transpose = */ false);
  }

  virtual bool solveLtRange(const float* data, int64_t lumpsBegin, int64_t lumpsEnd, float* C,
                            int64_t ldc) override {
    return levelSolve(data, lumpsBegin, lumpsEnd, C, ldc, /* transpose = */ true);
  }

  // D += alpha * A * C, A the symmetric n x n block at offset (lower half stored)
  virtual void symm(const float* data, int64_t offset, int64_t n, const float* C, int64_t offC,
                    int64_t ldc, float* D, int64_t ldd, float alpha) override {
    if (n <= 0 || nRHS <= 0) return;
    MetalBufRef d = findMetalBuf(data, "MetalSolveCtx<float>::symm");
    MetalBufRef c = findMetalBuf(C, "MetalSolveCtx<float>::symm");
    MetalBufRef dst = findMetalBuf(D, "MetalSolveCtx<float>::symm");
    MetalDense::symm(stream, d, offset, n, c, offC, ldc, dst, ldd, nRHS, alpha);
  }

  virtual void solveL(const float* data, int64_t offset, int64_t n, float* C, int64_t offC,
                      int64_t ldc) override {
    if (n <= 0 || nRHS <= 0) return;
    MetalBufRef d = findMetalBuf(data, "MetalSolveCtx<float>::solveL");
    MetalBufRef c = findMetalBuf(C, "MetalSolveCtx<float>::solveL");
    MetalDense::solveL(stream, d, offset, n, c, offC, ldc, nRHS);
  }

  // tempVecBuffer (nRows x nRHS, row-major) = alpha * M * A, M the nRows x nCols block at offset
  virtual void gemv(const float* data, int64_t offset, int64_t nRows, int64_t nCols, const float* A,
                    int64_t offA, int64_t lda, float alpha) override {
    if (nRows <= 0 || nCols <= 0 || nRHS <= 0) return;
    tempVecBuffer.resizeToAtLeast(nRows * nRHS);
    MetalBufRef d = findMetalBuf(data, "MetalSolveCtx<float>::gemv");
    MetalBufRef a = findMetalBuf(A, "MetalSolveCtx<float>::gemv");
    MetalBufRef temp{(__bridge id<MTLBuffer>)tempVecBuffer.buffer(), 0};
    MetalDense::gemv(stream, d, offset, nRows, nCols, a, offA, lda, nRHS, alpha, temp);
  }

  virtual void assembleVec(int64_t chainColPtr, int64_t numColItems, float* C, int64_t ldc) override {
    @autoreleasepool {
      if (numColItems <= 0) return;

      auto cBufferInfo = MetalBufferRegistry::instance().findBuffer(C);
      if (!cBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::assembleVec: buffer not found");
      }
      id<MTLBuffer> cBuffer = (__bridge id<MTLBuffer>)cBufferInfo.first;
      size_t cOffset = cBufferInfo.second;

      id<MTLComputePipelineState> pipeline =
          (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
              "assembleVec_kernel_float");

      // startRow = chainRowsTillEnd[chainColPtr - 1] (element before chain start)
      int64_t startRow =
          (chainColPtr > 0) ? sym.devChainRowsTillEnd.ptr()[chainColPtr - 1] : 0;

      int64_t nRHS64 = nRHS;
      stream.dispatch(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowsTillEnd.buffer()
                        offset:chainColPtr * sizeof(int64_t)
                       atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer()
                        offset:chainColPtr * sizeof(int64_t)
                       atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer() offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)tempVecBuffer.buffer() offset:0 atIndex:3];
            [encoder setBytes:&numColItems length:sizeof(int64_t) atIndex:4];
            [encoder setBuffer:cBuffer offset:cOffset atIndex:5];
            [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:6];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:7];
            [encoder setBytes:&startRow length:sizeof(int64_t) atIndex:8];
          },
          (NSUInteger)numColItems);
    }
  }

  virtual void solveLt(const float* data, int64_t offset, int64_t n, float* C, int64_t offC,
                       int64_t ldc) override {
    if (n <= 0 || nRHS <= 0) return;
    MetalBufRef d = findMetalBuf(data, "MetalSolveCtx<float>::solveLt");
    MetalBufRef c = findMetalBuf(C, "MetalSolveCtx<float>::solveLt");
    MetalDense::solveLt(stream, d, offset, n, c, offC, ldc, nRHS);
  }

  // A += alpha * M^T * tempVecBuffer, M the nRows x nCols block at offset
  virtual void gemvT(const float* data, int64_t offset, int64_t nRows, int64_t nCols, float* A,
                     int64_t offA, int64_t lda, float alpha) override {
    if (nRows <= 0 || nCols <= 0 || nRHS <= 0) return;
    MetalBufRef d = findMetalBuf(data, "MetalSolveCtx<float>::gemvT");
    MetalBufRef a = findMetalBuf(A, "MetalSolveCtx<float>::gemvT");
    MetalBufRef temp{(__bridge id<MTLBuffer>)tempVecBuffer.buffer(), 0};
    MetalDense::gemvT(stream, d, offset, nRows, nCols, a, offA, lda, nRHS, alpha, temp);
  }

  virtual void assembleVecT(const float* C, int64_t ldc, int64_t chainColPtr,
                            int64_t numColItems) override {
    @autoreleasepool {
      if (numColItems <= 0) return;

      auto cBufferInfo = MetalBufferRegistry::instance().findBuffer(C);
      if (!cBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::assembleVecT: buffer not found");
      }
      id<MTLBuffer> cBuffer = (__bridge id<MTLBuffer>)cBufferInfo.first;
      size_t cOffset = cBufferInfo.second;

      id<MTLComputePipelineState> pipeline =
          (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
              "assembleVecT_kernel_float");

      // startRow = chainRowsTillEnd[chainColPtr - 1] (element before chain start)
      int64_t startRow =
          (chainColPtr > 0) ? sym.devChainRowsTillEnd.ptr()[chainColPtr - 1] : 0;

      int64_t nRHS64 = nRHS;
      stream.dispatch(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowsTillEnd.buffer()
                        offset:chainColPtr * sizeof(int64_t)
                       atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer()
                        offset:chainColPtr * sizeof(int64_t)
                       atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer() offset:0 atIndex:2];
            [encoder setBuffer:cBuffer offset:cOffset atIndex:3];
            [encoder setBytes:&numColItems length:sizeof(int64_t) atIndex:4];
            [encoder setBuffer:(__bridge id<MTLBuffer>)tempVecBuffer.buffer() offset:0 atIndex:5];
            [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:6];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:7];
            [encoder setBytes:&startRow length:sizeof(int64_t) atIndex:8];
          },
          (NSUInteger)numColItems);
    }
  }

  MetalSymbolicCtx& sym;
  int nRHS;
  MetalMirror<float> tempVecBuffer;
  MetalStream stream;  // last: destroyed (and flushed) before the buffers
};

// Batched numeric context for float
template <>
struct MetalNumericCtx<std::vector<float*>> : NumericCtx<std::vector<float*>> {
  MetalNumericCtx(MetalSymbolicCtx& sym_, int64_t tempBufSize, int64_t numSpans, int batchSize_)
      : sym(sym_), numSpans_(numSpans), batchSize(batchSize_) {
    tempBuffer.resizeToAtLeast(tempBufSize * batchSize);
  }

  virtual ~MetalNumericCtx() override {}

  virtual void pseudoFactorSpans(std::vector<float*>* data, int64_t spanBegin,
                                 int64_t spanEnd) override {
    throw std::runtime_error(
        "MetalNumericCtx<std::vector<float*>>::pseudoFactorSpans not yet implemented");
  }

  virtual void doElimination(const SymElimCtx& elimData, std::vector<float*>* data,
                             int64_t lumpsBegin, int64_t lumpsEnd) override {
    throw std::runtime_error(
        "MetalNumericCtx<std::vector<float*>>::doElimination not yet implemented");
  }

  virtual void potrf(int64_t n, std::vector<float*>* data, int64_t offA) override {
    throw std::runtime_error("MetalNumericCtx<std::vector<float*>>::potrf not yet implemented");
  }

  virtual void trsm(int64_t n, int64_t k, std::vector<float*>* data, int64_t offA,
                    int64_t offB) override {
    throw std::runtime_error("MetalNumericCtx<std::vector<float*>>::trsm not yet implemented");
  }

  virtual void saveSyrkGemm(int64_t m, int64_t n, int64_t k, const std::vector<float*>* data,
                            int64_t offset) override {
    throw std::runtime_error(
        "MetalNumericCtx<std::vector<float*>>::saveSyrkGemm not yet implemented");
  }

  virtual void prepareAssemble(int64_t targetLump) override {}

  virtual void assemble(std::vector<float*>* data, int64_t rectRowBegin, int64_t dstStride,
                        int64_t srcColDataOffset, int64_t srcRectWidth, int64_t numBlockRows,
                        int64_t numBlockCols) override {
    throw std::runtime_error("MetalNumericCtx<std::vector<float*>>::assemble not yet implemented");
  }

  MetalSymbolicCtx& sym;
  int64_t numSpans_;
  int batchSize;
  MetalMirror<float> tempBuffer;
};

// Batched solve context for float
template <>
struct MetalSolveCtx<std::vector<float*>> : SolveCtx<std::vector<float*>> {
  MetalSolveCtx(MetalSymbolicCtx& sym_, int nRHS_, int batchSize_)
      : sym(sym_), nRHS(nRHS_), batchSize(batchSize_) {}

  virtual ~MetalSolveCtx() override {}

  virtual void sparseElimSolveL(const SymElimCtx& elimData, const std::vector<float*>* data,
                                int64_t lumpsBegin, int64_t lumpsEnd, std::vector<float*>* C,
                                int64_t ldc) override {
    throw std::runtime_error(
        "MetalSolveCtx<std::vector<float*>>::sparseElimSolveL not yet implemented");
  }

  virtual void sparseElimSolveLt(const SymElimCtx& elimData, const std::vector<float*>* data,
                                 int64_t lumpsBegin, int64_t lumpsEnd, std::vector<float*>* C,
                                 int64_t ldc) override {
    throw std::runtime_error(
        "MetalSolveCtx<std::vector<float*>>::sparseElimSolveLt not yet implemented");
  }

  virtual void symm(const std::vector<float*>* data, int64_t offset, int64_t n,
                    const std::vector<float*>* C, int64_t offC, int64_t ldc, std::vector<float*>* D,
                    int64_t ldd, float alpha) override {
    throw std::runtime_error("MetalSolveCtx<std::vector<float*>>::symm not yet implemented");
  }

  virtual void solveL(const std::vector<float*>* data, int64_t offset, int64_t n,
                      std::vector<float*>* C, int64_t offC, int64_t ldc) override {
    throw std::runtime_error("MetalSolveCtx<std::vector<float*>>::solveL not yet implemented");
  }

  virtual void gemv(const std::vector<float*>* data, int64_t offset, int64_t nRows, int64_t nCols,
                    const std::vector<float*>* A, int64_t offA, int64_t lda, float alpha) override {
    throw std::runtime_error("MetalSolveCtx<std::vector<float*>>::gemv not yet implemented");
  }

  virtual void assembleVec(int64_t chainColPtr, int64_t numColItems, std::vector<float*>* C,
                           int64_t ldc) override {
    throw std::runtime_error("MetalSolveCtx<std::vector<float*>>::assembleVec not yet implemented");
  }

  virtual void solveLt(const std::vector<float*>* data, int64_t offset, int64_t n,
                       std::vector<float*>* C, int64_t offC, int64_t ldc) override {
    throw std::runtime_error("MetalSolveCtx<std::vector<float*>>::solveLt not yet implemented");
  }

  virtual void gemvT(const std::vector<float*>* data, int64_t offset, int64_t nRows, int64_t nCols,
                     std::vector<float*>* A, int64_t offA, int64_t lda, float alpha) override {
    throw std::runtime_error("MetalSolveCtx<std::vector<float*>>::gemvT not yet implemented");
  }

  virtual void assembleVecT(const std::vector<float*>* C, int64_t ldc, int64_t chainColPtr,
                            int64_t numColItems) override {
    throw std::runtime_error("MetalSolveCtx<std::vector<float*>>::assembleVecT not yet implemented");
  }

  MetalSymbolicCtx& sym;
  int nRHS;
  int batchSize;
};

// Factory method implementations
NumericCtxBase* MetalSymbolicCtx::createNumericCtxForType(type_index tIdx, int64_t tempBufSize,
                                                          int batchSize) {
  int64_t numSpans = skel.spanStart.size() - 1;

  // Check for double precision - not supported on Metal
  if (tIdx == type_index(typeid(double)) || tIdx == type_index(typeid(std::vector<double*>))) {
    throw std::runtime_error(
        "Metal backend does not support double precision. "
        "Apple Silicon GPUs lack double-precision floating point support. "
        "Please use float or select a different backend (BackendFast or BackendCuda).");
  }

  if (tIdx == type_index(typeid(float))) {
    return new MetalNumericCtx<float>(*this, tempBufSize, numSpans);
  }
  if (tIdx == type_index(typeid(std::vector<float*>))) {
    return new MetalNumericCtx<std::vector<float*>>(*this, tempBufSize, numSpans, batchSize);
  }

  throw std::runtime_error("MetalSymbolicCtx: unsupported numeric type");
}

SolveCtxBase* MetalSymbolicCtx::createSolveCtxForType(type_index tIdx, int nRHS, int batchSize) {
  // Check for double precision - not supported on Metal
  if (tIdx == type_index(typeid(double)) || tIdx == type_index(typeid(std::vector<double*>))) {
    throw std::runtime_error(
        "Metal backend does not support double precision. "
        "Apple Silicon GPUs lack double-precision floating point support. "
        "Please use float or select a different backend (BackendFast or BackendCuda).");
  }

  if (tIdx == type_index(typeid(float))) {
    return new MetalSolveCtx<float>(*this, nRHS);
  }
  if (tIdx == type_index(typeid(std::vector<float*>))) {
    return new MetalSolveCtx<std::vector<float*>>(*this, nRHS, batchSize);
  }

  throw std::runtime_error("MetalSymbolicCtx: unsupported solve type");
}

// Public factory function
OpsPtr metalOps() { return OpsPtr(new MetalOps); }

}  // end namespace BaSpaCho
