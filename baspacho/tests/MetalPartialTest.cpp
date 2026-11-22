/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Tests of the partial factor/solve API on Metal (factorUpTo/From, solveL/Lt and their UpTo/From
// variants, addMvFrom, pseudoFactorFrom), mirroring PartialFactorSolveTest.
// Matrix and vectors are MetalMirror buffers that stay on the GPU between calls; vectors are
// stored with a stride larger than the order (padding filled with a sentinel that is checked
// afterwards), with 1 and more right-hand sides. Results are compared with a double precision
// Eigen reference. Problems are of three kinds:
//  * SparseElim: random structure with an independent set (sparse elimination ranges) and a
//    no-cross barrier placed after it,
//  * Fragmented: one span per lump (as the "fragmented" CPU tests; only for solve/addMv),
//  * LargeLumps: few large parameters with a denser structure, giving supernodes larger than the
//    64x64 tiles of the Metal dense kernels.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include "baspacho/baspacho/CoalescedBlockMatrix.h"
#include "baspacho/baspacho/EliminationTree.h"
#include "baspacho/baspacho/MetalDefs.h"
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/SparseStructure.h"
#include "baspacho/baspacho/Utils.h"
#include "baspacho/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace ::BaSpaCho::testing_utils;
using namespace std;
using namespace ::testing;

namespace {

using MatD = Eigen::MatrixXd;

// relative tolerance (float computation vs double reference)
constexpr double kTol = 4e-5;
constexpr float kSentinel = 1234.5f;

// this value must match the value in EliminationTree.cpp - so that the no-cross
// barriers are placed without preventing the sparse elimination from happening
constexpr int64_t minNumSparseElimNodes = 50;

constexpr int kNumProblems = 10;

enum class Kind { SparseElim, Fragmented, LargeLumps };

const char* kindName(Kind k) {
  switch (k) {
    case Kind::SparseElim:
      return "SparseElim";
    case Kind::Fragmented:
      return "Fragmented";
    case Kind::LargeLumps:
      return "LargeLumps";
  }
  return "?";
}

struct Problem {
  unique_ptr<Solver> solver;
  vector<float> data;
  int64_t nocross;    // span index of the barrier
  int64_t order;      // matrix order
  int64_t barrierAt;  // row/col index of the barrier
  int64_t afterBar;   // order - barrierAt
  MatD mat;           // densified data, lower half significant
};

// diagDamp < 0 means "order * 2" (well conditioned matrix, for factor tests)
Problem makeProblem(Kind kind, int i, double diagDamp) {
  CoalescedBlockMatrixSkel* skelPtr = nullptr;
  unique_ptr<CoalescedBlockMatrixSkel> skel;
  vector<int64_t> elimRanges;
  int64_t nocross;

  if (kind == Kind::SparseElim) {
    int numParams = 215;
    auto colBlocks = randomCols(numParams, 0.03, 57 + i);
    colBlocks = makeIndependentElimSet(colBlocks, 0, 150);
    SparseStructure sortedSs = columnsToCscStruct(colBlocks).transpose();
    nocross = (7 * i) % (210 - minNumSparseElimNodes) + minNumSparseElimNodes + 1;

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 3, 47 + i);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.processTree(/* compute sparse elim ranges = */ true, {nocross});
    et.computeAggregateStruct();
    skel = make_unique<CoalescedBlockMatrixSkel>(et.computeSpanStart(), et.lumpToSpan,
                                                 et.colStart, et.rowParam);
    elimRanges = std::move(et.sparseElimRanges);
  } else if (kind == Kind::Fragmented) {
    int numParams = 215;
    auto colBlocks = randomCols(numParams, 0.03, 57 + i);
    colBlocks = makeIndependentElimSet(colBlocks, 0, 150);
    SparseStructure sortedSs = columnsToCscStruct(colBlocks);
    nocross = (7 * i) % (210 - minNumSparseElimNodes) + minNumSparseElimNodes + 1;

    vector<int64_t> spanStart = randomVec(sortedSs.ptrs.size() - 1, 2, 3, 47 + i);
    spanStart.push_back(0);
    cumSumVec(spanStart);
    vector<int64_t> lumpToSpan(sortedSs.ptrs.size());
    iota(lumpToSpan.begin(), lumpToSpan.end(), 0);
    skel = make_unique<CoalescedBlockMatrixSkel>(spanStart, lumpToSpan, sortedSs.ptrs,
                                                 sortedSs.inds);
  } else {
    int numParams = 60;
    auto colBlocks = randomCols(numParams, 0.15, 91 + i);
    SparseStructure sortedSs = columnsToCscStruct(colBlocks).transpose();
    nocross = 20 + (7 * i) % 30;

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 3, 12, 31 + i);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.processTree(/* compute sparse elim ranges = */ false, {nocross});
    et.computeAggregateStruct();
    skel = make_unique<CoalescedBlockMatrixSkel>(et.computeSpanStart(), et.lumpToSpan,
                                                 et.colStart, et.rowParam);
  }
  skelPtr = skel.get();
  if (kind == Kind::LargeLumps) {
    // make sure supernodes are larger than the 64x64 tiles of the Metal dense kernels
    int64_t maxLumpSize = 0;
    for (int64_t l = 0; l < skelPtr->numLumps(); l++) {
      maxLumpSize = std::max(maxLumpSize, skelPtr->lumpStart[l + 1] - skelPtr->lumpStart[l]);
    }
    EXPECT_GT(maxLumpSize, 64);
  }

  Problem p;
  p.nocross = nocross;
  p.order = skelPtr->order();
  p.barrierAt = skelPtr->spanStart[nocross];
  p.afterBar = p.order - p.barrierAt;
  p.data = randomData<float>(skelPtr->dataSize(), -1.0, 1.0, 9 + i);
  skelPtr->damp(p.data, 0.0f, float(diagDamp < 0 ? p.order * 2.0 : diagDamp));
  p.mat = skelPtr->densify(p.data).cast<double>();
  EXPECT_EQ(skelPtr->spanOffsetInLump[nocross], 0);
  p.solver = make_unique<Solver>(std::move(*skel), std::move(elimRanges), vector<int64_t>{},
                                 metalOps());
  return p;
}

string traceMsg(Kind kind, int i, int nRHS = 0, int64_t pad = 0) {
  stringstream ss;
  ss << kindName(kind) << " problem " << i << ", nRHS " << nRHS << ", stride padding " << pad;
  return ss.str();
}

// (order x nRHS) vectors stored column-major with stride order + pad, padding = sentinel
struct Vecs {
  int64_t order, stride, nRHS;
  vector<float> host;

  Vecs(int64_t order_, int64_t pad, int64_t nRHS_, int seed)
      : order(order_), stride(order_ + pad), nRHS(nRHS_) {
    host = randomData<float>(stride * nRHS, -1.0, 1.0, seed);
    setPaddingToSentinel();
  }

  void setPaddingToSentinel() {
    for (int64_t c = 0; c < nRHS; c++) {
      for (int64_t r = order; r < stride; r++) {
        host[c * stride + r] = kSentinel;
      }
    }
  }

  void setZero() {
    for (int64_t c = 0; c < nRHS; c++) {
      for (int64_t r = 0; r < order; r++) {
        host[c * stride + r] = 0.0f;
      }
    }
  }

  MatD mat() const {
    MatD m(order, nRHS);
    for (int64_t c = 0; c < nRHS; c++) {
      for (int64_t r = 0; r < order; r++) {
        m(r, c) = host[c * stride + r];
      }
    }
    return m;
  }

  int64_t paddingViolations() const {
    int64_t n = 0;
    for (int64_t c = 0; c < nRHS; c++) {
      for (int64_t r = order; r < stride; r++) {
        n += host[c * stride + r] != kSentinel;
      }
    }
    return n;
  }
};

double relErr(const MatD& computed, const MatD& ref) {
  return (computed - ref).norm() / std::max(ref.norm(), 1e-30);
}

// (nRHS, stride padding) combinations used in the solve and addMv tests
const vector<pair<int, int64_t>> kRhsConfigs{{1, 0}, {3, 0}, {1, 5}, {4, 3}};

}  // namespace

// ---------------------------------------------------------------------------------------------
// Factor
// ---------------------------------------------------------------------------------------------

void testPartialFactor(Kind kind) {
  for (int i = 0; i < kNumProblems; i++) {
    SCOPED_TRACE(traceMsg(kind, i));
    Problem p = makeProblem(kind, i, -1.0);
    if (kind == Kind::SparseElim) {
      ASSERT_GE(p.solver->sparseEliminationRanges().size(), 2);
    }

    MatD L = p.mat;
    Eigen::LLT<Eigen::Ref<MatD>> llt(L);
    MatD decBl = L.bottomLeftCorner(p.afterBar, p.barrierAt);
    MatD expected = L;
    expected.bottomRightCorner(p.afterBar, p.afterBar) =
        p.mat.bottomRightCorner(p.afterBar, p.afterBar) - decBl * decBl.transpose();

    {
      MetalMirror<float> dataGpu(p.data);
      p.solver->factorUpTo(dataGpu.ptr(), p.nocross);
      dataGpu.get(p.data);
    }
    MatD computed = p.solver->skel().densify(p.data).cast<double>();

    MatD expL = expected.triangularView<Eigen::Lower>();
    MatD compL = computed.triangularView<Eigen::Lower>();
    ASSERT_NEAR(relErr(compL, expL), 0, kTol);
  }
}

TEST(MetalPartial, PartialFactor_SparseElim) { testPartialFactor(Kind::SparseElim); }
TEST(MetalPartial, PartialFactor_LargeLumps) { testPartialFactor(Kind::LargeLumps); }

// factorUpTo + factorFrom on the same GPU buffer, without coming back to the CPU in between
void testSplitFactor(Kind kind) {
  for (int i = 0; i < kNumProblems; i++) {
    SCOPED_TRACE(traceMsg(kind, i));
    Problem p = makeProblem(kind, i, -1.0);

    MatD L = p.mat;
    Eigen::LLT<Eigen::Ref<MatD>> llt(L);

    {
      MetalMirror<float> dataGpu(p.data);
      p.solver->factorUpTo(dataGpu.ptr(), p.nocross);
      p.solver->factorFrom(dataGpu.ptr(), p.nocross);
      dataGpu.get(p.data);
    }
    MatD computed = p.solver->skel().densify(p.data).cast<double>();

    MatD expL = L.triangularView<Eigen::Lower>();
    MatD compL = computed.triangularView<Eigen::Lower>();
    ASSERT_NEAR(relErr(compL, expL), 0, kTol);
  }
}

TEST(MetalPartial, SplitFactor_SparseElim) { testSplitFactor(Kind::SparseElim); }
TEST(MetalPartial, SplitFactor_LargeLumps) { testSplitFactor(Kind::LargeLumps); }

// ---------------------------------------------------------------------------------------------
// Solve (the matrix data is used as the L factor directly)
// ---------------------------------------------------------------------------------------------

enum class SolveOp { L, Lt, LUpTo, LtUpTo, LFrom, LtFrom };

MatD solveRef(SolveOp op, const MatD& mat, const MatD& b, int64_t barrierAt) {
  int64_t order = mat.rows();
  int64_t afterBar = order - barrierAt;
  MatD L = mat.triangularView<Eigen::Lower>();
  MatD x = b;
  switch (op) {
    case SolveOp::L:
      x = L.triangularView<Eigen::Lower>().solve(b);
      break;
    case SolveOp::Lt:
      x = L.triangularView<Eigen::Lower>().transpose().solve(b);
      break;
    case SolveOp::LUpTo:
      x.topRows(barrierAt) = L.topLeftCorner(barrierAt, barrierAt)
                                 .triangularView<Eigen::Lower>()
                                 .solve(b.topRows(barrierAt));
      x.bottomRows(afterBar) -= L.bottomLeftCorner(afterBar, barrierAt) * x.topRows(barrierAt);
      break;
    case SolveOp::LtUpTo:
      x.topRows(barrierAt) -=
          L.bottomLeftCorner(afterBar, barrierAt).transpose() * b.bottomRows(afterBar);
      x.topRows(barrierAt) = L.topLeftCorner(barrierAt, barrierAt)
                                 .triangularView<Eigen::Lower>()
                                 .transpose()
                                 .solve(x.topRows(barrierAt));
      break;
    case SolveOp::LFrom:
      x.bottomRows(afterBar) = L.bottomRightCorner(afterBar, afterBar)
                                   .triangularView<Eigen::Lower>()
                                   .solve(b.bottomRows(afterBar));
      break;
    case SolveOp::LtFrom:
      x.bottomRows(afterBar) = L.bottomRightCorner(afterBar, afterBar)
                                   .triangularView<Eigen::Lower>()
                                   .transpose()
                                   .solve(b.bottomRows(afterBar));
      break;
  }
  return x;
}

void runSolveOp(SolveOp op, const Solver& solver, const float* data, int64_t nocross, float* vec,
                int64_t stride, int nRHS) {
  switch (op) {
    case SolveOp::L:
      solver.solveL(data, vec, stride, nRHS);
      break;
    case SolveOp::Lt:
      solver.solveLt(data, vec, stride, nRHS);
      break;
    case SolveOp::LUpTo:
      solver.solveLUpTo(data, nocross, vec, stride, nRHS);
      break;
    case SolveOp::LtUpTo:
      solver.solveLtUpTo(data, nocross, vec, stride, nRHS);
      break;
    case SolveOp::LFrom:
      solver.solveLFrom(data, nocross, vec, stride, nRHS);
      break;
    case SolveOp::LtFrom:
      solver.solveLtFrom(data, nocross, vec, stride, nRHS);
      break;
  }
}

void testSolve(SolveOp op, Kind kind) {
  for (int i = 0; i < kNumProblems; i++) {
    Problem p = makeProblem(kind, i, 3.0);
    if (kind == Kind::SparseElim) {
      ASSERT_GE(p.solver->sparseEliminationRanges().size(), 2);
    }
    MetalMirror<float> dataGpu(p.data);

    for (auto [nRHS, pad] : kRhsConfigs) {
      SCOPED_TRACE(traceMsg(kind, i, nRHS, pad));
      Vecs v(p.order, pad, nRHS, 49 + i + 7 * nRHS);
      MatD ref = solveRef(op, p.mat, v.mat(), p.barrierAt);

      {
        MetalMirror<float> vecGpu(v.host);
        runSolveOp(op, *p.solver, dataGpu.ptr(), p.nocross, vecGpu.ptr(), v.stride, nRHS);
        vecGpu.get(v.host);
      }

      ASSERT_EQ(v.paddingViolations(), 0);
      ASSERT_NEAR(relErr(v.mat(), ref), 0, kTol);
    }
  }
}

#define SOLVE_TESTS(Op)                                                              \
  TEST(MetalPartial, Solve##Op##_SparseElim) { testSolve(SolveOp::Op, Kind::SparseElim); } \
  TEST(MetalPartial, Solve##Op##_Fragmented) { testSolve(SolveOp::Op, Kind::Fragmented); } \
  TEST(MetalPartial, Solve##Op##_LargeLumps) { testSolve(SolveOp::Op, Kind::LargeLumps); }

SOLVE_TESTS(L)
SOLVE_TESTS(Lt)
SOLVE_TESTS(LUpTo)
SOLVE_TESTS(LtUpTo)
SOLVE_TESTS(LFrom)
SOLVE_TESTS(LtFrom)

// ---------------------------------------------------------------------------------------------
// addMvFrom: out += alpha * Sym(A[after barrier]) * in, different strides for in/out
// ---------------------------------------------------------------------------------------------

void testAddMv(Kind kind, bool fromBarrier) {
  for (int i = 0; i < kNumProblems; i++) {
    Problem p = makeProblem(kind, i, -1.0);
    int64_t from = fromBarrier ? p.nocross : 0;
    int64_t barrierAt = fromBarrier ? p.barrierAt : 0;
    int64_t afterBar = p.order - barrierAt;
    MetalMirror<float> dataGpu(p.data);

    for (auto [nRHS, pad] : kRhsConfigs) {
      SCOPED_TRACE(traceMsg(kind, i, nRHS, pad));
      float alpha = (nRHS % 2) ? 3.5f : -1.0f;
      Vecs in(p.order, pad, nRHS, 49 + i + 7 * nRHS);
      Vecs out(p.order, pad + 2, nRHS, 77 + i + 7 * nRHS);
      MatD inM = in.mat();
      MatD ref = out.mat();
      MatD sym = p.mat.bottomRightCorner(afterBar, afterBar).selfadjointView<Eigen::Lower>();
      ref.bottomRows(afterBar) += alpha * sym * inM.bottomRows(afterBar);

      {
        MetalMirror<float> inGpu(in.host), outGpu(out.host);
        p.solver->addMvFrom(dataGpu.ptr(), from, inGpu.ptr(), in.stride, outGpu.ptr(),
                            out.stride, nRHS, alpha);
        outGpu.get(out.host);
        vector<float> inAfter(in.host.size());
        inGpu.get(inAfter);
        ASSERT_EQ(inAfter, in.host);  // input untouched
      }

      ASSERT_EQ(out.paddingViolations(), 0);
      ASSERT_NEAR(relErr(out.mat(), ref), 0, kTol);
    }
  }
}

TEST(MetalPartial, AddMv_SparseElim) { testAddMv(Kind::SparseElim, false); }
TEST(MetalPartial, AddMvFrom_SparseElim) { testAddMv(Kind::SparseElim, true); }
TEST(MetalPartial, AddMv_Fragmented) { testAddMv(Kind::Fragmented, false); }
TEST(MetalPartial, AddMvFrom_Fragmented) { testAddMv(Kind::Fragmented, true); }
TEST(MetalPartial, AddMv_LargeLumps) { testAddMv(Kind::LargeLumps, false); }
TEST(MetalPartial, AddMvFrom_LargeLumps) { testAddMv(Kind::LargeLumps, true); }

// ---------------------------------------------------------------------------------------------
// pseudoFactorFrom: diagonal blocks factored, blocks below solved, no Schur complement update
// ---------------------------------------------------------------------------------------------

void testPseudoFactor(Kind kind) {
  for (int i = 0; i < kNumProblems; i++) {
    SCOPED_TRACE(traceMsg(kind, i));
    Problem p = makeProblem(kind, i, -1.0);
    int64_t from = (i % 2) ? p.nocross : 0;
    const CoalescedBlockMatrixSkel& skel = p.solver->skel();

    MatD expected = p.mat;
    for (int64_t s = from; s < skel.numSpans(); s++) {
      int64_t start = skel.spanStart[s];
      int64_t end = skel.spanStart[s + 1];
      int64_t size = end - start;
      auto diagBlock = expected.block(start, start, size, size);
      auto belowDiagBlock = expected.block(end, start, p.order - end, size);
      { Eigen::LLT<Eigen::Ref<MatD>> llt(diagBlock); }
      diagBlock.triangularView<Eigen::Lower>().transpose().solveInPlace<Eigen::OnTheRight>(
          belowDiagBlock);
    }

    {
      MetalMirror<float> dataGpu(p.data);
      p.solver->pseudoFactorFrom(dataGpu.ptr(), from);
      dataGpu.get(p.data);
    }
    MatD computed = skel.densify(p.data).cast<double>();

    MatD expL = expected.triangularView<Eigen::Lower>();
    MatD compL = computed.triangularView<Eigen::Lower>();
    ASSERT_NEAR(relErr(compL, expL), 0, kTol);
  }
}

TEST(MetalPartial, PseudoFactor_SparseElim) { testPseudoFactor(Kind::SparseElim); }
TEST(MetalPartial, PseudoFactor_LargeLumps) { testPseudoFactor(Kind::LargeLumps); }

// ---------------------------------------------------------------------------------------------
// A whole computation kept on the GPU: b = A * x (addMvFrom on a copy of A), factor A in two
// steps, solve with the four partial solves, then read back only the result and compare with x.
// Also the plain factor + solve path, alternating over the problems.
// ---------------------------------------------------------------------------------------------

void testGpuResidentPipeline(Kind kind) {
  for (int i = 0; i < kNumProblems; i++) {
    Problem p = makeProblem(kind, i, -1.0);
    auto [nRHS, pad] = kRhsConfigs[i % kRhsConfigs.size()];
    bool split = (i % 2) == 0;
    SCOPED_TRACE(traceMsg(kind, i, nRHS, pad) + (split ? ", split ops" : ", whole ops"));

    Vecs x(p.order, pad, nRHS, 13 + i);
    Vecs b = x;
    b.setZero();

    MetalMirror<float> matGpu(p.data), dataGpu(p.data), xGpu(x.host), bGpu(b.host);
    const Solver& s = *p.solver;
    s.addMvFrom(matGpu.ptr(), 0, xGpu.ptr(), x.stride, bGpu.ptr(), b.stride, nRHS, 1.0f);
    if (split) {
      s.factorUpTo(dataGpu.ptr(), p.nocross);
      s.factorFrom(dataGpu.ptr(), p.nocross);
      s.solveLUpTo(dataGpu.ptr(), p.nocross, bGpu.ptr(), b.stride, nRHS);
      s.solveLFrom(dataGpu.ptr(), p.nocross, bGpu.ptr(), b.stride, nRHS);
      s.solveLtFrom(dataGpu.ptr(), p.nocross, bGpu.ptr(), b.stride, nRHS);
      s.solveLtUpTo(dataGpu.ptr(), p.nocross, bGpu.ptr(), b.stride, nRHS);
    } else {
      s.factor(dataGpu.ptr());
      s.solve(dataGpu.ptr(), bGpu.ptr(), b.stride, nRHS);
    }
    bGpu.get(b.host);

    ASSERT_EQ(b.paddingViolations(), 0);
    ASSERT_NEAR(relErr(b.mat(), x.mat()), 0, 1e-4);
  }
}

TEST(MetalPartial, GpuResidentPipeline_SparseElim) { testGpuResidentPipeline(Kind::SparseElim); }
TEST(MetalPartial, GpuResidentPipeline_LargeLumps) { testGpuResidentPipeline(Kind::LargeLumps); }

// Partial solves at every cut (lump boundary) of the dense part: the level solve (one schedule
// for all the ranges) and the lump-by-lump solve give the same results
TEST(MetalPartial, LevelVsLumpwise_AllCuts) {
  for (Kind kind : {Kind::SparseElim, Kind::LargeLumps}) {
    Problem p = makeProblem(kind, 0, 3.0);
    MetalMirror<float> dataGpu(p.data);
    const auto& skel = p.solver->skel();
    const auto& ranges = p.solver->sparseEliminationRanges();
    int64_t denseStart = ranges.empty() ? 0 : ranges.back();
    int64_t numCuts = 0;
    for (int64_t l = denseStart; l <= skel.numLumps(); l++) {
      int64_t span = skel.lumpToSpan[l];
      for (SolveOp op : {SolveOp::LUpTo, SolveOp::LtUpTo, SolveOp::LFrom, SolveOp::LtFrom}) {
        SCOPED_TRACE(string(kindName(kind)) + " cut at span " + to_string(span) + " op " +
                     to_string((int)op));
        Vecs v(p.order, 1, 2, 7 + (int)l);
        vector<float> results[2];
        for (int levelSolve = 0; levelSolve < 2; levelSolve++) {
          MetalSolveOptions options;
          options.levelSolve = levelSolve;
          MetalSolveOptionsScope scope(options);
          results[levelSolve] = v.host;
          MetalMirror<float> vecGpu(results[levelSolve]);
          runSolveOp(op, *p.solver, dataGpu.ptr(), span, vecGpu.ptr(), v.stride, 2);
          vecGpu.get(results[levelSolve]);
        }
        Eigen::Map<Eigen::VectorXf> r0(results[0].data(), results[0].size()),
            r1(results[1].data(), results[1].size());
        ASSERT_LT((r1 - r0).norm(), 1e-5 * std::max(1.0f, r0.norm()));
      }
      numCuts++;
    }
    EXPECT_GT(numCuts, 2);
  }
}
