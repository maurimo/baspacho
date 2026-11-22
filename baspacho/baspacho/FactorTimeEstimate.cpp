/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "baspacho/baspacho/FactorTimeEstimate.h"
#include <stdexcept>
#include <typeindex>
#include "baspacho/baspacho/MatOps.h"
#include "baspacho/baspacho/Solver.h"

namespace BaSpaCho {

using namespace std;

namespace {

struct ModelSymElimCtx : SymElimCtx {};

struct ModelNumericCtx : NumericCtx<double> {
  ModelNumericCtx(const CoalescedBlockMatrixSkel& skel, const ComputationModel& model,
                  FactorTimeEstimate& est)
      : skel(skel), model(model), est(est) {}
  void pseudoFactorSpans(double*, int64_t, int64_t) override {
    throw runtime_error("model backend: pseudoFactorSpans not supported");
  }
  void doElimination(const SymElimCtx&, double*, int64_t lumpsBegin, int64_t lumpsEnd) override {
    for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
      double n = skel.lumpStart[l + 1] - skel.lumpStart[l];
      int64_t c0 = skel.chainColPtr[l], c1 = skel.chainColPtr[l + 1];
      int64_t nDiag = skel.lumpToSpan[l + 1] - skel.lumpToSpan[l];
      double rows = skel.chainRowsTillEnd[c1 - 1] - skel.chainRowsTillEnd[c0 + nDiag - 1];
      double blocks = c1 - c0 - nDiag;
      est.sparse += model.potrfEst(n);
      if (blocks > 0) {
        est.sparse += model.trsmEst(n, rows) + model.sygeEst(rows, rows, n) +
                      model.asmblEst(blocks, blocks);
      }
    }
  }
  void add(int op, double t) {
    est.supernodal += t;
    est.opTime[op] += t;
    est.opCount[op]++;
  }
  void potrf(int64_t n, double*, int64_t) override { add(0, model.potrfEst(n)); }
  void trsm(int64_t n, int64_t k, double*, int64_t, int64_t) override {
    add(1, model.trsmEst(n, k));
  }
  void saveSyrkGemm(int64_t m, int64_t n, int64_t k, const double*, int64_t) override {
    add(2, model.sygeEst(m, n, k));
  }
  void prepareAssemble(int64_t) override {}
  void assemble(double*, int64_t rectRowBegin, int64_t, int64_t srcColDataOffset,
                int64_t srcRectWidth, int64_t numBlockRows, int64_t numBlockCols) override {
    // rows from the target lump to the end of the column
    double rowsFull =
        skel.chainRowsTillEnd[srcColDataOffset + numBlockRows - 1] - rectRowBegin;
    add(3, model.asmblEst(numBlockRows, numBlockCols) +
               model.asmblElemEst(srcRectWidth, rowsFull));
  }
  const CoalescedBlockMatrixSkel& skel;
  const ComputationModel& model;
  FactorTimeEstimate& est;
};

struct ModelSymbolicCtx : SymbolicCtx {
  ModelSymbolicCtx(const CoalescedBlockMatrixSkel& skel, const ComputationModel& model,
                   FactorTimeEstimate& est)
      : skel(skel), model(model), est(est) {}
  SymElimCtxPtr prepareElimination(int64_t, int64_t) override {
    return SymElimCtxPtr(new ModelSymElimCtx);
  }
  NumericCtxBase* createNumericCtxForType(type_index tIdx, int64_t, int) override {
    if (tIdx != type_index(typeid(double))) {
      throw runtime_error("model backend: double only");
    }
    return new ModelNumericCtx(skel, model, est);
  }
  SolveCtxBase* createSolveCtxForType(type_index, int, int) override {
    throw runtime_error("model backend: no solve");
  }
  PermutedCoalescedAccessor deviceAccessor() override { return PermutedCoalescedAccessor(); }
  const CoalescedBlockMatrixSkel& skel;
  const ComputationModel& model;
  FactorTimeEstimate& est;
};

struct ModelOps : Ops {
  ModelOps(const ComputationModel& model, FactorTimeEstimate& est) : model(model), est(est) {}
  SymbolicCtxPtr createSymbolicCtx(const CoalescedBlockMatrixSkel& skel,
                                   const vector<int64_t>&) override {
    return SymbolicCtxPtr(new ModelSymbolicCtx(skel, model, est));
  }
  const ComputationModel& model;
  FactorTimeEstimate& est;
};

}  // namespace

FactorTimeEstimate estimateFactorTime(const CoalescedBlockMatrixSkel& skel,
                                      const vector<int64_t>& sparseElimRanges,
                                      const ComputationModel& model) {
  FactorTimeEstimate est;
  CoalescedBlockMatrixSkel skelCopy = skel;
  vector<int64_t> ranges = sparseElimRanges;
  vector<int64_t> perm(skel.numSpans());
  for (int64_t i = 0; i < (int64_t)perm.size(); i++) {
    perm[i] = i;
  }
  Solver modelSolver(std::move(skelCopy), std::move(ranges), std::move(perm),
                     OpsPtr(new ModelOps(model, est)));
  modelSolver.factor<double>(nullptr);
  return est;
}

FactorTimeEstimate Solver::estimatedFactorTime(const ComputationModel& model) const {
  return estimateFactorTime(factorSkel, sparseElimRanges, model);
}

}  // end namespace BaSpaCho
