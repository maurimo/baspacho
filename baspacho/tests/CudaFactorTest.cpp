/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include "baspacho/baspacho/CoalescedBlockMatrix.h"
#include "baspacho/baspacho/CudaDefs.h"
#include "baspacho/baspacho/EliminationTree.h"
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/SparseStructure.h"
#include "baspacho/baspacho/Utils.h"
#include "baspacho/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace ::BaSpaCho::testing_utils;
using namespace std;
using namespace ::testing;

template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
struct Epsilon;
template <>
struct Epsilon<double> {
  static constexpr double value = 1e-10;
  static constexpr double value2 = 1e-8;
};
template <>
struct Epsilon<float> {
  static constexpr float value = 1e-5;
  static constexpr float value2 = 5e-5;
};

template <typename T>
void testCoalescedFactor(OpsPtr&& ops) {
  vector<set<int64_t>> colBlocks{{0, 3, 5}, {1}, {2, 4}, {3}, {4}, {5}};
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
  vector<int64_t> spanStart{0, 2, 5, 7, 10, 12, 15};
  vector<int64_t> lumpToSpan{0, 2, 4, 6};
  SparseStructure groupedSs = columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
  CoalescedBlockMatrixSkel factorSkel(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);

  vector<T> data(factorSkel.dataSize());
  iota(data.begin(), data.end(), 13);
  factorSkel.damp(data, T(5), T(50));

  Matrix<T> verifyMat = factorSkel.densify(data);
  Eigen::LLT<Eigen::Ref<Matrix<T>>> llt(verifyMat);

  Solver solver(std::move(factorSkel), {}, {}, std::move(ops));

  // call factor on gpu data
  {
    DevMirror<T> dataGpu(data);
    solver.factor(dataGpu.ptr);
    dataGpu.get(data);
  }

  Matrix<T> computedMat = solver.skel().densify(data);

  ASSERT_NEAR(Matrix<T>((verifyMat - computedMat).template triangularView<Eigen::Lower>()).norm(),
              0, Epsilon<T>::value);
}

TEST(CudaFactor, CoalescedFactor_double) { testCoalescedFactor<double>(cudaOps()); }

TEST(CudaFactor, CoalescedFactor_float) { testCoalescedFactor<float>(cudaOps()); }

template <typename T>
void testCoalescedFactor_Many(const std::function<OpsPtr()>& genOps) {
  for (int i = 0; i < 20; i++) {
    auto colBlocks = randomCols(115, 0.037, 57 + i);
    SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

    vector<int64_t> permutation = ss.fillReducingPermutation();
    vector<int64_t> invPerm = inversePermutation(permutation);
    SparseStructure sortedSs = ss.symmetricPermutation(invPerm, false);

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47 + i);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.processTree(/* compute sparse elim ranges = */ false);
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan, et.colStart,
                                        et.rowParam);

    vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
    factorSkel.damp(data, T(0), T(factorSkel.order() * 1.5));

    Matrix<T> verifyMat = factorSkel.densify(data);
    Eigen::LLT<Eigen::Ref<Matrix<T>>> llt(verifyMat);

    Solver solver(std::move(factorSkel), {}, {}, genOps());

    // call factor on gpu data
    {
      DevMirror<T> dataGpu(data);
      solver.factor(dataGpu.ptr);
      dataGpu.get(data);
    }

    Matrix<T> computedMat = solver.skel().densify(data);

    ASSERT_NEAR(Matrix<T>((verifyMat - computedMat).template triangularView<Eigen::Lower>()).norm(),
                0, Epsilon<T>::value2);
  }
}

TEST(CudaFactor, CoalescedFactor_Many_double) {
  testCoalescedFactor_Many<double>([] { return cudaOps(); });
}

TEST(CudaFactor, CoalescedFactor_Many_float) {
  testCoalescedFactor_Many<float>([] { return cudaOps(); });
}

template <typename T>
void testSparseElim_Many(const std::function<OpsPtr()>& genOps) {
  for (int i = 0; i < 20; i++) {
    auto colBlocks = randomCols(115, 0.03, 57 + i);
    colBlocks = makeIndependentElimSet(colBlocks, 0, 60);
    SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

    vector<int64_t> permutation = ss.fillReducingPermutation();
    vector<int64_t> invPerm = inversePermutation(permutation);
    SparseStructure sortedSs = ss;

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47 + i);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.processTree(/* compute sparse elim ranges = */ true);
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan, et.colStart,
                                        et.rowParam);

    vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
    factorSkel.damp(data, T(0), T(factorSkel.order() * 1.5));

    Matrix<T> verifyMat = factorSkel.densify(data);
    Eigen::LLT<Eigen::Ref<Matrix<T>>> llt(verifyMat);

    ASSERT_GE(et.sparseElimRanges.size(), 2);
    int64_t largestIndep = et.sparseElimRanges[1];
    Solver solver(move(factorSkel), move(et.sparseElimRanges), {}, genOps());

    NumericCtxPtr<T> numCtx = solver.internalSymbolicContext().createNumericCtx<T>(0, nullptr);

    // call doElimination with data on device
    {
      DevMirror<T> dataGpu(data);
      numCtx->doElimination(solver.internalGetElimCtx(0), dataGpu.ptr, 0, largestIndep);
      dataGpu.get(data);
    }

    Matrix<T> computedMat = solver.skel().densify(data);

    ASSERT_NEAR(Matrix<T>((verifyMat - computedMat).template triangularView<Eigen::Lower>())
                    .leftCols(largestIndep)
                    .norm(),
                0, Epsilon<T>::value);
  }
}

TEST(CudaFactor, SparseElim_Many_double) {
  testSparseElim_Many<double>([] { return cudaOps(); });
}

TEST(CudaFactor, SparseElim_Many_float) {
  testSparseElim_Many<float>([] { return cudaOps(); });
}

template <typename T>
void testSparseElimAndFactor_Many(const std::function<OpsPtr()>& genOps) {
  for (int i = 0; i < 20; i++) {
    auto colBlocks = randomCols(115, 0.03, 57 + i);
    colBlocks = makeIndependentElimSet(colBlocks, 0, 60);
    SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

    vector<int64_t> permutation = ss.fillReducingPermutation();
    vector<int64_t> invPerm = inversePermutation(permutation);
    SparseStructure sortedSs = ss;

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47 + i);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.processTree(/* compute sparse elim ranges = */ true);
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan, et.colStart,
                                        et.rowParam);

    vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
    factorSkel.damp(data, T(0), T(factorSkel.order() * 1.5));

    Matrix<T> verifyMat = factorSkel.densify(data);
    Eigen::LLT<Eigen::Ref<Matrix<T>>> llt(verifyMat);

    ASSERT_GE(et.sparseElimRanges.size(), 2);
    int64_t largestIndep = et.sparseElimRanges[1];
    Solver solver(move(factorSkel), move(et.sparseElimRanges), {}, genOps());

    // call factor on gpu data
    {
      DevMirror<T> dataGpu(data);
      solver.factor(dataGpu.ptr);
      dataGpu.get(data);
    }

    Matrix<T> computedMat = solver.skel().densify(data);
    ASSERT_NEAR(Matrix<T>((verifyMat - computedMat).template triangularView<Eigen::Lower>()).norm(),
                0, Epsilon<T>::value2);
  }
}

TEST(CudaFactor, SparseElimAndFactor_Many_double) {
  testSparseElimAndFactor_Many<double>([] { return cudaOps(); });
}

TEST(CudaFactor, SparseElimAndFactor_Many_float) {
  testSparseElimAndFactor_Many<float>([] { return cudaOps(); });
}

// Problem with (sparseElim) or without sparse elimination, and positive definite data. In
// `npdLump` (in the sparse elimination range or among the dense lumps) the diagonal block is
// then made indefinite by a large negative diagonal entry
template <typename T>
static unique_ptr<Solver> makeNotPosDefProblem(OpsPtr&& ops, bool sparseElim, vector<T>& pdData,
                                               vector<T>& npdData) {
  auto colBlocks = randomCols(115, 0.03, 57);
  colBlocks = makeIndependentElimSet(colBlocks, 0, 60);
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose();
  vector<int64_t> paramSize = randomVec(ss.ptrs.size() - 1, 2, 5, 47);
  EliminationTree et(paramSize, ss);
  et.buildTree();
  et.processTree(/* compute sparse elim ranges = */ sparseElim);
  et.computeAggregateStruct();
  CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan, et.colStart,
                                      et.rowParam);
  vector<int64_t> elimRanges = et.sparseElimRanges;
  EXPECT_EQ(elimRanges.size() >= 2, sparseElim);
  auto solver =
      make_unique<Solver>(std::move(factorSkel), std::move(elimRanges), vector<int64_t>{},
                          std::move(ops));
  const auto& skel = solver->skel();
  pdData = randomData<T>(skel.dataSize(), -1.0, 1.0, 9);
  skel.damp(pdData, T(0), T(skel.order() * 1.3));
  int64_t denseStart = sparseElim ? solver->sparseEliminationRanges().back() : 0;
  int64_t npdLump = sparseElim ? 3 : (denseStart + skel.numLumps()) / 2;
  npdData = pdData;
  npdData[skel.chainData[skel.chainColPtr[npdLump]]] = -T(skel.order() * 1000);
  return solver;
}

static bool allFinite(const vector<double>& v) {
  return std::all_of(v.begin(), v.end(), [](double x) { return std::isfinite(x); });
}
static bool allFinite(const vector<float>& v) {
  return std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
}

template <typename T>
void testNotPositiveDefinite(bool sparseElim) {
  vector<T> pdData, npdData;
  auto solver = makeNotPosDefProblem<T>(cudaOps(), sparseElim, pdData, npdData);
  for (vector<T>* d : {&pdData, &npdData}) {
    DevMirror<T> dataGpu(*d);
    solver->factor(dataGpu.ptr);
    dataGpu.get(*d);
  }
  EXPECT_TRUE(allFinite(pdData));
  EXPECT_FALSE(allFinite(npdData));
}

TEST(CudaFactor, NotPositiveDefinite_double) {
  testNotPositiveDefinite<double>(false);
  testNotPositiveDefinite<double>(true);
}
TEST(CudaFactor, NotPositiveDefinite_float) {
  testNotPositiveDefinite<float>(false);
  testNotPositiveDefinite<float>(true);
}
