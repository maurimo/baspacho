/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Tests of the dense operations of the Metal backend (potrf/trsm of the factorization, and
// solveL/solveLt/gemv/gemvT/symm of the solve), called directly on the numeric/solve contexts
// and compared with a double precision Eigen reference. Sizes are chosen around the tile size
// of the Metal kernels (64), and offsets/strides are not aligned. Data outside of the block
// written by an operation must be left untouched (eg. the upper half of the diagonal blocks).
// The same checks are run with the reference CPU backend as baseline.

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <functional>
#include <random>
#include <tuple>
#include <vector>
#include "baspacho/baspacho/CoalescedBlockMatrix.h"
#include "baspacho/baspacho/MatOps.h"
#ifdef BASPACHO_USE_METAL
#include "baspacho/baspacho/MetalDefs.h"
#endif

using namespace BaSpaCho;
using namespace std;

namespace {

using MatD = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatCD = Eigen::MatrixXd;  // column-major

constexpr double kTol = 2e-5;     // relative error (Frobenius norm) vs double reference
constexpr float kSentinel = 1234.5f;  // value of the data that must not be modified
constexpr int64_t kOrder = 1100;  // order of the skeleton (bounds gemv rows)

// host "mirror" for the CPU baselines, same interface as MetalMirror
template <typename T>
struct HostMirror {
  explicit HostMirror(const vector<T>& v) : data(v) {}
  T* ptr() { return data.data(); }
  void get(vector<T>& v) const { v = data; }
  vector<T> data;
};

// skeleton with a single lump of size kOrder (the dense ops do not use the skeleton)
struct Env {
  explicit Env(const function<OpsPtr()>& genOps)
      : skel({0, kOrder}, {0, 1}, {0, 1}, {0}), ops(genOps()) {
    vector<int64_t> perm(kOrder);
    for (int64_t i = 0; i < kOrder; i++) {
      perm[i] = i;
    }
    sym = ops->createSymbolicCtx(skel, perm);
  }
  CoalescedBlockMatrixSkel skel;
  OpsPtr ops;
  SymbolicCtxPtr sym;
};

vector<float> randomVec(int64_t size, mt19937& gen) {
  uniform_real_distribution<float> dist(-1.0f, 1.0f);
  vector<float> v(size);
  for (auto& x : v) {
    x = dist(gen);
  }
  return v;
}

// random SPD matrix (well conditioned), rounded to float values
MatD randomSpd(int64_t n, mt19937& gen) {
  vector<float> r = randomVec(n * n, gen);
  MatD R = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
               r.data(), n, n)
               .cast<double>();
  MatD A = R * R.transpose() / double(n) + MatD::Identity(n, n);
  return A.cast<float>().cast<double>();
}

// lower Cholesky factor of a random SPD matrix, rounded to float values
MatD randomLower(int64_t n, mt19937& gen) {
  MatD L = randomSpd(n, gen).llt().matrixL();
  return L.cast<float>().cast<double>();
}

// writes the lower half of M (row-major n x n) at data[off], upper half set to sentinel
void putLower(vector<float>& data, int64_t off, const MatD& M) {
  int64_t n = M.rows();
  for (int64_t r = 0; r < n; r++) {
    for (int64_t c = 0; c < n; c++) {
      data[off + r * n + c] = c <= r ? float(M(r, c)) : kSentinel;
    }
  }
}

MatD getRowMajor(const vector<float>& data, int64_t off, int64_t rows, int64_t cols) {
  MatD M(rows, cols);
  for (int64_t r = 0; r < rows; r++) {
    for (int64_t c = 0; c < cols; c++) {
      M(r, c) = data[off + r * cols + c];
    }
  }
  return M;
}

MatCD getColMajor(const vector<float>& data, int64_t off, int64_t rows, int64_t cols,
                  int64_t ld) {
  MatCD M(rows, cols);
  for (int64_t c = 0; c < cols; c++) {
    for (int64_t r = 0; r < rows; r++) {
      M(r, c) = data[off + c * ld + r];
    }
  }
  return M;
}

double relErr(const Eigen::MatrixXd& computed, const Eigen::MatrixXd& expected) {
  return (computed - expected).norm() / max(expected.norm(), 1e-30);
}

// checks that data and orig coincide outside of the "written" index set
void expectUntouched(const vector<float>& data, const vector<float>& orig,
                     const function<bool(int64_t)>& written, const string& what) {
  int64_t numModified = 0;
  for (int64_t i = 0; i < (int64_t)data.size(); i++) {
    if (!written(i) && data[i] != orig[i]) {
      numModified++;
    }
  }
  EXPECT_EQ(numModified, 0) << what << ": data outside of the output was modified";
}

template <template <typename> class Mirror>
void testPotrf(const function<OpsPtr()>& genOps) {
  Env env(genOps);
  mt19937 gen(37);
  for (int64_t n : {1, 2, 3, 5, 16, 63, 64, 65, 100, 128, 129, 200, 300}) {
    const int64_t off = 13;
    MatD A = randomSpd(n, gen);
    vector<float> data = randomVec(off + n * n + 7, gen);
    putLower(data, off, A);
    vector<float> orig = data;

    {
      NumericCtxPtr<float> numCtx = env.sym->createNumericCtx<float>(0, (float*)nullptr);
      Mirror<float> dev(data);
      numCtx->potrf(n, dev.ptr(), off);
      dev.get(data);
    }

    MatD L = getRowMajor(data, off, n, n).triangularView<Eigen::Lower>();
    MatD Lref = A.llt().matrixL();
    EXPECT_LT(relErr(L, Lref), kTol) << "potrf n=" << n;
    expectUntouched(
        data, orig,
        [&](int64_t i) {
          int64_t j = i - off;
          return j >= 0 && j < n * n && (j % n) <= (j / n);
        },
        "potrf n=" + to_string(n));
  }
}

template <template <typename> class Mirror>
void testTrsm(const function<OpsPtr()>& genOps) {
  Env env(genOps);
  mt19937 gen(39);
  for (auto [n, k] : vector<pair<int64_t, int64_t>>{
           {1, 1}, {3, 7}, {6, 1}, {64, 100}, {65, 3}, {130, 513}, {200, 64}, {300, 30}}) {
    const int64_t offA = 5, offB = offA + n * n + 3;
    MatD L = randomLower(n, gen);
    vector<float> data = randomVec(offB + k * n + 9, gen);
    putLower(data, offA, L);
    MatD B = getRowMajor(data, offB, k, n);
    vector<float> orig = data;

    {
      NumericCtxPtr<float> numCtx = env.sym->createNumericCtx<float>(0, (float*)nullptr);
      Mirror<float> dev(data);
      numCtx->trsm(n, k, dev.ptr(), offA, offB);
      dev.get(data);
    }

    // X * L^T = B
    MatD Xref = L.triangularView<Eigen::Lower>().solve(B.transpose()).transpose();
    EXPECT_LT(relErr(getRowMajor(data, offB, k, n), Xref), kTol) << "trsm n=" << n << " k=" << k;
    expectUntouched(
        data, orig, [&](int64_t i) { return i >= offB && i < offB + k * n; },
        "trsm n=" + to_string(n) + " k=" + to_string(k));
  }
}

template <template <typename> class Mirror>
void testSolveLLt(const function<OpsPtr()>& genOps) {
  Env env(genOps);
  mt19937 gen(41);
  for (int nRHS : {1, 3}) {
    for (int64_t n : {1, 2, 7, 63, 64, 65, 129, 200, 300}) {
      for (bool transpose : {false, true}) {
        const int64_t offM = 9, offC = 4, ldc = n + 5;
        MatD L = randomLower(n, gen);
        vector<float> mat = randomVec(offM + n * n + 3, gen);
        putLower(mat, offM, L);
        vector<float> vec = randomVec(offC + ldc * nRHS + 2, gen);
        MatCD C = getColMajor(vec, offC, n, nRHS, ldc);
        vector<float> origMat = mat, origVec = vec;

        {
          SolveCtxPtr<float> slvCtx = env.sym->createSolveCtx<float>(nRHS, (float*)nullptr);
          Mirror<float> devMat(mat), devVec(vec);
          if (transpose) {
            slvCtx->solveLt(devMat.ptr(), offM, n, devVec.ptr(), offC, ldc);
          } else {
            slvCtx->solveL(devMat.ptr(), offM, n, devVec.ptr(), offC, ldc);
          }
          devVec.get(vec);
          devMat.get(mat);
        }

        MatCD Xref = transpose ? MatCD(L.transpose().triangularView<Eigen::Upper>().solve(C))
                               : MatCD(L.triangularView<Eigen::Lower>().solve(C));
        string what = string(transpose ? "solveLt" : "solveL") + " n=" + to_string(n) +
                      " nRHS=" + to_string(nRHS);
        EXPECT_LT(relErr(getColMajor(vec, offC, n, nRHS, ldc), Xref), kTol) << what;
        expectUntouched(
            vec, origVec,
            [&](int64_t i) {
              int64_t j = i - offC;
              return j >= 0 && (j % ldc) < n && (j / ldc) < nRHS;
            },
            what);
        expectUntouched(mat, origMat, [](int64_t) { return false; }, what + " (matrix)");
      }
    }
  }
}

// gemv (to the internal temporary) followed by gemvT (from the temporary), as in the solve:
// y += alpha2 * M^T * (alpha1 * M * x)
template <template <typename> class Mirror>
void testGemvGemvT(const function<OpsPtr()>& genOps) {
  Env env(genOps);
  mt19937 gen(43);
  for (int nRHS : {1, 3}) {
    for (auto [nRows, nCols] : vector<pair<int64_t, int64_t>>{
             {1, 1}, {5, 3}, {100, 64}, {513, 65}, {1000, 130}, {3, 200}}) {
      const int64_t offM = 7, offA = 3, lda = nCols + 4;
      const float alpha1 = -1.0f, alpha2 = 0.5f;
      vector<float> mat = randomVec(offM + nRows * nCols + 5, gen);
      vector<float> x = randomVec(offA + lda * nRHS + 1, gen);
      vector<float> y = randomVec(offA + lda * nRHS + 1, gen);
      MatD M = getRowMajor(mat, offM, nRows, nCols);
      MatCD X = getColMajor(x, offA, nCols, nRHS, lda);
      MatCD Y = getColMajor(y, offA, nCols, nRHS, lda);
      vector<float> origX = x, origY = y;

      {
        SolveCtxPtr<float> slvCtx = env.sym->createSolveCtx<float>(nRHS, (float*)nullptr);
        Mirror<float> devMat(mat), devX(x), devY(y);
        slvCtx->gemv(devMat.ptr(), offM, nRows, nCols, devX.ptr(), offA, lda, alpha1);
        slvCtx->gemvT(devMat.ptr(), offM, nRows, nCols, devY.ptr(), offA, lda, alpha2);
        devX.get(x);
        devY.get(y);
      }

      MatCD Yref = Y + double(alpha2) * M.transpose() * (double(alpha1) * M * X);
      string what = "gemv/gemvT nRows=" + to_string(nRows) + " nCols=" + to_string(nCols) +
                    " nRHS=" + to_string(nRHS);
      EXPECT_LT(relErr(getColMajor(y, offA, nCols, nRHS, lda) - Y, Yref - Y), kTol) << what;
      expectUntouched(x, origX, [](int64_t) { return false; }, what + " (x)");
      expectUntouched(
          y, origY,
          [&](int64_t i) {
            int64_t j = i - offA;
            return j >= 0 && (j % lda) < nCols && (j / lda) < nRHS;
          },
          what + " (y)");
    }
  }
}

// D += alpha * A * C, A symmetric with only the lower half stored
template <template <typename> class Mirror>
void testSymm(const function<OpsPtr()>& genOps) {
  Env env(genOps);
  mt19937 gen(47);
  for (int nRHS : {1, 3}) {
    for (int64_t n : {1, 3, 16, 64, 65, 130, 250}) {
      const int64_t offM = 2, offC = 6, ldc = n + 3, ldd = n + 8;
      const float alpha = -0.75f;
      MatD A = randomSpd(n, gen);
      vector<float> mat = randomVec(offM + n * n + 4, gen);
      putLower(mat, offM, A);
      vector<float> c = randomVec(offC + ldc * nRHS + 1, gen);
      vector<float> d = randomVec(offC + ldd * nRHS + 1, gen);
      MatCD C = getColMajor(c, offC, n, nRHS, ldc);
      MatCD D = getColMajor(d, offC, n, nRHS, ldd);
      vector<float> origD = d;

      {
        SolveCtxPtr<float> slvCtx = env.sym->createSolveCtx<float>(nRHS, (float*)nullptr);
        Mirror<float> devMat(mat), devC(c), devD(d);
        slvCtx->symm(devMat.ptr(), offM, n, devC.ptr(), offC, ldc, devD.ptr(), ldd, alpha);
        devD.get(d);
      }

      MatCD Dref = D + double(alpha) * MatCD(A) * C;
      string what = "symm n=" + to_string(n) + " nRHS=" + to_string(nRHS);
      EXPECT_LT(relErr(getColMajor(d, offC, n, nRHS, ldd) - D, Dref - D), kTol) << what;
      expectUntouched(
          d, origD,
          [&](int64_t i) {
            int64_t j = i - offC;
            return j >= 0 && (j % ldd) < n && (j / ldd) < nRHS;
          },
          what);
    }
  }
}

}  // namespace

#define DENSE_TESTS(Suite, Mirror, genOps)                                            \
  TEST(Suite, Potrf) { testPotrf<Mirror>(genOps); }                                   \
  TEST(Suite, Trsm) { testTrsm<Mirror>(genOps); }                                     \
  TEST(Suite, SolveL_SolveLt) { testSolveLLt<Mirror>(genOps); }                       \
  TEST(Suite, Gemv_GemvT) { testGemvGemvT<Mirror>(genOps); }                          \
  TEST(Suite, Symm) { testSymm<Mirror>(genOps); }

DENSE_TESTS(DenseOpsRefBaseline, HostMirror, [] { return simpleOps(); })

#ifdef BASPACHO_USE_METAL
DENSE_TESTS(MetalDense, MetalMirror, [] { return metalOps(); })
#endif
