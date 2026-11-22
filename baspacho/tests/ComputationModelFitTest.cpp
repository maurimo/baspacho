/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#include <vector>
#include "baspacho/baspacho/ComputationModelFit.h"
#include "baspacho/baspacho/Solver.h"
#include "baspacho/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace ::BaSpaCho::testing_utils;
using namespace std;

namespace {

const ComputationModel kTrue(
    Eigen::Vector4d(2e-6, 1e-8, 3e-9, 1e-11),
    (Eigen::Matrix<double, 6, 1>() << 1e-6, 0, 2e-9, 1e-9, 1e-10, 2e-11).finished(),
    (Eigen::Matrix<double, 6, 1>() << 5e-7, 1e-9, 4e-10, 5e-9, 1e-9, 3e-11).finished(),
    Eigen::Vector4d(3e-7, 4e-8, 2e-7, 4e-9), 6e-10);

// samples of `kTrue` with multiplicative noise and a few outliers (descheduled process)
ComputationModelSamples syntheticSamples() {
  mt19937 gen(7);
  // sizes log-uniform in [1, 400] (as in factorizations: many small operations, a few large)
  uniform_real_distribution<double> logSz(0.0, log(400.0));
  auto sz = [&](mt19937& g) { return floor(exp(logSz(g))); };
  normal_distribution<double> noise(0.0, 0.05);
  auto t = [&](double est, int i) { return est * exp(noise(gen)) * (i % 97 == 0 ? 20.0 : 1.0); };
  ComputationModelSamples s;
  for (int i = 0; i < 3000; i++) {
    double m = sz(gen), n = sz(gen), k = sz(gen), br = floor(sz(gen) / 4) + 1;
    double bc = floor(sz(gen) / 8) + 1;
    s.potrf.push_back({n, t(kTrue.potrfEst(n), i)});
    s.trsm.push_back({n, k, t(kTrue.trsmEst(n, k), i)});
    s.syge.push_back({m, m + n, k, t(kTrue.sygeEst(m, m + n, k), i)});
    s.asmbl.push_back(
        {br + bc, bc, m, m + n, t(kTrue.asmblEst(br + bc, bc) + kTrue.asmblElemEst(m, m + n), i)});
  }
  return s;
}

}  // namespace

// the fit recovers the model (estimates within a few %), rejecting the outliers
TEST(ComputationModelFit, RecoversModel) {
  ComputationModelSamples s = syntheticSamples();
  for (bool timeWeighted : {false, true}) {
    ComputationModelFitStats stats[4];
    ComputationModel m = fitComputationModel(s, timeWeighted, 4.0, 4, stats);
    for (int i = 0; i < 4; i++) {
      EXPECT_LT(stats[i].medianRelErr, 0.06);
      EXPECT_GT(stats[i].kept, stats[i].samples * 95 / 100);
      EXPECT_LT(stats[i].kept, stats[i].samples);  // (the outliers)
      EXPECT_NEAR(stats[i].totalRatio, 1.0, 0.05);
    }
    for (double n : {1.0, 10.0, 100.0, 300.0}) {  // (within the range of the samples)
      EXPECT_NEAR(m.potrfEst(n) / kTrue.potrfEst(n), 1.0, 0.05);
      EXPECT_NEAR(m.trsmEst(n, n) / kTrue.trsmEst(n, n), 1.0, 0.05);
      EXPECT_NEAR(m.sygeEst(n, 2 * n, n) / kTrue.sygeEst(n, 2 * n, n), 1.0, 0.05);
    }
    for (double p : {m.potrfParams.minCoeff(), m.trsmParams.minCoeff(), m.sygeParams.minCoeff(),
                     m.asmblParams.minCoeff(), m.asmblElemParam}) {
      EXPECT_GE(p, 0.0);
    }
  }
}

TEST(ComputationModelFit, SaveLoadAndParams) {
  ComputationModelSamples s = syntheticSamples();
  string path = ::testing::TempDir() + "baspacho_samples.txt";
  s.save(path, /* append = */ false);
  ComputationModelSamples l;
  l.load(path);
  remove(path.c_str());
  ASSERT_EQ(l.potrf.size(), s.potrf.size());
  ASSERT_EQ(l.asmbl.size(), s.asmbl.size());
  EXPECT_NEAR(l.syge[5][3], s.syge[5][3], 1e-8 * s.syge[5][3]);

  ComputationModel m = parseComputationModelParams(computationModelParams(kTrue));
  EXPECT_DOUBLE_EQ(m.sygeEst(10, 20, 30), kTrue.sygeEst(10, 20, 30));
  EXPECT_DOUBLE_EQ(m.asmblElemEst(10, 20), kTrue.asmblElemEst(10, 20));
}

// samples collected from a CPU factorization: every assemble paired with its syrk/gemm
TEST(ComputationModelFit, CollectFromFactor) {
  auto cols = randomCols(150, 0.04, 9);
  SparseStructure ss = columnsToCscStruct(cols).transpose();
  vector<int64_t> paramSize = randomVec(150, 1, 4, 10);
  Settings settings;
  settings.backend = BackendFast;
  settings.numThreads = 1;
  auto solver = createSolver(settings, paramSize, ss);
  ComputationModelSamples s;
  s.collectFrom(*solver);
  solver->enableStats(true);
  vector<double> data = randomData<double>(solver->dataSize(), -1.0, 1.0, 11);
  solver->skel().damp(data, 0.0, (double)solver->order() * 1.2);
  solver->factor(data.data());
  solver->enableStats(false);
  int64_t denseLumps =
      solver->skel().numLumps() -
      (solver->sparseEliminationRanges().empty() ? 0 : solver->sparseEliminationRanges().back());
  EXPECT_EQ((int64_t)s.potrf.size(), denseLumps);
  EXPECT_EQ(s.asmbl.size(), s.syge.size());
  for (size_t i = 0; i < s.asmbl.size(); i++) {
    EXPECT_EQ(s.asmbl[i][2], s.syge[i][0]);  // m of the board
    EXPECT_EQ(s.asmbl[i][3], s.syge[i][1]);  // rows to the end
  }
}
