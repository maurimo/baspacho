/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// createSolver with Settings::numThreads = 1 must not start any thread (the parallel analysis
// must not touch dispenso's global thread pool), and the analysis must give the same result
// with any number of threads. Separate test binary: no other test may start threads before.
#include <gtest/gtest.h>
#include <algorithm>
#include <fstream>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/SparseStructure.h"

using namespace BaSpaCho;
using namespace std;

// threads of this process (Linux), -1 if unknown
static int processThreads() {
  ifstream status("/proc/self/status");
  string line;
  while (getline(status, line)) {
    if (line.rfind("Threads:", 0) == 0) {
      return stoi(line.substr(8));
    }
  }
  return -1;
}

// BA-like problem, large enough for the analysis loops to have many chunks
static pair<vector<int64_t>, SparseStructure> makeProblem() {
  const int64_t nPts = 30000, nCams = 600;
  mt19937 gen(3);
  uniform_int_distribution<int64_t> camDist(0, nCams - 1);
  vector<vector<int64_t>> rows(nPts + nCams);
  for (int64_t p = 0; p < nPts; p++) {
    for (int k = 0; k < 4; k++) {
      rows[nPts + camDist(gen)].push_back(p);
    }
  }
  for (int64_t c = 0; c + 1 < nCams; c++) {
    rows[nPts + c + 1].push_back(nPts + c);
  }
  SparseStructure ss;
  ss.ptrs.push_back(0);
  for (int64_t r = 0; r < (int64_t)rows.size(); r++) {
    rows[r].push_back(r);
    sort(rows[r].begin(), rows[r].end());
    rows[r].erase(unique(rows[r].begin(), rows[r].end()), rows[r].end());
    ss.inds.insert(ss.inds.end(), rows[r].begin(), rows[r].end());
    ss.ptrs.push_back(ss.inds.size());
  }
  vector<int64_t> paramSize(nPts, 3);
  paramSize.resize(nPts + nCams, 9);
  return {paramSize, ss};
}

TEST(AnalysisThreads, SingleThreadedAnalysisStartsNoThreads) {
  int before = processThreads();
  if (before < 0) {
    GTEST_SKIP() << "thread count not available on this platform";
  }
  auto [paramSize, ss] = makeProblem();
  Settings settings;
  settings.backend = BackendRef;
  settings.numThreads = 1;
  SolverPtr serial = createSolver(settings, paramSize, ss);
  EXPECT_EQ(processThreads(), before);

  // the parallel analysis gives the same structure
  settings.numThreads = 4;
  SolverPtr parallel = createSolver(settings, paramSize, ss);
  EXPECT_EQ(serial->paramToSpan(), parallel->paramToSpan());
  EXPECT_EQ(serial->sparseEliminationRanges(), parallel->sparseEliminationRanges());
  EXPECT_EQ(serial->skel().lumpStart, parallel->skel().lumpStart);
  EXPECT_EQ(serial->skel().chainColPtr, parallel->skel().chainColPtr);
  EXPECT_EQ(serial->skel().chainRowSpan, parallel->skel().chainRowSpan);
  EXPECT_EQ(serial->skel().chainData, parallel->skel().chainData);
}
