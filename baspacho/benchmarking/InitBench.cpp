/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Benchmark of solver creation (symbolic analysis) vs numeric factor/solve on
// block-sparse problems loaded from files:
//  * BAL bundle adjustment problems (-bal file.txt), only the block structure is used
//  * g2o pose graphs (-g2o file.g2o), SE2/SE3 vertices/edges
//  * Matrix Market symmetric matrices (-mtx file.mtx), block structure detected
//    by grouping consecutive rows with identical sparsity pattern (supervariables)
// Numeric values are random (diagonally dominant), and a solve with residual
// check is run to verify correctness.

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <memory>
#include <typeindex>
#include <unordered_map>
#include "baspacho/baspacho/ComputationModel.h"
#include "baspacho/baspacho/MatOps.h"
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/SparseStructure.h"
#include "baspacho/baspacho/Utils.h"
#ifdef BASPACHO_USE_CUBLAS
#include "baspacho/baspacho/CudaDefs.h"
#endif
#ifdef BASPACHO_HAVE_CHOLMOD
#include "BenchCholmod.h"
#endif

using namespace BaSpaCho;
using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

struct Problem {
  string name;
  vector<int64_t> paramSize;
  SparseStructure ss;  // lower-half CSR of blocks, diagonal included
  int64_t numElim = 0; // if >0, first numElim params form an independent set (BA points)
};

// builds lower csr from (unordered, possibly duplicated, any-triangle) pairs
static SparseStructure lowerCsrFromPairs(int64_t n, vector<pair<int64_t, int64_t>>& pairs) {
  for (auto& p : pairs) {
    if (p.first < p.second) swap(p.first, p.second);
  }
  for (int64_t i = 0; i < n; i++) pairs.emplace_back(i, i);
  sort(pairs.begin(), pairs.end());
  pairs.erase(unique(pairs.begin(), pairs.end()), pairs.end());
  SparseStructure ss;
  ss.ptrs.assign(n + 1, 0);
  ss.inds.reserve(pairs.size());
  for (auto& p : pairs) {
    ss.ptrs[p.first + 1]++;
    ss.inds.push_back(p.second);
  }
  for (int64_t i = 0; i < n; i++) ss.ptrs[i + 1] += ss.ptrs[i];
  return ss;
}

static Problem loadBal(const string& path, int camSize, int ptSize) {
  FILE* f = fopen(path.c_str(), "r");
  if (!f) throw runtime_error("cannot open " + path);
  long nCams, nPts, nObs;
  if (fscanf(f, "%ld %ld %ld", &nCams, &nPts, &nObs) != 3 || nCams < 0 || nPts < 0 || nObs < 0) {
    throw runtime_error("bad bal header");
  }
  vector<pair<int64_t, int64_t>> pairs;
  pairs.reserve(nObs);
  for (long i = 0; i < nObs; i++) {
    long c, p;
    double x, y;
    if (fscanf(f, "%ld %ld %lf %lf", &c, &p, &x, &y) != 4) throw runtime_error("bad bal obs");
    if (c < 0 || c >= nCams || p < 0 || p >= nPts) {
      throw runtime_error("bal observation " + to_string(i) + " out of range");
    }
    pairs.emplace_back(nPts + c, p);  // points first, then cameras
  }
  fclose(f);
  Problem prob;
  prob.paramSize.assign(nPts, ptSize);
  prob.paramSize.resize(nPts + nCams, camSize);
  prob.ss = lowerCsrFromPairs(nPts + nCams, pairs);
  prob.numElim = nPts;
  return prob;
}

static Problem loadG2o(const string& path) {
  ifstream in(path);
  if (!in) throw runtime_error("cannot open " + path);
  unordered_map<int64_t, int64_t> idToIdx;
  vector<int64_t> sizes;
  vector<pair<int64_t, int64_t>> edges;
  string line, tag;
  while (getline(in, line)) {
    istringstream ls(line);
    if (!(ls >> tag)) continue;  // blank line
    if (tag.rfind("VERTEX", 0) == 0) {
      int64_t id;
      if (!(ls >> id)) throw runtime_error("bad g2o vertex: " + line);
      int sz = tag.find("SE2") != string::npos || tag.find("XY") != string::npos ? 3 : 6;
      if (tag == "VERTEX_XY") sz = 2;
      if (tag == "VERTEX_TRACKXYZ" || tag == "VERTEX_POINT_XYZ") sz = 3;
      if (idToIdx.emplace(id, sizes.size()).second) sizes.push_back(sz);
    } else if (tag.rfind("EDGE", 0) == 0) {
      int64_t a, b;
      if (!(ls >> a >> b)) throw runtime_error("bad g2o edge: " + line);
      edges.emplace_back(a, b);
      // some datasets have no VERTEX lines: infer vertices from edges
      int sz = tag.find("SE2") != string::npos ? 3 : 6;
      for (int64_t id : {a, b}) {
        if (idToIdx.emplace(id, sizes.size()).second) sizes.push_back(sz);
      }
    }
  }
  Problem prob;
  prob.paramSize = sizes;
  vector<pair<int64_t, int64_t>> pairs;
  for (auto [a, b] : edges) {
    auto ia = idToIdx.find(a), ib = idToIdx.find(b);
    if (ia != idToIdx.end() && ib != idToIdx.end()) pairs.emplace_back(ia->second, ib->second);
  }
  prob.ss = lowerCsrFromPairs(sizes.size(), pairs);
  return prob;
}

// Matrix Market, symmetric (or general, symmetrized) pattern; detects blocks
static Problem loadMtx(const string& path, int maxBlock) {
  ifstream in(path);
  if (!in) throw runtime_error("cannot open " + path);
  string line;
  getline(in, line);
  while (in.peek() == '%') getline(in, line);
  int64_t nr, nc, nnz;
  in >> nr >> nc >> nnz;
  if (!in || nr != nc || nr <= 0 || nnz < 0) throw runtime_error("bad mtx header");
  getline(in, line);
  vector<vector<int32_t>> adj(nr);  // full symmetric adjacency incl. diagonal
  for (int64_t k = 0; k < nnz; k++) {
    if (!getline(in, line)) break;
    const char* s = line.c_str();
    char* e;
    int64_t i = strtoll(s, &e, 10) - 1;
    int64_t j = strtoll(e, &e, 10) - 1;
    if (i < 0 || i >= nr || j < 0 || j >= nr) {
      throw runtime_error("mtx entry " + to_string(k) + " out of range: " + line);
    }
    adj[i].push_back(j);
    if (i != j) adj[j].push_back(i);
  }
  for (int64_t i = 0; i < nr; i++) {
    adj[i].push_back(i);
    sort(adj[i].begin(), adj[i].end());
    adj[i].erase(unique(adj[i].begin(), adj[i].end()), adj[i].end());
  }
  // group consecutive rows with identical pattern (supervariables)
  vector<int64_t> blockOf(nr), sizes;
  for (int64_t i = 0; i < nr;) {
    int64_t j = i + 1;
    while (j < nr && j - i < maxBlock && adj[j] == adj[i]) j++;
    for (int64_t k = i; k < j; k++) blockOf[k] = sizes.size();
    sizes.push_back(j - i);
    i = j;
  }
  vector<pair<int64_t, int64_t>> pairs;
  for (int64_t i = 0; i < nr; i++) {
    if (i > 0 && blockOf[i] == blockOf[i - 1]) continue;  // same pattern as previous row
    for (int32_t j : adj[i]) {
      if (blockOf[j] < blockOf[i]) pairs.emplace_back(blockOf[i], blockOf[j]);
    }
  }
  adj.clear();
  Problem prob;
  prob.paramSize = sizes;
  prob.ss = lowerCsrFromPairs(sizes.size(), pairs);
  return prob;
}

// synthetic 3D FEM-like problem: n^3 nodes with bs dofs, 27-point connectivity
static Problem makeGrid3d(int64_t n, int64_t bs) {
  auto id = [n](int64_t x, int64_t y, int64_t z) { return (x * n + y) * n + z; };
  vector<pair<int64_t, int64_t>> pairs;
  for (int64_t x = 0; x < n; x++)
    for (int64_t y = 0; y < n; y++)
      for (int64_t z = 0; z < n; z++)
        for (int64_t dx = 0; dx <= 1; dx++)
          for (int64_t dy = -1; dy <= 1; dy++)
            for (int64_t dz = -1; dz <= 1; dz++) {
              int64_t X = x + dx, Y = y + dy, Z = z + dz;
              if ((dx == 0 && (dy < 0 || (dy == 0 && dz <= 0))) || X >= n || Y < 0 || Y >= n ||
                  Z < 0 || Z >= n)
                continue;
              pairs.emplace_back(id(X, Y, Z), id(x, y, z));
            }
  Problem prob;
  prob.paramSize.assign(n * n * n, bs);
  prob.ss = lowerCsrFromPairs(n * n * n, pairs);
  prob.name = "grid3d_" + to_string(n) + "_bs" + to_string(bs);
  return prob;
}

// Synthetic problems, spec "type:arg:arg...":
//  grid2d:n:bs                 n x n grid, 9-point stencil
//  grid3d:n:bs                 n^3 grid, 27-point stencil
//  knn2d:N:k:bs, knn3d:N:k:bs  N random points in the unit square/cube, each connected to its
//                              k nearest neighbors (mesh-like)
//  ba:cams:pts:obs:win:cs      bundle adjustment with locality: cameras on a trajectory (odometry
//                              between consecutive ones), each point seen by `obs` cameras within
//                              `win` of a random camera; points (size 3) first, cameras (size cs)
//  slam:poses:loops:bs         pose graph: 2D random walk on a lattice, odometry chain, up to
//                              `loops` loop closures per pose with earlier poses in the same cell
//  powerlaw:N:m:bs             Barabasi-Albert preferential attachment, m edges per node
//  banded:N:bw:extra:bs        band of half-width bw plus `extra` random long-range edges
//  er:N:deg:bs                 Erdos-Renyi, average degree deg
static Problem makeGenerated(const string& spec) {
  vector<string> a;
  {
    stringstream ss(spec);
    string t;
    while (getline(ss, t, ':')) a.push_back(t);
  }
  auto arg = [&](size_t i) -> int64_t {
    if (i >= a.size()) throw runtime_error("missing argument " + to_string(i) + " in " + spec);
    int64_t v = stoll(a[i]);
    if (v <= 0) throw runtime_error("arguments must be positive in " + spec);
    return v;
  };
  const string& type = a.at(0);
  mt19937_64 gen(12345);
  vector<pair<int64_t, int64_t>> pairs;
  Problem prob;
  prob.name = spec;
  auto finish = [&](int64_t n, int64_t bs) {
    prob.paramSize.assign(n, bs);
    prob.ss = lowerCsrFromPairs(n, pairs);
    return prob;
  };
  if (type == "grid2d") {
    int64_t n = arg(1), bs = arg(2);
    for (int64_t x = 0; x < n; x++)
      for (int64_t y = 0; y < n; y++)
        for (int64_t dx = 0; dx <= 1; dx++)
          for (int64_t dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy <= 0) continue;
            int64_t X = x + dx, Y = y + dy;
            if (X < n && Y >= 0 && Y < n) pairs.emplace_back(X * n + Y, x * n + y);
          }
    return finish(n * n, bs);
  }
  if (type == "grid3d") {
    Problem p = makeGrid3d(arg(1), arg(2));
    p.name = spec;
    return p;
  }
  if (type == "knn2d" || type == "knn3d") {
    int dim = type == "knn2d" ? 2 : 3;
    int64_t N = arg(1), k = arg(2), bs = arg(3);
    uniform_real_distribution<double> u(0, 1);
    vector<array<double, 3>> pts(N);
    for (auto& p : pts) p = {u(gen), u(gen), dim == 3 ? u(gen) : 0.0};
    // bucket grid with ~2 points per cell
    int64_t cells = max<int64_t>(1, (int64_t)pow(N / 2.0, 1.0 / dim));
    auto cellOf = [&](double v) { return min<int64_t>(cells - 1, (int64_t)(v * cells)); };
    auto key = [&](int64_t cx, int64_t cy, int64_t cz) { return (cz * cells + cy) * cells + cx; };
    int64_t numCells = dim == 3 ? cells * cells * cells : cells * cells;
    vector<vector<int64_t>> bucket(numCells);
    for (int64_t i = 0; i < N; i++)
      bucket[key(cellOf(pts[i][0]), cellOf(pts[i][1]), dim == 3 ? cellOf(pts[i][2]) : 0)]
          .push_back(i);
    for (int64_t i = 0; i < N; i++) {
      int64_t cx = cellOf(pts[i][0]), cy = cellOf(pts[i][1]), cz = dim == 3 ? cellOf(pts[i][2]) : 0;
      vector<pair<double, int64_t>> cand;
      for (int64_t r = 1; (int64_t)cand.size() < k + 1 && r <= cells; r++) {
        cand.clear();
        for (int64_t z = (dim == 3 ? cz - r : 0); z <= (dim == 3 ? cz + r : 0); z++)
          for (int64_t y = cy - r; y <= cy + r; y++)
            for (int64_t x = cx - r; x <= cx + r; x++) {
              if (x < 0 || y < 0 || z < 0 || x >= cells || y >= cells || z >= cells) continue;
              for (int64_t j : bucket[key(x, y, z)]) {
                double d = 0;
                for (int c = 0; c < 3; c++) d += (pts[i][c] - pts[j][c]) * (pts[i][c] - pts[j][c]);
                cand.emplace_back(d, j);
              }
            }
      }
      int64_t m = min<int64_t>(k + 1, cand.size());
      partial_sort(cand.begin(), cand.begin() + m, cand.end());
      for (int64_t q = 0; q < m; q++)
        if (cand[q].second != i) pairs.emplace_back(i, cand[q].second);
    }
    return finish(N, bs);
  }
  if (type == "ba") {
    int64_t cams = arg(1), npts = arg(2), obs = arg(3), win = arg(4), cs = arg(5);
    uniform_int_distribution<int64_t> camDist(0, cams - 1), offDist(-win, win);
    for (int64_t p = 0; p < npts; p++) {
      int64_t c0 = camDist(gen);
      for (int64_t o = 0; o < obs; o++) {
        int64_t c = min(cams - 1, max<int64_t>(0, c0 + offDist(gen)));
        pairs.emplace_back(npts + c, p);
      }
    }
    for (int64_t c = 0; c + 1 < cams; c++) pairs.emplace_back(npts + c + 1, npts + c);
    prob.paramSize.assign(npts, 3);
    prob.paramSize.resize(npts + cams, cs);
    prob.ss = lowerCsrFromPairs(npts + cams, pairs);
    prob.numElim = npts;
    return prob;
  }
  if (type == "slam") {
    int64_t poses = arg(1), loops = arg(2), bs = arg(3);
    uniform_int_distribution<int> dir(0, 3);
    map<pair<int64_t, int64_t>, vector<int64_t>> visits;
    int64_t x = 0, y = 0;
    for (int64_t i = 0; i < poses; i++) {
      if (i > 0) pairs.emplace_back(i, i - 1);
      auto& v = visits[{x, y}];
      int64_t added = 0;
      for (auto it = v.rbegin(); it != v.rend() && added < loops; ++it)
        if (*it < i - 2) pairs.emplace_back(i, *it), added++;
      v.push_back(i);
      int d = dir(gen);
      x += d == 0 ? 1 : d == 1 ? -1 : 0;
      y += d == 2 ? 1 : d == 3 ? -1 : 0;
    }
    return finish(poses, bs);
  }
  if (type == "powerlaw") {
    int64_t N = arg(1), m = arg(2), bs = arg(3);
    vector<int64_t> targets;  // nodes repeated by degree
    for (int64_t i = 0; i < N; i++) {
      for (int64_t e = 0; e < m && i > 0; e++) {
        int64_t j = targets.empty() ? 0 : targets[uniform_int_distribution<int64_t>(
                                              0, targets.size() - 1)(gen)];
        if (j == i) continue;
        pairs.emplace_back(i, j);
        targets.push_back(j);
        targets.push_back(i);
      }
      if (i == 0) targets.push_back(0);
    }
    return finish(N, bs);
  }
  if (type == "banded") {
    int64_t N = arg(1), bw = arg(2), extra = arg(3), bs = arg(4);
    for (int64_t i = 0; i < N; i++)
      for (int64_t j = max<int64_t>(0, i - bw); j < i; j++) pairs.emplace_back(i, j);
    uniform_int_distribution<int64_t> nd(0, N - 1);
    for (int64_t e = 0; e < extra; e++) {
      int64_t i = nd(gen), j = nd(gen);
      if (i != j) pairs.emplace_back(i, j);
    }
    return finish(N, bs);
  }
  if (type == "er") {
    int64_t N = arg(1), deg = arg(2), bs = arg(3);
    uniform_int_distribution<int64_t> nd(0, N - 1);
    for (int64_t e = 0; e < N * deg / 2; e++) {
      int64_t i = nd(gen), j = nd(gen);
      if (i != j) pairs.emplace_back(i, j);
    }
    return finish(N, bs);
  }
  throw runtime_error("unknown generator " + type);
}

// fill input blocks of A with random values making it diagonally dominant
template <typename Acc>
static void fillRandom(const Problem& p, const Acc& acc, vector<double>& data) {
  mt19937 gen(37);
  uniform_real_distribution<double> dist(-1.0, 1.0);
  int64_t n = p.paramSize.size();
  vector<double> rowAbsSum(n, 0.0);
  for (int64_t i = 0; i < n; i++) {
    for (int64_t k = p.ss.ptrs[i]; k < p.ss.ptrs[i + 1]; k++) {
      int64_t j = p.ss.inds[k];
      if (j == i) continue;
      auto b = acc.block(data.data(), i, j);
      for (int r = 0; r < b.rows(); r++)
        for (int c = 0; c < b.cols(); c++) b(r, c) = dist(gen);
      double s = b.cwiseAbs().sum();
      rowAbsSum[i] += s;
      rowAbsSum[j] += s;
    }
  }
  for (int64_t i = 0; i < n; i++) {
    auto d = acc.diagBlock(data.data(), i);
    for (int r = 0; r < d.rows(); r++) {
      for (int c = 0; c < r; c++) d(r, c) = d(c, r) = 0.1 * dist(gen);
      d(r, r) = rowAbsSum[i] + d.rows() + 1.0;
    }
  }
}

// y = A x, A given by its input blocks in factor data (before factorization)
template <typename Acc>
static void spmv(const Problem& p, const Acc& acc, const vector<double>& data,
                 const vector<double>& x, vector<double>& y) {
  using namespace Eigen;
  int64_t n = p.paramSize.size();
  y.assign(x.size(), 0.0);
  auto seg = [&](vector<double>& v, int64_t i) {
    return Map<VectorXd>(v.data() + acc.paramStart(i), acc.paramSize(i));
  };
  auto cseg = [&](const vector<double>& v, int64_t i) {
    return Map<const VectorXd>(v.data() + acc.paramStart(i), acc.paramSize(i));
  };
  for (int64_t i = 0; i < n; i++) {
    for (int64_t k = p.ss.ptrs[i]; k < p.ss.ptrs[i + 1]; k++) {
      int64_t j = p.ss.inds[k];
      if (j == i) {
        MatrixXd d = acc.diagBlock((double*)data.data(), i);
        d.triangularView<StrictlyUpper>() = d.triangularView<StrictlyLower>().transpose();
        seg(y, i) += d * cseg(x, i);
      } else {
        auto b = acc.block((double*)data.data(), i, j);
        seg(y, i) += b * cseg(x, j);
        seg(y, j) += b.transpose() * cseg(x, i);
      }
    }
  }
}

struct Opts {
  int reps = 3;
  int numThreads = 16;
  bool fast = true, cuda = false, cholmod = false, givenElim = true, autoElim = true;
  bool verbose = false;
  bool createOnly = false;
  vector<string> models;  // -model: estimate the factor time with computation models
  bool fastOnly = false;  // -fastOnly: with -model, skip the exact analysis
  string cm;              // -cm: computation model of the analysis in factor runs
  string analysis;        // -analysis exact|fast (default: auto)
  string collect;         // -collect file: append the timings of the factor operations
};

static uint64_t hashVec(uint64_t h, const vector<int64_t>& v) {
  for (int64_t x : v) {
    h ^= (uint64_t)x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  }
  return h ^ v.size();
}

// fingerprint of the factor structure, to verify that analysis changes are neutral
static uint64_t skelHash(const Solver& solver) {
  const auto& sk = solver.skel();
  uint64_t h = 1469598103934665603ULL;
  for (auto* v : {&sk.spanStart, &sk.lumpToSpan, &sk.chainColPtr, &sk.chainRowSpan, &sk.chainData,
                  &sk.chainRowsTillEnd, &sk.boardColPtr, &sk.boardRowLump, &sk.boardChainColOrd,
                  &sk.boardRowPtr, &sk.boardColLump, &sk.boardColOrd}) {
    h = hashVec(h, *v);
  }
  h = hashVec(h, solver.paramToSpan());
  h = hashVec(h, solver.sparseEliminationRanges());
  return h;
}

// "Model" backend: runs the factorization loops of the Solver on the skeleton only (no numeric
// data is allocated, data pointers are null), adding the time estimated by a computation model
// for each operation of the supernodal (dense) part. The sparse elimination (not covered by the
// model, and batched on the GPU backends) is estimated separately, per lump, as its potrf, the
// trsm and syrk of all the rows below, and the assembly of its blocks.
struct ModelSymElimCtx : SymElimCtx {};

struct ModelNumericCtx : NumericCtx<double> {
  ModelNumericCtx(const CoalescedBlockMatrixSkel& skel, const ComputationModel& model, double& time,
                  double& sparseTime)
      : skel(skel), model(model), time(time), sparseTime(sparseTime) {}
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
      sparseTime += model.potrfEst(n);
      if (blocks > 0) {
        sparseTime += model.trsmEst(n, rows) + model.sygeEst(rows, rows, n) +
                      model.asmblEst(blocks, blocks);
      }
    }
  }
  void add(int op, double t) {
    time += t;
    opTime[op] += t;
    opCount[op]++;
  }
  void potrf(int64_t n, double*, int64_t) override { add(0, model.potrfEst(n)); }
  void trsm(int64_t n, int64_t k, double*, int64_t, int64_t) override {
    add(1, model.trsmEst(n, k));
  }
  void saveSyrkGemm(int64_t m, int64_t n, int64_t k, const double*, int64_t) override {
    add(2, model.sygeEst(m, n, k));
  }
  void prepareAssemble(int64_t) override {}
  void assemble(double*, int64_t, int64_t, int64_t, int64_t, int64_t numBlockRows,
                int64_t numBlockCols) override {
    add(3, model.asmblEst(numBlockRows, numBlockCols));
  }
  ~ModelNumericCtx() override {
    if (getenv("BASPACHO_MERGE_DEBUG")) {
      fprintf(stderr, "MOCK potrf %.6g/%lld trsm %.6g/%lld syge %.6g/%lld asmbl %.6g/%lld\n",
              opTime[0], (long long)opCount[0], opTime[1], (long long)opCount[1], opTime[2],
              (long long)opCount[2], opTime[3], (long long)opCount[3]);
    }
  }
  double opTime[4] = {0, 0, 0, 0};
  int64_t opCount[4] = {0, 0, 0, 0};
  const CoalescedBlockMatrixSkel& skel;
  const ComputationModel& model;
  double& time;
  double& sparseTime;
};

struct ModelSymbolicCtx : SymbolicCtx {
  ModelSymbolicCtx(const CoalescedBlockMatrixSkel& skel, const ComputationModel& model,
                   double& time, double& sparseTime)
      : skel(skel), model(model), time(time), sparseTime(sparseTime) {}
  SymElimCtxPtr prepareElimination(int64_t, int64_t) override {
    return SymElimCtxPtr(new ModelSymElimCtx);
  }
  NumericCtxBase* createNumericCtxForType(type_index tIdx, int64_t, int) override {
    if (tIdx != type_index(typeid(double))) {
      throw runtime_error("model backend: double only");
    }
    return new ModelNumericCtx(skel, model, time, sparseTime);
  }
  SolveCtxBase* createSolveCtxForType(type_index, int, int) override {
    throw runtime_error("model backend: no solve");
  }
  PermutedCoalescedAccessor deviceAccessor() override { return PermutedCoalescedAccessor(); }
  const CoalescedBlockMatrixSkel& skel;
  const ComputationModel& model;
  double& time;
  double& sparseTime;
};

struct ModelOps : Ops {
  ModelOps(const ComputationModel& model, double& time, double& sparseTime)
      : model(model), time(time), sparseTime(sparseTime) {}
  SymbolicCtxPtr createSymbolicCtx(const CoalescedBlockMatrixSkel& skel,
                                   const vector<int64_t>&) override {
    return SymbolicCtxPtr(new ModelSymbolicCtx(skel, model, time, sparseTime));
  }
  const ComputationModel& model;
  double& time;
  double& sparseTime;
};

// time of the factorization of `solver`'s factor estimated by `model` (no data allocated):
// supernodal part, and estimate of the sparse elimination part
static pair<double, double> modelledFactorTime(const Solver& solver,
                                               const ComputationModel& model) {
  double time = 0, sparseTime = 0;
  CoalescedBlockMatrixSkel skel = solver.skel();
  vector<int64_t> ranges = solver.sparseEliminationRanges();
  vector<int64_t> perm = solver.paramToSpan();
  Solver modelSolver(std::move(skel), std::move(ranges), std::move(perm),
                     OpsPtr(new ModelOps(model, time, sparseTime)));
  modelSolver.factor<double>(nullptr);
  return {time, sparseTime};
}

// a model by name, or "file:path" with the 20 parameters (potrf 4, trsm 6, syge 6, asmbl 4)
static const ComputationModel* namedModel(const string& name) {
  static map<string, unique_ptr<ComputationModel>> fromFiles;
  if (name.rfind("file:", 0) == 0) {
    auto& m = fromFiles[name];
    if (!m) {
      ifstream in(name.substr(5));
      vector<double> v;
      double x;
      while (in >> x) v.push_back(x);
      if (v.size() != 20) throw runtime_error("model file " + name + ": expected 20 numbers");
      m.reset(new ComputationModel(Eigen::Vector4d(v[0], v[1], v[2], v[3]),
                                   Eigen::Map<Eigen::Matrix<double, 6, 1>>(&v[4]),
                                   Eigen::Map<Eigen::Matrix<double, 6, 1>>(&v[10]),
                                   Eigen::Vector4d(v[16], v[17], v[18], v[19])));
    }
    return m.get();
  }
  if (name == "cpu") return &ComputationModel::model_OpenBlas_i7_1185g7;
  if (name == "cuda") return &ComputationModel::model_Cuda117_2080Ti;
  if (name == "metal") return &ComputationModel::model_Metal_CI_paravirtual;
  throw runtime_error("unknown model " + name + " (cpu, cuda, metal)");
}

// for each model: analysis (exact with that model, and fast) and estimated factor time
static void runModels(const Problem& p, const Opts& o) {
  for (const string& m : o.models) {
    const ComputationModel* model = namedModel(m);
    printf("%-28s model=%-5s", p.name.c_str(), m.c_str());
    for (bool fast : {false, true}) {
      if (!fast && o.fastOnly) {
        printf(" exact: analysis=%9.4fs factor(model)=%9.4fs sparse(model)=%9.4fs nnzL=0 lumps=0 |",
               0.0, 0.0, 0.0);
        continue;
      }
      Settings settings;
      settings.backend = BackendRef;  // not used: the analysis only depends on the model
      settings.numThreads = o.numThreads;
      settings.findSparseEliminationRanges = o.autoElim;
      settings.computationModel = model;
      settings.analysis = fast ? AnalysisFast : AnalysisExact;
      vector<int64_t> elimRanges;
      if (o.givenElim && p.numElim > 0) elimRanges = {0, p.numElim};
      double tCreate = 1e30;
      SolverPtr solver;
      for (int r = 0; r < o.reps; r++) {
        solver.reset();
        auto t0 = hrc::now();
        solver = createSolver(settings, p.paramSize, p.ss, elimRanges);
        tCreate = min(tCreate, tdelta(hrc::now() - t0).count());
      }
      auto [tFactor, tSparse] = modelledFactorTime(*solver, *model);
      printf(" %s: analysis=%9.4fs factor(model)=%9.4fs sparse(model)=%9.4fs nnzL=%lld lumps=%lld |",
             fast ? "fast" : "exact", tCreate, tFactor, tSparse, (long long)solver->dataSize(),
             (long long)solver->skel().numLumps());
    }
    printf("\n");
    fflush(stdout);
  }
}

// flops of the supernodal factorization (potrf, trsm, syrk/gemm of each lump, at the scalar level
// including the explicit zeros): deterministic proxy of the factor cost, to compare analyses
static double factorFlops(const CoalescedBlockMatrixSkel& skel) {
  double flops = 0;
  for (int64_t l = 0; l < skel.numLumps(); l++) {
    double n = skel.lumpStart[l + 1] - skel.lumpStart[l];
    double colData = skel.chainData[skel.chainColPtr[l + 1]] - skel.chainData[skel.chainColPtr[l]];
    double r = colData / n - n;  // rows below the diagonal block
    flops += n * n * n / 3 + n * n * r + n * r * r;
  }
  return flops;
}

static double minOf(const vector<double>& v) { return *min_element(v.begin(), v.end()); }

// returns false if the solution is inaccurate (or not finite)
static bool runBackend(const Problem& p, const Opts& o, BackendType backend, const string& label) {
  Settings settings;
  settings.backend = backend;
  settings.numThreads = o.numThreads;
  settings.findSparseEliminationRanges = o.autoElim;
  if (!o.cm.empty()) settings.computationModel = namedModel(o.cm);
  if (o.analysis == "exact") settings.analysis = AnalysisExact;
  if (o.analysis == "fast") settings.analysis = AnalysisFast;
  vector<int64_t> elimRanges;
  if (o.givenElim && p.numElim > 0) elimRanges = {0, p.numElim};

  vector<double> tCreate;
  SolverPtr solver;
  for (int r = 0; r < o.reps + 1; r++) {  // first run is "cold"
    solver.reset();
    auto t0 = hrc::now();
    solver = createSolver(settings, p.paramSize, p.ss, elimRanges);
    tCreate.push_back(tdelta(hrc::now() - t0).count());
  }
  solver->enableStats(false);
  if (o.createOnly) {
    printf("%-28s %-8s create(cold)=%9.4fs create=%9.4fs nnzL=%lld lumps=%lld skelHash=%016llx "
           "flops=%.4g\n",
           p.name.c_str(), label.c_str(), tCreate[0],
           minOf(vector<double>(tCreate.begin() + 1, tCreate.end())),
           (long long)solver->dataSize(), (long long)solver->skel().numLumps(),
           (unsigned long long)skelHash(*solver), factorFlops(solver->skel()));
    return true;
  }

  auto acc = solver->accessor();
  vector<double> data(solver->dataSize(), 0.0);
  fillRandom(p, acc, data);
  int64_t order = solver->order();
  vector<double> xTrue(order), b, x;
  mt19937 gen(11);
  uniform_real_distribution<double> dist(-1.0, 1.0);
  for (auto& v : xTrue) v = dist(gen);
  spmv(p, acc, data, xTrue, b);

  vector<double> tFactor, tSolve;
  vector<double> factored;
  FILE* collect = nullptr;
  if (!o.collect.empty()) {
    collect = fopen(o.collect.c_str(), "a");
    auto& sym = solver->internalSymbolicContext();
    sym.potrfStat.callBack = [=](double t, int, int n) {
      fprintf(collect, "potrf %d %.9g\n", n, t);
    };
    sym.trsmStat.callBack = [=](double t, int, int n, int k) {
      fprintf(collect, "trsm %d %d %.9g\n", n, k, t);
    };
    sym.sygeStat.callBack = [=](double t, int, int m, int n, int k) {
      fprintf(collect, "syge %d %d %d %.9g\n", m, n, k, t);
    };
    sym.asmblStat.callBack = [=](double t, int, int br, int bc) {
      fprintf(collect, "asmbl %d %d %.9g\n", br, bc, t);
    };
  }
  for (int r = 0; r < o.reps + 1; r++) {
    solver->enableStats(collect && r > 0);
    vector<double> work = data;
    x = b;
    double tf, ts;
    if (backend == BackendCuda) {
#ifdef BASPACHO_USE_CUBLAS
      DevMirror<double> dWork(work), dX(x);
      cuCHECK(cudaDeviceSynchronize());
      auto t0 = hrc::now();
      solver->factor(dWork.ptr);
      cuCHECK(cudaDeviceSynchronize());
      tf = tdelta(hrc::now() - t0).count();
      t0 = hrc::now();
      solver->solve(dWork.ptr, dX.ptr, order, 1);
      cuCHECK(cudaDeviceSynchronize());
      ts = tdelta(hrc::now() - t0).count();
      dX.get(x);
#endif
    } else {
      auto t0 = hrc::now();
      solver->factor(work.data());
      tf = tdelta(hrc::now() - t0).count();
      t0 = hrc::now();
      solver->solve(work.data(), x.data(), order, 1);
      ts = tdelta(hrc::now() - t0).count();
    }
    if (r > 0) {  // first is warm-up
      tFactor.push_back(tf);
      tSolve.push_back(ts);
    }
  }
  double err = 0, nrm = 0;
  for (int64_t i = 0; i < order; i++) {
    err += (x[i] - xTrue[i]) * (x[i] - xTrue[i]);
    nrm += xTrue[i] * xTrue[i];
  }
  double relErr = sqrt(err / nrm);
  if (collect) {
    solver->enableStats(false);
    fclose(collect);
  }
  double createWarm = minOf(vector<double>(tCreate.begin() + 1, tCreate.end()));
  if (!o.cm.empty()) {
    auto [tm, ts] = modelledFactorTime(*solver, *namedModel(o.cm));
    printf("%-28s %-8s model(%s): factor=%9.4fs sparse=%9.4fs\n", p.name.c_str(), label.c_str(),
           o.cm.c_str(), tm, ts);
  }
  printf(
      "%-28s %-8s create(cold)=%9.4fs create=%9.4fs factor=%9.4fs solve=%9.4fs "
      "create/factor=%6.2f nnzL=%lld lumps=%lld elimRanges=%zu relErr=%.1e skelHash=%016llx "
      "flops=%.4g\n",
      p.name.c_str(), label.c_str(), tCreate[0], createWarm, minOf(tFactor), minOf(tSolve),
      createWarm / minOf(tFactor), (long long)solver->dataSize(),
      (long long)solver->skel().numLumps(), solver->sparseEliminationRanges().size(), relErr,
      (unsigned long long)skelHash(*solver), factorFlops(solver->skel()));
  fflush(stdout);
  if (!(relErr <= 1e-6)) {  // also NaN
    fprintf(stderr, "ERROR: large relative error %g on %s/%s\n", relErr, p.name.c_str(),
            label.c_str());
    return false;
  }
  return true;
}

static void help() {
  cout << "InitBench (-bal f.txt | -g2o f.g2o | -mtx f.mtx) [options]\n"
       << "  -name s       name to print\n"
       << "  -reps n       repetitions (default 3)\n"
       << "  -t n          num threads for BackendFast (default 16)\n"
       << "  -cuda         also run BackendCuda\n"
       << "  -createOnly   only benchmark solver creation\n"
       << "  -nofast       skip BackendFast\n"
       << "  -cholmod      also run CHOLMOD (analysis+factor)\n"
       << "  -noGivenElim  for BAL, don't pass the points as a given elimination range\n"
       << "  -noAutoElim   disable automatic detection of sparse elimination ranges\n"
       << "  -grid3d n bs  synthetic n^3 grid, block size bs, 27-point stencil\n"
       << "  -gen spec     synthetic problem, see makeGenerated (eg. knn3d:20000:8:3)\n"
       << "  -model list   only analysis, factor time estimated by the computation models in the\n"
       << "                comma separated list (cpu, cuda, metal, file:path), exact and fast\n"
       << "  -cm name      computation model of the analysis in factor runs (and its estimate)\n"
       << "  -analysis a   exact or fast (default auto)\n"
       << "  -collect f    append the timings of the factor operations to f\n"
       << "  -maxbs n      max detected block size for -mtx (default 16)\n";
}

static int run(int argc, char* argv[]) {
  Opts o;
  string bal, g2o, mtx, name;
  int64_t gridN = 0, gridBs = 3;
  string genSpec;
  int maxbs = 16;
  for (int i = 1; i < argc; i++) {
    string a = argv[i];
    auto next = [&]() -> string {
      if (i + 1 >= argc) throw runtime_error("missing value for " + a);
      return argv[++i];
    };
    if (a == "-bal") bal = next();
    else if (a == "-g2o") g2o = next();
    else if (a == "-mtx") mtx = next();
    else if (a == "-grid3d") gridN = stoi(next()), gridBs = stoi(next());
    else if (a == "-gen") genSpec = next();
    else if (a == "-fastOnly") o.fastOnly = true;
    else if (a == "-cm") o.cm = next();
    else if (a == "-analysis") o.analysis = next();
    else if (a == "-collect") o.collect = next();
    else if (a == "-model") {
      stringstream ms(next());
      string m;
      while (getline(ms, m, ',')) o.models.push_back(m);
    }
    else if (a == "-name") name = next();
    else if (a == "-reps") o.reps = stoi(next());
    else if (a == "-t") o.numThreads = stoi(next());
    else if (a == "-cuda") o.cuda = true;
    else if (a == "-createOnly") o.createOnly = true;
    else if (a == "-nofast") o.fast = false;
    else if (a == "-cholmod") o.cholmod = true;
    else if (a == "-noGivenElim") o.givenElim = false;
    else if (a == "-noAutoElim") o.autoElim = false;
    else if (a == "-maxbs") maxbs = stoi(next());
    else {
      help();
      return a == "-h" ? 0 : 1;
    }
  }
  if (o.reps < 1 || o.numThreads < 1 || maxbs < 1 || gridN < 0 || gridBs < 1) {
    fprintf(stderr, "invalid option: -reps, -t, -maxbs and the -grid3d sizes must be positive\n");
    return 1;
  }
  Problem p;
  auto t0 = hrc::now();
  if (!bal.empty()) p = loadBal(bal, 9, 3), p.name = bal;
  else if (!g2o.empty()) p = loadG2o(g2o), p.name = g2o;
  else if (!mtx.empty()) p = loadMtx(mtx, maxbs), p.name = mtx;
  else if (gridN > 0) p = makeGrid3d(gridN, gridBs);
  else if (!genSpec.empty()) p = makeGenerated(genSpec);
  else {
    help();
    return 1;
  }
  if (!name.empty()) p.name = name;
  else if (p.name.find('/') != string::npos) p.name = p.name.substr(p.name.find_last_of('/') + 1);
  int64_t totSize = 0;
  for (auto s : p.paramSize) totSize += s;
  fprintf(stderr, "# %s: %zu blocks, order %lld, %zu block-nnz (lower), loaded in %.2fs\n",
          p.name.c_str(), p.paramSize.size(), (long long)totSize, p.ss.inds.size(),
          tdelta(hrc::now() - t0).count());

  bool ok = true;
  if (!o.models.empty()) {
    runModels(p, o);
    return 0;
  }
  if (o.fast) ok &= runBackend(p, o, BackendFast, "fast" + to_string(o.numThreads));
#ifdef BASPACHO_USE_CUBLAS
  if (o.cuda) {
    {  // init cuda context outside of timings
      Settings s;
      s.backend = BackendCuda;
      auto warm = createSolver(s, {1}, SparseStructure({0, 1}, {0}));
    }
    ok &= runBackend(p, o, BackendCuda, "cuda");
  }
#endif
#ifdef BASPACHO_HAVE_CHOLMOD
  if (o.cholmod) {
    auto res = benchmarkCholmodSolve(p.paramSize, p.ss, {}, 0);
    printf("%-28s %-8s analysis=%9.4fs factor=%9.4fs analysis/factor=%6.2f\n", p.name.c_str(),
           "cholmod", res.analysisTime, res.factorTime, res.analysisTime / res.factorTime);
  }
#endif
  return ok ? 0 : 1;  // inaccurate solutions fail (eg. in CI)
}

int main(int argc, char* argv[]) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    fprintf(stderr, "ERROR: %s\n", e.what());
    return 1;
  }
}
