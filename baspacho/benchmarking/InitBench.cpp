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
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
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
  if (fscanf(f, "%ld %ld %ld", &nCams, &nPts, &nObs) != 3) throw runtime_error("bad bal header");
  vector<pair<int64_t, int64_t>> pairs;
  pairs.reserve(nObs);
  for (long i = 0; i < nObs; i++) {
    long c, p;
    double x, y;
    if (fscanf(f, "%ld %ld %lf %lf", &c, &p, &x, &y) != 4) throw runtime_error("bad bal obs");
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
    ls >> tag;
    if (tag.rfind("VERTEX", 0) == 0) {
      int64_t id;
      ls >> id;
      int sz = tag.find("SE2") != string::npos || tag.find("XY") != string::npos ? 3 : 6;
      if (tag == "VERTEX_XY") sz = 2;
      if (tag == "VERTEX_TRACKXYZ" || tag == "VERTEX_POINT_XYZ") sz = 3;
      if (idToIdx.emplace(id, sizes.size()).second) sizes.push_back(sz);
    } else if (tag.rfind("EDGE", 0) == 0) {
      int64_t a, b;
      ls >> a >> b;
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
  if (nr != nc) throw runtime_error("matrix not square");
  getline(in, line);
  vector<vector<int32_t>> adj(nr);  // full symmetric adjacency incl. diagonal
  for (int64_t k = 0; k < nnz; k++) {
    if (!getline(in, line)) break;
    const char* s = line.c_str();
    char* e;
    int64_t i = strtoll(s, &e, 10) - 1;
    int64_t j = strtoll(e, &e, 10) - 1;
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

static double minOf(const vector<double>& v) { return *min_element(v.begin(), v.end()); }

static void runBackend(const Problem& p, const Opts& o, BackendType backend, const string& label) {
  Settings settings;
  settings.backend = backend;
  settings.numThreads = o.numThreads;
  settings.findSparseEliminationRanges = o.autoElim;
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
    printf("%-28s %-8s create(cold)=%9.4fs create=%9.4fs nnzL=%lld lumps=%lld skelHash=%016llx\n",
           p.name.c_str(), label.c_str(), tCreate[0],
           minOf(vector<double>(tCreate.begin() + 1, tCreate.end())),
           (long long)solver->dataSize(), (long long)solver->skel().numLumps(),
           (unsigned long long)skelHash(*solver));
    return;
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
  for (int r = 0; r < o.reps + 1; r++) {
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
  double createWarm = minOf(vector<double>(tCreate.begin() + 1, tCreate.end()));
  printf(
      "%-28s %-8s create(cold)=%9.4fs create=%9.4fs factor=%9.4fs solve=%9.4fs "
      "create/factor=%6.2f nnzL=%lld lumps=%lld elimRanges=%zu relErr=%.1e skelHash=%016llx\n",
      p.name.c_str(), label.c_str(), tCreate[0], createWarm, minOf(tFactor), minOf(tSolve),
      createWarm / minOf(tFactor), (long long)solver->dataSize(),
      (long long)solver->skel().numLumps(), solver->sparseEliminationRanges().size(), relErr,
      (unsigned long long)skelHash(*solver));
  fflush(stdout);
  if (relErr > 1e-6) {
    fprintf(stderr, "ERROR: large relative error %g on %s/%s\n", relErr, p.name.c_str(),
            label.c_str());
  }
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
       << "  -maxbs n      max detected block size for -mtx (default 16)\n";
}

int main(int argc, char* argv[]) {
  Opts o;
  string bal, g2o, mtx, name;
  int64_t gridN = 0, gridBs = 3;
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
  Problem p;
  auto t0 = hrc::now();
  if (!bal.empty()) p = loadBal(bal, 9, 3), p.name = bal;
  else if (!g2o.empty()) p = loadG2o(g2o), p.name = g2o;
  else if (!mtx.empty()) p = loadMtx(mtx, maxbs), p.name = mtx;
  else if (gridN > 0) p = makeGrid3d(gridN, gridBs);
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

  if (o.fast) runBackend(p, o, BackendFast, "fast" + to_string(o.numThreads));
#ifdef BASPACHO_USE_CUBLAS
  if (o.cuda) {
    {  // init cuda context outside of timings
      Settings s;
      s.backend = BackendCuda;
      auto warm = createSolver(s, {1}, SparseStructure({0, 1}, {0}));
    }
    runBackend(p, o, BackendCuda, "cuda");
  }
#endif
#ifdef BASPACHO_HAVE_CHOLMOD
  if (o.cholmod) {
    auto res = benchmarkCholmodSolve(p.paramSize, p.ss, {}, 0);
    printf("%-28s %-8s analysis=%9.4fs factor=%9.4fs analysis/factor=%6.2f\n", p.name.c_str(),
           "cholmod", res.analysisTime, res.factorTime, res.analysisTime / res.factorTime);
  }
#endif
  return 0;
}
