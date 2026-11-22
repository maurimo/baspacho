/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "baspacho/baspacho/MergeOptimizer.h"
#include <algorithm>
#include <queue>
#include <random>
#include <tuple>
#include "baspacho/baspacho/DebugMacros.h"

namespace BaSpaCho {

using namespace std;

MergeOptimizer::MergeOptimizer(const ComputationModel& model, vector<int64_t> parent_,
                               vector<int64_t> size_, vector<int64_t> rows_,
                               vector<bool> canMerge_,
                               vector<bool> sparse_, const vector<vector<Board>>& colBoards)
    : model(model),
      numNodes(parent_.size()),
      parent(std::move(parent_)),
      size(std::move(size_)),
      rows(std::move(rows_)),
      canMerge(std::move(canMerge_)),
      sparse(std::move(sparse_)) {
  mergeWith.assign(numNodes, -1);
  perRow.resize(numNodes);
  colSyge.assign(numNodes, Eigen::Vector2d::Zero());
  colAsmbl.assign(numNodes, 0.0);
  for (int64_t c = 0; c < numNodes; c++) {
    if (sparse[c]) {
      continue;  // sparse elimination: not part of the supernodal factor
    }
    for (const Board& b : colBoards[c]) {
      BASPACHO_CHECK_GT(b.c, c);
      perRow[b.c].push_back(Board{(int32_t)c, b.bc, b.m, b.brDown, b.rowsDown});
      colSyge[c] += model.sygeLinEst(b.m, b.m + b.rowsDown);
      colAsmbl[c] += asmbl(b.bc + b.brDown, b.bc, b.m, b.m + b.rowsDown);
    }
    total += colCost(c, size[c]);
  }
}

double MergeOptimizer::colCost(int64_t c, double n) const {
  // (factorLump calls no trsm without rows below the diagonal)
  return model.potrfEst(n) + (rows[c] > 0 ? model.trsmEst(n, rows[c]) : 0.0) + colSyge[c][0] +
         colSyge[c][1] * n + colAsmbl[c];
}

int64_t MergeOptimizer::root(int64_t a) {
  int64_t r = a;
  while (mergeWith[r] != -1) {
    r = mergeWith[r];
  }
  while (a != r) {
    int64_t next = mergeWith[a];
    mergeWith[a] = r;
    a = next;
  }
  return r;
}

double MergeOptimizer::mergeDelta(int64_t k, int64_t p) {
  double delta = colCost(p, size[k] + size[p]) - colCost(k, size[k]) - colCost(p, size[p]);
  const auto& kB = perRow[k];
  const auto& pB = perRow[p];
  for (size_t ik = 0, ip = 0; ik < kB.size() && ip < pB.size();) {
    if (kB[ik].c < pB[ip].c) {
      ik++;
    } else if (kB[ik].c > pB[ip].c) {
      ip++;
    } else {
      int64_t c = kB[ik].c;
      if (mergeWith[c] == -1) {
        const Board& e1 = kB[ik];
        const Board& e2 = pB[ip];
        double n = size[c];
        double m = e1.m + e2.m, bc = e1.bc + e2.bc;
        delta += model.sygeEst(m, m + e2.rowsDown, n) +
                 asmbl(bc + e2.brDown, bc, m, m + e2.rowsDown) -
                 model.sygeEst(e1.m, e1.m + e1.rowsDown, n) -
                 asmbl(e1.bc + e1.brDown, e1.bc, e1.m, e1.m + e1.rowsDown) -
                 model.sygeEst(e2.m, e2.m + e2.rowsDown, n) -
                 asmbl(e2.bc + e2.brDown, e2.bc, e2.m, e2.m + e2.rowsDown);
      }
      ik++;
      ip++;
    }
  }
  return delta;
}

void MergeOptimizer::applyMerge(int64_t k, int64_t p) {
  total += mergeDelta(k, p);
  mergeWith[k] = p;
  size[p] += size[k];
  numMerges++;

  auto& kB = perRow[k];
  auto& pB = perRow[p];
  tmp.clear();
  tmp.reserve(kB.size() + pB.size());
  auto alive = [&](int64_t c) { return mergeWith[c] == -1; };  // (k is no longer alive)
  for (size_t ik = 0, ip = 0; ik < kB.size() || ip < pB.size();) {
    if (ip >= pB.size() || (ik < kB.size() && kB[ik].c < pB[ip].c)) {
      if (alive(kB[ik].c)) {
        tmp.push_back(kB[ik]);
      }
      ik++;
    } else if (ik >= kB.size() || kB[ik].c > pB[ip].c) {
      if (alive(pB[ip].c)) {
        tmp.push_back(pB[ip]);
      }
      ip++;
    } else {
      int64_t c = kB[ik].c;
      if (alive(c)) {
        const Board& e1 = kB[ik];
        const Board& e2 = pB[ip];
        Board m{(int32_t)c, e1.bc + e2.bc, e1.m + e2.m, e2.brDown, e2.rowsDown};
        colSyge[c] += model.sygeLinEst(m.m, m.m + m.rowsDown) -
                      model.sygeLinEst(e1.m, e1.m + e1.rowsDown) -
                      model.sygeLinEst(e2.m, e2.m + e2.rowsDown);
        colAsmbl[c] += asmbl(m.bc + m.brDown, m.bc, m.m, m.m + m.rowsDown) -
                       asmbl(e1.bc + e1.brDown, e1.bc, e1.m, e1.m + e1.rowsDown) -
                       asmbl(e2.bc + e2.brDown, e2.bc, e2.m, e2.m + e2.rowsDown);
        tmp.push_back(m);
      }
      ik++;
      ip++;
    }
  }
  swap(pB, tmp);
  kB = vector<Board>();
}

void MergeOptimizer::steepest(double noise, uint64_t seed) {
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> unif(-1.0, 1.0);
  // children lumps of each lump
  vector<vector<int64_t>> kids(numNodes);
  auto isCandidate = [&](int64_t k) {
    return mergeWith[k] == -1 && canMerge[k] && !sparse[k] && parent[k] != -1;
  };
  for (int64_t k = 0; k < numNodes; k++) {
    if (mergeWith[k] == -1 && parent[k] != -1) {
      kids[root(parent[k])].push_back(k);
    }
  }
  // (-delta, k, version of k's entry): stale entries are skipped
  vector<int64_t> version(numNodes, 0);
  priority_queue<tuple<double, int64_t, int64_t>> heap;
  auto rank = [&](double d) { return noise > 0 ? d * (1.0 + noise * unif(gen)) : d; };
  auto push = [&](int64_t k) {
    if (isCandidate(k)) {
      heap.emplace(-rank(mergeDelta(k, root(parent[k]))), k, ++version[k]);
    }
  };
  auto refreshAll = [&]() {
    heap = decltype(heap)();
    for (int64_t k = 0; k < numNodes; k++) {
      push(k);
    }
  };
  refreshAll();
  while (true) {
    if (heap.empty() || get<0>(heap.top()) <= 0) {
      // no improving merge according to the (maybe stale) values: refresh all
      refreshAll();
      if (heap.empty() || get<0>(heap.top()) <= 0) {
        break;
      }
    }
    auto [negD, k, ver] = heap.top();
    heap.pop();
    if (ver != version[k] || !isCandidate(k)) {
      continue;
    }
    int64_t p = root(parent[k]);
    double d = mergeDelta(k, p);
    double r = rank(d);
    if (!heap.empty() && -r < get<0>(heap.top())) {
      heap.emplace(-r, k, ++version[k]);  // no longer the best
      continue;
    }
    if (d >= 0) {
      heap.emplace(-r, k, ++version[k]);
      continue;  // (the refresh above decides when to stop)
    }
    applyMerge(k, p);
    auto& pk = kids[p];
    pk.erase(std::find(pk.begin(), pk.end(), k));
    pk.insert(pk.end(), kids[k].begin(), kids[k].end());
    kids[k] = vector<int64_t>();
    push(p);
    for (int64_t c : pk) {
      push(c);
    }
  }
}

}  // end namespace BaSpaCho
