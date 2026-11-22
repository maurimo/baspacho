/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "baspacho/baspacho/EliminationTree.h"
#include <algorithm>
#include <queue>
#include <sstream>
#include <string>
#include <tuple>
#include "baspacho/baspacho/ComputationModel.h"
#include "baspacho/baspacho/DebugMacros.h"
#include "baspacho/baspacho/Utils.h"

namespace BaSpaCho {

using namespace std;

EliminationTree::EliminationTree(const vector<int64_t>& paramSize, const SparseStructure& ss,
                                 const ComputationModel* compMod)
    : paramSize(paramSize),
      ss(ss),
      compMod(compMod ? *compMod : ComputationModel::model_OpenBlas_i7_1185g7) {
  BASPACHO_CHECK_EQ(paramSize.size(), ss.ptrs.size() - 1);
}

void EliminationTree::buildTree() {
  int64_t ord = ss.order();
  parent.assign(ord, -1);

  nodeSize = paramSize;
  nodeRows.assign(ord, 0);
  nodeRowBlocks.assign(ord, 0);
  vector<int64_t> tags(ord);

  // skeleton of the algo to iterate on fillup's nodes is from Eigen's
  // `SimplicialCholesky_impl.h` (by Gael Guennebaud),
  // in turn from LDL by Timothy A. Davis.
  // The off-diagonal structure of L is collected by rows (L(k,i), i < k) in flat
  // arrays, and then transposed to get the (sorted) columns.
  vector<int64_t> lRowPtr(ord + 1, 0), lRowCols;
  lRowCols.reserve(2 * ss.inds.size());
  for (int64_t k = 0; k < ord; ++k) {
    /* L(k,:) pattern: all nodes reachable in etree from nz in A(0:k-1,k) */
    parent[k] = -1; /* parent of k is not yet known */
    tags[k] = k;    /* mark node k as visited */
    /* L(k,k) is nonzero */

    int64_t start = ss.ptrs[k];
    int64_t end = ss.ptrs[k + 1];
    for (int64_t q = start; q < end; q++) {
      int64_t i = ss.inds[q];
      if (i >= k) {
        continue;
      }
      /* follow path from i to root of etree, stop at flagged node */
      for (; tags[i] != k; i = parent[i]) {
        /* find parent of i if not yet determined */
        if (parent[i] == -1) {
          parent[i] = k;
        }

        tags[i] = k;
        /* L(k,i) is nonzero */
        nodeRows[i] += paramSize[k];
        nodeRowBlocks[i]++;
        lRowCols.push_back(i);
      }
    }
    lRowPtr[k + 1] = lRowCols.size();
  }

  // columns of L, including the diagonal entry (first), rows are sorted
  vector<int64_t> lColPtr(ord + 1);
  lColPtr[0] = 0;
  for (int64_t i = 0; i < ord; i++) {
    lColPtr[i + 1] = lColPtr[i] + nodeRowBlocks[i] + 1;
  }
  vector<int64_t> lColRows(lColPtr[ord]);
  {
    vector<int64_t> pos(lColPtr.begin(), lColPtr.end() - 1);
    for (int64_t i = 0; i < ord; i++) {
      lColRows[pos[i]++] = i;
    }
    for (int64_t k = 0; k < ord; k++) {
      for (int64_t q = lRowPtr[k], qEnd = lRowPtr[k + 1]; q < qEnd; q++) {
        lColRows[pos[lRowCols[q]]++] = k;
      }
    }
  }
  lRowCols = vector<int64_t>();  // release memory

  sygeCosts.resize(ord);
  asmblCosts.resize(ord);
  perRowNodeStats.resize(ord);
  for (int64_t row = 0; row < ord; row++) {
    perRowNodeStats[row].clear();
    perRowNodeStats[row].reserve(lRowPtr[row + 1] - lRowPtr[row] + 1);
  }
  for (int64_t col = 0; col < ord; col++) {
    int64_t skippedRows = 0;
    int64_t skippedBlocks = 0;
    Eigen::Vector2d sygeC{0, 0}, asmblC{0, 0};
    for (int64_t q = lColPtr[col + 1] - 1; q >= lColPtr[col]; q--) {
      int64_t row = lColRows[q];

      sygeC += compMod.sygeLinEst(skippedRows + paramSize[row], paramSize[row]);
      asmblC += compMod.asmblLinEst(skippedBlocks + 1);

      perRowNodeStats[row].emplace_back(col, 1, paramSize[row], skippedBlocks, skippedRows);

      skippedRows += paramSize[row];
      skippedBlocks++;
    }
    sygeCosts[col] = sygeC;
    asmblCosts[col] = asmblC;
  }
}

void EliminationTree::buildTreeFast() {
  int64_t ord = ss.order();
  fastSymbolic = true;
  nodeSize = paramSize;

  // elimination tree (Liu, with path compression), ss is the lower half by rows
  parent.assign(ord, -1);
  vector<int64_t> ancestor(ord, -1);
  for (int64_t k = 0; k < ord; k++) {
    for (int64_t q = ss.ptrs[k], qEnd = ss.ptrs[k + 1]; q < qEnd; q++) {
      for (int64_t i = ss.inds[q], inext; i != -1 && i < k; i = inext) {
        inext = ancestor[i];
        ancestor[i] = k;
        if (inext == -1) {
          parent[i] = k;
        }
      }
    }
  }

  // postorder (iterative dfs on the children lists)
  vector<int64_t> head(ord, -1), next(ord, -1), post(ord), stack;
  for (int64_t j = ord - 1; j >= 0; j--) {
    if (parent[j] != -1) {
      next[j] = head[parent[j]];
      head[parent[j]] = j;
    }
  }
  int64_t k = 0;
  for (int64_t r = 0; r < ord; r++) {
    if (parent[r] != -1) {
      continue;
    }
    stack.push_back(r);
    while (!stack.empty()) {
      int64_t p = stack.back();
      int64_t c = head[p];
      if (c == -1) {
        stack.pop_back();
        post[k++] = p;
      } else {
        head[p] = next[c];
        stack.push_back(c);
      }
    }
  }
  BASPACHO_CHECK_EQ(k, ord);

  // columns of the lower half: rows i > j having an entry in column j
  vector<int64_t> colPtr(ord + 1, 0), colRows;
  for (int64_t r = 0; r < ord; r++) {
    for (int64_t q = ss.ptrs[r], qEnd = ss.ptrs[r + 1]; q < qEnd; q++) {
      if (ss.inds[q] < r) {
        colPtr[ss.inds[q] + 1]++;
      }
    }
  }
  for (int64_t j = 0; j < ord; j++) {
    colPtr[j + 1] += colPtr[j];
  }
  colRows.resize(colPtr[ord]);
  {
    vector<int64_t> pos(colPtr.begin(), colPtr.end() - 1);
    for (int64_t r = 0; r < ord; r++) {
      for (int64_t q = ss.ptrs[r], qEnd = ss.ptrs[r + 1]; q < qEnd; q++) {
        if (ss.inds[q] < r) {
          colRows[pos[ss.inds[q]]++] = r;
        }
      }
    }
  }

  // column counts (Gilbert-Ng-Peyton, as cs_counts), both in blocks and weighted by the param
  // sizes (scalar rows): count(j) = sum over the rows i whose row subtree contains j
  vector<int64_t> first(ord, -1), maxfirst(ord, -1), prevleaf(ord, -1);
  vector<int64_t> delta(ord, 0), wdelta(ord, 0);
  for (int64_t i = 0; i < ord; i++) {
    ancestor[i] = i;
  }
  for (int64_t kk = 0; kk < ord; kk++) {
    int64_t j = post[kk];
    if (first[j] == -1) {  // j is a leaf
      delta[j] = 1;
      wdelta[j] = paramSize[j];
    }
    for (; j != -1 && first[j] == -1; j = parent[j]) {
      first[j] = kk;
    }
  }
  for (int64_t kk = 0; kk < ord; kk++) {
    int64_t j = post[kk];
    if (parent[j] != -1) {
      delta[parent[j]]--;
      wdelta[parent[j]] -= paramSize[j];
    }
    for (int64_t q = colPtr[j], qEnd = colPtr[j + 1]; q < qEnd; q++) {
      int64_t i = colRows[q];
      if (first[j] <= maxfirst[i]) {
        continue;  // A(i,j) is not a leaf of the row subtree of i
      }
      maxfirst[i] = first[j];
      int64_t jprev = prevleaf[i];
      prevleaf[i] = j;
      delta[j]++;
      wdelta[j] += paramSize[i];
      if (jprev != -1) {  // subtract at the least common ancestor of jprev and j
        int64_t lca = jprev;
        while (lca != ancestor[lca]) {
          lca = ancestor[lca];
        }
        for (int64_t s = jprev, sparent; s != lca; s = sparent) {
          sparent = ancestor[s];
          ancestor[s] = lca;
        }
        delta[lca]--;
        wdelta[lca] -= paramSize[i];
      }
    }
    if (parent[j] != -1) {
      ancestor[j] = parent[j];
    }
  }
  for (int64_t kk = 0; kk < ord; kk++) {  // sum the deltas over the subtrees
    int64_t j = post[kk];
    if (parent[j] != -1) {
      delta[parent[j]] += delta[j];
      wdelta[parent[j]] += wdelta[j];
    }
  }
  nodeRowBlocks.resize(ord);
  nodeRows.resize(ord);
  for (int64_t j = 0; j < ord; j++) {
    nodeRowBlocks[j] = delta[j] - 1;  // below the diagonal
    nodeRows[j] = wdelta[j] - paramSize[j];
  }
}

// Relaxed amalgamation (as CHOLMOD's, with larger size thresholds): a child is merged into its
// parent if no explicit zeros are added, or if the merged node is small enough for its fraction
// of explicit zeros: sizes (scalar columns) and zero fractions below. Tuned with InitBench on BA,
// pose graphs, meshes and FEM matrices: same or faster factor than the computation-model merges
static constexpr double kRelaxSize[3] = {8, 32, 96};
static constexpr double kRelaxZeros[3] = {0.8, 0.1, 0.05};

// estimated time of the factor ops of a node of `size` (made of `merged` spans), with `rows` rows
// in `blocks` blocks below the diagonal (assumed of equal size): closed form of the sums over the
// blocks of the computation model terms (experiments)
double EliminationTree::estimatedNodeCost(double size, double rows, double blocks,
                                          double merged) const {
  const auto& sp = compMod.sygeParams;
  const auto& ap = compMod.asmblParams;
  double b = blocks, a = blocks > 0 ? rows / blocks : 0.0, tri = b * (b + 1) / 2;
  double sumMN = a * (tri + b), sumMxN = a * a * tri;
  double syge0 = b * sp[0] + sumMN * sp[1] + sumMxN * sp[2];
  double syge1 = b * sp[3] + sumMN * sp[4] + sumMxN * sp[5];
  double asmbl0 = b * ap[0] + tri * ap[1];
  double asmbl1 = b * ap[2] + tri * ap[3];
  return compMod.potrfEst(size) + compMod.trsmEst(size, rows) + syge0 + syge1 * size + asmbl0 +
         asmbl1 * merged;
}

void EliminationTree::computeMergesFast() {
  int64_t ord = ss.order();
  numMergedNodes.assign(ord, 1);
  mergeWith.assign(ord, -1);
  numMerges = 0;

  // explicit zeros of the (merged) nodes: merging k into p, the columns of k get the rows of p
  // (and its diagonal block) they don't have
  vector<double> zeros(ord, 0.0);
  auto pickUpScore = [&](int64_t k, int64_t p) -> double {
    return ((double)nodeRows[k]) / (nodeRows[p] + nodeSize[p]);
  };
  priority_queue<tuple<double, int64_t, int64_t>> mergeCandidates;
  for (int64_t k = ord - 1; k >= 0; k--) {
    if (forbidMerge[k] || parent[k] == -1) {
      continue;
    }
    mergeCandidates.emplace(pickUpScore(k, parent[k]), k, parent[k]);
  }
  while (!mergeCandidates.empty()) {
    auto [score, k, p] = mergeCandidates.top();
    mergeCandidates.pop();
    int64_t oldP = p;
    p = mergedRoot(p);
    if (oldP != p) {  // parent was merged, re-prioritize
      mergeCandidates.emplace(pickUpScore(k, p), k, p);
      continue;
    }
    double sk = nodeSize[k], sp = nodeSize[p], sm = sk + sp;
    double z = zeros[k] + zeros[p] + sk * std::max(0.0, (double)(nodeRows[p] + sp - nodeRows[k]));
    double frac = z / (sm * (sm + 1) / 2 + sm * nodeRows[p]);
    static const std::string mode = getenv("BASPACHO_MERGE") ? getenv("BASPACHO_MERGE") : "relax";
    static const double zcap = getenv("BASPACHO_ZCAP") ? atof(getenv("BASPACHO_ZCAP")) : 0.3;
    static const std::vector<double> rx = [] {
      std::vector<double> v = {kRelaxSize[0], kRelaxSize[1], kRelaxSize[2],
                               kRelaxZeros[0], kRelaxZeros[1], kRelaxZeros[2]};
      if (const char* e = getenv("BASPACHO_RELAX")) {
        std::stringstream ss(e);
        std::string t;
        for (size_t i = 0; i < v.size() && std::getline(ss, t, ','); i++) v[i] = atof(t.c_str());
      }
      return v;
    }();
    bool relaxed = z == 0 || sm <= rx[0] || (sm <= rx[1] && frac < rx[3]) ||
                   (sm <= rx[2] && frac < rx[4]) || frac < rx[5];
    bool costBetter = false;
    if (mode != "relax") {
      double tk = estimatedNodeCost(sk, nodeRows[k], nodeRowBlocks[k], numMergedNodes[k]);
      double tp = estimatedNodeCost(sp, nodeRows[p], nodeRowBlocks[p], numMergedNodes[p]);
      double tm = estimatedNodeCost(sm, nodeRows[p], nodeRowBlocks[p],
                                    numMergedNodes[p] + numMergedNodes[k]);
      costBetter = tm < tk + tp && frac < zcap;
    }
    bool willMerge = mode == "relax" ? relaxed : mode == "cost" ? costBetter : relaxed || costBetter;
    if (willMerge) {
      zeros[p] = z;
      mergeWith[k] = p;
      nodeSize[p] += nodeSize[k];
      numMergedNodes[p] += numMergedNodes[k];
      numMerges++;
    }
  }
}

// Compute node heights:
// depending on the structure of the matrix, if we have many small leaf
// nodes (height = 0) we can eliminate them with a sparse elimination
// step that skips the node-merge adding fill in. It's then possible
// to eliminate progressively nodes with height=1, 2, etc till when
// this strategy is no longer effective and we're better off with
// merged nodes and blas operations.
// In order to do so we sort nodes according to height.
// TODO: consider allowing some initial merge of very small nodes?
void EliminationTree::computeNodeHeights(const vector<int64_t>& noCrossPoints) {
  int64_t ord = ss.order();

  unmergedHeightNode.resize(ord);
  forbidMerge.assign(ord, false);

  vector<int64_t> height(ord, 0);
  for (size_t rangeIndex = 0; rangeIndex < noCrossPoints.size() + 1; rangeIndex++) {
    int64_t rangeStart = rangeIndex == 0 ? 0 : noCrossPoints[rangeIndex - 1];
    int64_t rangeEnd = rangeIndex < noCrossPoints.size() ? noCrossPoints[rangeIndex] : ord;

    for (int64_t k = rangeStart; k < rangeEnd; k++) {
      unmergedHeightNode[k] = make_tuple(height[k], nodeSize[k], k);

      int64_t par = parent[k];
      if (par == -1) {
        continue;
      }
      if (par >= rangeEnd) {
        forbidMerge[k] = true;
      }
      height[par] = max(height[par], height[k] + 1);
    }
    sort(unmergedHeightNode.begin() + rangeStart, unmergedHeightNode.begin() + rangeEnd);
  }
}

static constexpr int64_t maxSparseElimNodeSize = 12;
static constexpr int64_t minNumSparseElimNodes = 50;

void EliminationTree::computeSparseElimRanges(const vector<int64_t>& noCrossPoints) {
  int64_t ord = ss.order();
  sparseElimRanges.push_back(0);

  for (size_t rangeIndex = 0; rangeIndex < noCrossPoints.size() + 1; rangeIndex++) {
    int64_t rangeStart = rangeIndex == 0 ? 0 : noCrossPoints[rangeIndex - 1];
    int64_t rangeEnd = rangeIndex < noCrossPoints.size() ? noCrossPoints[rangeIndex] : ord;

    int64_t k0 = rangeStart;
    while (k0 < rangeEnd) {
      int64_t k1 = k0;
      int64_t mergeHeight = get<0>(unmergedHeightNode[k0]);

      int64_t numEasyMerge = 0;
      while (k1 < rangeEnd && get<0>(unmergedHeightNode[k1]) == mergeHeight &&
             get<1>(unmergedHeightNode[k1]) <= maxSparseElimNodeSize) {
        int64_t p = parent[k1];
        if (p >= 0) {
          double fillAfterMerge = ((double)nodeRows[k1]) / (nodeRows[p] + nodeSize[p]);
          if (fillAfterMerge > 0.8) {
            numEasyMerge++;
          }
        }
        k1++;
      }

      // skip and stop searching if 1. too small, or 2. most nodes are easily merged
      if ((k1 - k0) < minNumSparseElimNodes || (k1 - k0) < numEasyMerge * 3) {
        break;
      }

      for (int64_t k = k0; k < k1; k++) {
        forbidMerge[get<2>(unmergedHeightNode[k])] = true;
      }
      sparseElimRanges.push_back(k1);
      k0 = k1;
    }
    if (k0 < rangeEnd) {
      break;
    }
  }
  if (sparseElimRanges.size() == 1) {
    sparseElimRanges.pop_back();
  }
}

// the node a node was merged into (following mergeWith), with path compression
int64_t EliminationTree::mergedRoot(int64_t p) {
  int64_t root = p;
  while (mergeWith[root] != -1) {
    root = mergeWith[root];
  }
  while (p != root) {
    int64_t next = mergeWith[p];
    if (next != root) {
      mergeWith[p] = root;
    }
    p = next;
  }
  return root;
}

void EliminationTree::computeMerges() {
  int64_t ord = ss.order();
  numMergedNodes.assign(ord, 1);
  mergeWith.assign(ord, -1);
  numMerges = 0;

  // score used to select which child/parent pair we will try to merge first
  auto pickUpScore = [&](int64_t k, int64_t p) -> double {
    return ((double)nodeRows[k]) / (nodeRows[p] + nodeSize[p]);
  };

  priority_queue<tuple<double, int64_t, int64_t>> mergeCandidates;
  for (int64_t k = ord - 1; k >= 0; k--) {
    if (forbidMerge[k]) {
      continue;
    }
    int64_t p = parent[k];
    if (p == -1) {
      continue;
    }
    mergeCandidates.emplace(pickUpScore(k, p), k, p);
  }

  while (!mergeCandidates.empty()) {
    auto wasFillAfterMergeKP = mergeCandidates.top();
    int64_t k = std::get<1>(wasFillAfterMergeKP);
    int64_t p = std::get<2>(wasFillAfterMergeKP);
    mergeCandidates.pop();

    auto oldP = p;
    p = mergedRoot(p);

    // parent was merged? value changed, re-prioritize
    if (oldP != p) {
      mergeCandidates.emplace(pickUpScore(k, p), k, p);
      continue;
    }

    double sk = nodeSize[k], rk = nodeRows[k], sp = nodeSize[p], rp = nodeRows[p], sm = sp + sk;

    // To decide if we're merging the nodes, we compare the estimated runtimes of the (factor)
    // operations related to the individual unmerged nodes with the runtime of the operations
    // in the merged node. This isn't 100% accurate because some children elimination operations
    // might also merge and this is not accounted, but the current estimate should be rather
    // accurate.
    double tk = compMod.potrfEst(sk) + compMod.trsmEst(sk, rk) +  //
                sygeCosts[k][0] + sygeCosts[k][1] * sk +          //
                asmblCosts[k][0] + asmblCosts[k][1] * numMergedNodes[k];
    double tp = compMod.potrfEst(sp) + compMod.trsmEst(sp, rp) +  //
                sygeCosts[p][0] + sygeCosts[p][1] * sp +          //
                asmblCosts[p][0] + asmblCosts[p][1] * numMergedNodes[p];
    double tm = compMod.potrfEst(sm) + compMod.trsmEst(sm, rp) +  //
                sygeCosts[p][0] + sygeCosts[p][1] * sm +          //
                asmblCosts[p][0] + asmblCosts[p][1] * (numMergedNodes[k] + numMergedNodes[p]);
    bool willMerge = tm < tk + tp;

    if (willMerge) {
      int64_t prevNodeSize = nodeSize[p];
      int64_t prevNumMergedNodes = numMergedNodes[p];
      mergeWith[k] = p;
      nodeSize[p] += nodeSize[k];
      numMergedNodes[p] += numMergedNodes[k];
      numMerges++;

      auto& kRD = perRowNodeStats[k];
      auto& pRD = perRowNodeStats[p];
      for (size_t ik = 0, ip = 0; ik < kRD.size() || ip < pRD.size(); /* */) {
        if ((ip >= pRD.size()) || (ik < kRD.size() && kRD[ik].colIdx < pRD[ip].colIdx)) {
          if (kRD[ik].colIdx != k) {
            tmpRowStats.push_back(kRD[ik]);
          }
          ik++;
        } else if ((ik >= kRD.size()) || (ip < pRD.size() && kRD[ik].colIdx > pRD[ip].colIdx)) {
          if (pRD[ip].colIdx != p) {
            tmpRowStats.push_back(pRD[ip]);
          }
          ip++;
        } else {
          int64_t c = pRD[ip].colIdx;
          auto& sygeC = sygeCosts[c];
          auto& asmblC = asmblCosts[c];
          sygeC -= compMod.sygeLinEst(kRD[ik].rowsDown + kRD[ik].rows, kRD[ik].rows);
          asmblC -= compMod.asmblLinEst(kRD[ik].rBlocksDown + kRD[ik].rBlocks);
          sygeC -= compMod.sygeLinEst(pRD[ip].rowsDown + pRD[ip].rows, pRD[ip].rows);
          asmblC -= compMod.asmblLinEst(pRD[ip].rBlocksDown + pRD[ip].rBlocks);
          sygeC += compMod.sygeLinEst(pRD[ip].rowsDown + (kRD[ik].rows + pRD[ip].rows),
                                      (kRD[ik].rows + pRD[ip].rows));
          asmblC += compMod.asmblLinEst(pRD[ip].rBlocksDown + (kRD[ik].rBlocks + pRD[ip].rBlocks));

          tmpRowStats.emplace_back(c, kRD[ik].rBlocks + pRD[ip].rBlocks,
                                   kRD[ik].rows + pRD[ip].rows, pRD[ip].rBlocksDown,
                                   pRD[ip].rowsDown);
          ik++;
          ip++;
        }
      }
      auto& sygeC = sygeCosts[p];
      auto& asmblC = asmblCosts[p];
      sygeC -= compMod.sygeLinEst(nodeRows[p] + prevNodeSize, prevNodeSize);
      asmblC -= compMod.asmblLinEst(nodeRowBlocks[p] + prevNumMergedNodes);
      sygeC += compMod.sygeLinEst(nodeRows[p] + nodeSize[p], nodeSize[p]);
      asmblC += compMod.asmblLinEst(nodeRowBlocks[p] + numMergedNodes[p]);
      tmpRowStats.emplace_back(p, numMergedNodes[p], nodeSize[p], nodeRowBlocks[p], nodeRows[p]);
      swap(perRowNodeStats[p], tmpRowStats);
      tmpRowStats.clear();
    }  // willmerge
  }
}

// collapse pointer to parent, make parent become root ancestor
void EliminationTree::collapseMergePointers() {
  int64_t ord = ss.order();

  for (int64_t k = ord - 1; k >= 0; k--) {
    int64_t p = mergeWith[k];
    if (p == -1) {
      continue;
    }
    int64_t a = mergeWith[p];
    if (a != -1) {
      mergeWith[k] = a;
    }
  }
}

void EliminationTree::processTree(bool detectSparseElimRanges, const vector<int64_t>& noCrossPoints,
                                  bool findOnlyElims) {
  int64_t ord = ss.order();

  computeNodeHeights(noCrossPoints);

  // Compute the sparse elimination ranges (after permutation is applied),
  // and set a flag to forbid merge of nodes which will be sparse-eliminated
  if (detectSparseElimRanges) {
    computeSparseElimRanges(noCrossPoints);
  }

  if (findOnlyElims) {
    mergeWith.assign(ord, -1);
    numMergedNodes.assign(ord, 1);
    numMerges = 0;
  } else {
    if (fastSymbolic) {
      computeMergesFast();
    } else {
      computeMerges();
    }
    collapseMergePointers();
  }

  // We create `lumpStart` and `lumpToSpan` arrays for the
  // permuted aggregated parameters.
  // We also set an `unpermutedRootSpanToLump` to register
  // the lumpIndex of an unpermuted root node.
  int64_t numLumps = ord - numMerges, lumpIndex = 0;
  lumpStart.resize(numLumps + 1);   // permuted
  lumpToSpan.resize(numLumps + 1);  // permuted
  vector<int64_t> unpermutedRootSpanToLump(ord, -1);

  for (int64_t i = 0; i < ord; i++) {
    int64_t k = std::get<2>(unmergedHeightNode[i]);
    if (mergeWith[k] != -1) {
      continue;
    }
    unpermutedRootSpanToLump[k] = lumpIndex;
    lumpStart[lumpIndex] = nodeSize[k];
    lumpToSpan[lumpIndex] = numMergedNodes[k];
    lumpIndex++;
  }
  BASPACHO_CHECK_EQ(lumpIndex, numLumps);

  cumSumVec(lumpStart);
  cumSumVec(lumpToSpan);

  // we compute the (inverse) permutation: for each span we have a pointer
  // to the beginning of the lump, it is advanced to get the index the span
  // is mapped to (the lumpToSpan pointers are rewinded later)
  permInverse.resize(ord);
  for (int64_t i = 0; i < ord; i++) {
    int64_t p = mergeWith[i];
    int64_t lumpIndex1 = unpermutedRootSpanToLump[p == -1 ? i : p];
    permInverse[i] = lumpToSpan[lumpIndex1]++;  // advance
  }
  rewindVec(lumpToSpan);  // restore after advancing
}

void EliminationTree::computeAggregateStruct(bool fillOnlyForElims) {
  int64_t ord = ss.order();
  int64_t numLumps = ord - numMerges;

  if (fillOnlyForElims) {
    SparseStructure tperm =  // lower-half csc
        ss.symmetricPermutation(permInverse, /* lowerHalf = */ false,
                                /* sortIndices = */ false);
    for (int64_t e = 0; e < (int64_t)sparseElimRanges.size() - 1; e++) {
      tperm = tperm.addIndependentEliminationFill(sparseElimRanges[e], sparseElimRanges[e + 1]);
    }
    tperm = tperm.transpose();

    vector<int64_t> tags(ord, -1);  // check if row el was added already
    colStart.push_back(0);
    for (int64_t a = 0; a < numLumps; a++) {
      int64_t aStart = lumpToSpan[a];
      int64_t aEnd = lumpToSpan[a + 1];
      int64_t pStart = tperm.ptrs[aStart];
      int64_t pEnd = tperm.ptrs[aEnd];
      for (int64_t i = pStart; i < pEnd; i++) {
        int64_t p = tperm.inds[i];
        if (tags[p] < (int64_t)a) {
          rowParam.push_back(p);  // L(p,a) is set
          tags[p] = a;
        }
      }
      sort(rowParam.begin() + colStart[colStart.size() - 1], rowParam.end());
      colStart.push_back(rowParam.size());
    }
    return;
  }

  // Supernodal symbolic factorization: the structure of (merged) lump `a` is the
  // union of the diagonal block, the entries of A in the columns of `a`, and the
  // structures of the children lumps (the lumps whose first row below the diagonal
  // block falls in `a`). This is equivalent to computing the full elimination fill
  // and taking the union of the columns of each lump, but it only costs as much as
  // the size of the lump-level structure.
  vector<int64_t> spanToLump(ord);
  for (int64_t a = 0; a < numLumps; a++) {
    for (int64_t sp = lumpToSpan[a]; sp < lumpToSpan[a + 1]; sp++) {
      spanToLump[sp] = a;
    }
  }

  // entries of the permuted A (lower half), grouped by lump of column
  vector<int64_t> lumpAPtr(numLumps + 1, 0);
  for (int64_t i = 0; i < ord; i++) {
    int64_t newI = permInverse[i];
    for (int64_t q = ss.ptrs[i], qEnd = ss.ptrs[i + 1]; q < qEnd; q++) {
      int64_t newJ = permInverse[ss.inds[q]];
      lumpAPtr[spanToLump[std::min(newI, newJ)]]++;
    }
  }
  cumSumVec(lumpAPtr);
  vector<int64_t> lumpARows(lumpAPtr[numLumps]);
  for (int64_t i = 0; i < ord; i++) {
    int64_t newI = permInverse[i];
    for (int64_t q = ss.ptrs[i], qEnd = ss.ptrs[i + 1]; q < qEnd; q++) {
      int64_t newJ = permInverse[ss.inds[q]];
      lumpARows[lumpAPtr[spanToLump[std::min(newI, newJ)]]++] = std::max(newI, newJ);
    }
  }
  rewindVec(lumpAPtr);

  vector<int64_t> firstChild(numLumps, -1), nextSibling(numLumps, -1);
  vector<int64_t> tags(ord, -1);
  colStart.assign(1, 0);
  rowParam.clear();
  rowParam.reserve(lumpARows.size() + ord);
  for (int64_t a = 0; a < numLumps; a++) {
    int64_t aStart = lumpToSpan[a];
    int64_t aEnd = lumpToSpan[a + 1];
    // diagonal block
    for (int64_t sp = aStart; sp < aEnd; sp++) {
      rowParam.push_back(sp);
      tags[sp] = a;
    }
    int64_t belowDiag = rowParam.size();

    // entries of A
    for (int64_t q = lumpAPtr[a], qEnd = lumpAPtr[a + 1]; q < qEnd; q++) {
      int64_t r = lumpARows[q];
      if (tags[r] != a) {
        BASPACHO_CHECK_GE(r, aEnd);
        tags[r] = a;
        rowParam.push_back(r);
      }
    }

    // structures of children
    for (int64_t b = firstChild[a]; b != -1; b = nextSibling[b]) {
      for (int64_t q = colStart[b], qEnd = colStart[b + 1]; q < qEnd; q++) {
        int64_t r = rowParam[q];
        if (r >= aEnd && tags[r] != a) {
          tags[r] = a;
          rowParam.push_back(r);
        }
      }
    }

    sort(rowParam.begin() + belowDiag, rowParam.end());
    colStart.push_back(rowParam.size());

    // register as child of the lump containing the first row below the diagonal
    if ((int64_t)rowParam.size() > belowDiag) {
      int64_t p = spanToLump[rowParam[belowDiag]];
      nextSibling[a] = firstChild[p];
      firstChild[p] = a;
    }
  }
}

vector<int64_t> EliminationTree::computeSpanStart() {
  vector<int64_t> spanStart(paramSize.size() + 1);
  leftPermute(spanStart.begin(), permInverse, paramSize);
  cumSumVec(spanStart);

  return spanStart;
}

}  // end namespace BaSpaCho
