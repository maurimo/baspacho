/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <Eigen/Geometry>
#include <string>
#include "baspacho/baspacho/SparseStructure.h"

namespace BaSpaCho {

struct ComputationModel;

// (internal) class encapsulating the computation of an elimination-tree
// and the heuristics applied on top of the tree structure. In patricular
// it's responsible of
// 1. automated computation of `sparse elimination ranges`, which is triggered
//   when there are large sets of leaf nodes that can be simulatenously eliminated
//   in parallel. Such sets of leafs are progressively removed from the tree under
//   analysis and related ranges are added to `sparseElimRanges`.
// 2. on the remaining part of the tree, merge child nodes with parent, according
//   to some heuristics depending on the zero-fill of the resulting nodes and the
//   expected runtimes of the resulting dense operations
struct EliminationTree {
  EliminationTree(const std::vector<int64_t>& paramSize, const SparseStructure& ss,
                  const ComputationModel* compMod = nullptr);

  // build the tree for processing
  void buildTree();

  // as buildTree, but only computes the elimination tree and the row counts of the columns of L
  // (in O(nnz(A) * alpha), Gilbert-Ng-Peyton), without the structure of L. The merges are then
  // decided by relaxed amalgamation (see computeMergesFast)
  void buildTreeFast();

  // executed the processing
  void processTree(bool detectSparseElimRanges, const std::vector<int64_t>& noCrossPoints = {},
                   bool findOnlyElims = false);

  // computes the aggregate struct (where nodes have been merged)
  void computeAggregateStruct(bool fillOnlyForElims = false);

  // utility helper to get span starts from interally stored (sorted) param sizes
  std::vector<int64_t> computeSpanStart();

  // internal (called by processTree)
  void computeNodeHeights(const std::vector<int64_t>& noCrossPoints);

  // internal (called by processTree)
  void computeSparseElimRanges(const std::vector<int64_t>& noCrossPoints);

  // internal (called by processTree)
  void computeMerges();

  // internal (called by processTree, after buildTreeFast)
  void computeMergesFast();

  bool fastSymbolic = false;  // set by buildTreeFast
  bool fundamentalOnly = false;      // computeMergesFast: only merges adding no zeros...
  double fundamentalMaxSize = 1e30;  // ... and only up to this size (scalar columns)

  // how the merges are chosen by processTree. Auto: Optimized after buildTreeFast, Greedy after
  // buildTree. Relaxed: relaxed amalgamation (computeMergesFast). Greedy: computeMerges, greedy
  // on an estimate of the merge's effect on the two nodes. Optimized/Deep: computeMergesOptimized
  // GreedyCorrected (for comparison): computeMerges' greedy procedure with the exact cost
  enum MergeMode {
    MergeAuto,
    MergeRelaxed,
    MergeGreedy,
    MergeOptimized,
    MergeDeep,
    MergeGreedyCorrected
  };
  MergeMode mergeMode = MergeAuto;
  int deepIterations = 0;     // MergeDeep: rounds of iterated local search
  double optimizedCost = -1;  // set by computeMergesOptimized: estimated time of the factor

  // merges minimizing the factor time estimated by the computation model with the exact cost of
  // the operations (MergeOptimizer, steepest descent): from the (capped) fundamental and the
  // relaxed supernodes, or (deep) on single spans from more starting points (including the result
  // of computeMerges after buildTree) followed by `deepIterations` rounds of local search
  void computeMergesOptimized(MergeMode mode);

  // span-level partition of computeMergesFast (fundamental: only fill-free merges, up to a size)
  std::vector<int64_t> partitionFast(bool fundamental, double fundamentalCap);

  // internal (called by processTree)
  void collapseMergePointers();

  // internal: the node `p` was merged into (following mergeWith, with path compression)
  int64_t mergedRoot(int64_t p);

  // init data
  std::vector<int64_t> paramSize;
  const SparseStructure& ss;  // input
  const ComputationModel& compMod;

  // generated data, buildTree
  std::vector<int64_t> parent;
  std::vector<int64_t> nodeSize;
  std::vector<int64_t> nodeRows;
  std::vector<int64_t> nodeRowBlocks;
  struct NodeStats {
    int64_t colIdx;
    int64_t rBlocks;
    int64_t rows;
    int64_t rBlocksDown;
    int64_t rowsDown;
    NodeStats(int64_t c, int64_t rbs, int64_t rs, int64_t rbd, int64_t rd)
        : colIdx(c), rBlocks(rbs), rows(rs), rBlocksDown(rbd), rowsDown(rd) {}
  };
  std::vector<std::vector<NodeStats>> perRowNodeStats;
  std::vector<NodeStats> tmpRowStats;

  // modelled costs of syge/asmbl ops for a node, as linear
  // function of node size (resp. size as number of col blocks)
  std::vector<Eigen::Vector2d> sygeCosts;
  std::vector<Eigen::Vector2d> asmblCosts;

  // generated data, processTree
  std::vector<int64_t> sparseElimRanges;
  std::vector<std::tuple<int64_t, int64_t, int64_t>> unmergedHeightNode;
  std::vector<bool> forbidMerge;
  std::vector<bool> isSparseElim;
  std::vector<int64_t> numMergedNodes;
  std::vector<int64_t> mergeWith;
  int64_t numMerges;

  // generated computing aggregate struct
  std::vector<int64_t> permInverse;
  std::vector<int64_t> lumpStart;
  std::vector<int64_t> lumpToSpan;
  std::vector<int64_t> colStart;
  std::vector<int64_t> rowParam;
};

}  // end namespace BaSpaCho
