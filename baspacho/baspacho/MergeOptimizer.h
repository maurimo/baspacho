/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <vector>
#include "baspacho/baspacho/ComputationModel.h"

namespace BaSpaCho {

// (internal) Chooses the supernodes (merges of tree nodes into their parent) minimizing the factor
// time estimated by a ComputationModel, with the exact cost of the factor operations as issued by
// Solver::factor for the resulting skeleton. A column lump S of size n, whose rows below the
// diagonal are grouped by target lump T, costs:
//     potrf(n) + trsm(n, R) + sum_T [ syge(m_T, M_T, n) + asmbl(br_T, bc_T) ]
// where R are the rows below the diagonal, m_T (bc_T) the rows (blocks) in T, M_T (br_T) the rows
// (blocks) from T to the end of the column. The structure of a lump is the structure of its top
// node, and merging a node k into its parent's lump p only changes:
//   * the column costs of k and p (k's disappears, p's grows to size n_k + n_p),
//   * the columns having rows in both k and p, where the two boards become one.
// The input nodes can be single spans or supernodes found beforehand, nodes are in topological
// order (parent[a] > a).
struct MergeOptimizer {
  // a board: column `c` (or, in the input, the row node) with `bc` blocks and `m` rows in a row
  // node, and `brDown` blocks and `rowsDown` rows after it in the column
  struct Board {  // (32 bits: the span-level structure of large problems is big)
    int32_t c;
    int32_t bc;
    int32_t m;
    int32_t brDown;
    int32_t rowsDown;
  };

  // `colBoards[a]`: boards of column a, by row node (`Board::c` = row node), in order
  MergeOptimizer(const ComputationModel& model, std::vector<int64_t> parent,
                 std::vector<int64_t> size, std::vector<int64_t> rows, std::vector<bool> canMerge,
                 std::vector<bool> sparse, const std::vector<std::vector<Board>>& colBoards);

  // current lump of node a
  int64_t root(int64_t a);

  // change of the total time if (alive) node k is merged into its parent's lump p
  double mergeDelta(int64_t k, int64_t p);

  void applyMerge(int64_t k, int64_t p);

  // steepest descent: always applies the merge with the best (most negative) time change, the
  // values being re-evaluated when popped, after the lump or the parent's lump changed, and all
  // of them before stopping
  // (with `noise` > 0 the merges are ranked by delta * (1 + noise * u), u uniform in [-1, 1],
  // only improving merges are applied)
  void steepest(double noise = 0, uint64_t seed = 0);

  double colCost(int64_t c, double n) const;

  // assembly of a board: br/bc blocks, m rows (mFull from the board to the end of the column)
  double asmbl(double br, double bc, double m, double mFull) const {
    return model.asmblEst(br, bc) + model.asmblElemEst(m, mFull);
  }

  const ComputationModel& model;
  int64_t numNodes;
  std::vector<int64_t> parent, size, rows;
  std::vector<bool> canMerge, sparse;
  std::vector<int64_t> mergeWith;
  std::vector<std::vector<Board>> perRow;  // boards in row node r, by column (alive ones valid)
  std::vector<Eigen::Vector2d> colSyge;    // syge costs of a column, linear in its size
  std::vector<double> colAsmbl;
  std::vector<Board> tmp;
  double total = 0;  // estimated time of the current partition
  int64_t numMerges = 0;
};

}  // end namespace BaSpaCho
