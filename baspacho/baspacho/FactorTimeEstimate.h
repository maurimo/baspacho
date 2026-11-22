/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <vector>
#include "baspacho/baspacho/CoalescedBlockMatrix.h"
#include "baspacho/baspacho/ComputationModel.h"

namespace BaSpaCho {

// Time of the factorization estimated by a computation model: the factorization loops of the
// Solver are run on the skeleton only (no numeric data), adding the estimate of each operation.
struct FactorTimeEstimate {
  double supernodal = 0;  // dense operations: potrf, trsm, syrk/gemm, assemble (the model's ops)
  double sparse = 0;      // sparse elimination ranges (not modelled as such, see below)
  double opTime[4] = {0, 0, 0, 0};    // supernodal part by operation: potrf, trsm, syge, asmbl
  int64_t opCount[4] = {0, 0, 0, 0};  // and number of calls
};

// The sparse elimination (batched on the GPU backends) is estimated, per lump, as its potrf,
// the trsm and syrk of all the rows below, and the assembly of its blocks.
FactorTimeEstimate estimateFactorTime(const CoalescedBlockMatrixSkel& skel,
                                      const std::vector<int64_t>& sparseElimRanges,
                                      const ComputationModel& model);

}  // end namespace BaSpaCho
