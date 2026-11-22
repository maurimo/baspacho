/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <string>
#include <vector>
#include "baspacho/baspacho/ComputationModel.h"

namespace BaSpaCho {

class Solver;

// Measured timings of the factor operations, to fit a ComputationModel (see doc/analysis.md)
struct ComputationModelSamples {
  std::vector<std::array<double, 2>> potrf;  // n, time
  std::vector<std::array<double, 3>> trsm;   // n, k, time
  std::vector<std::array<double, 4>> syge;   // m, n, k, time
  // br, bc, m, mFull, time: m and mFull from the syrk/gemm of the same board (just before it)
  std::vector<std::array<double, 5>> asmbl;

  // Records the operations of the factorizations of `solver` (its stats must be enabled, see
  // Solver::enableStats; this object must outlive the factorizations, the callbacks refer to it)
  void collectFrom(Solver& solver);

  // text file, one operation per line ("potrf n t", "trsm n k t", "syge m n k t",
  // "asmbl br bc m mFull t"); load appends
  void save(const std::string& path, bool append = true) const;
  void load(const std::string& path);

  // (last syrk/gemm seen by collectFrom)
  double lastSygeM = -1, lastSygeN = -1;
};

struct ComputationModelFitStats {
  int64_t samples = 0;    // of the operation
  int64_t kept = 0;       // not rejected as outliers
  double medianRelErr = 0;  // on the kept samples
  double totalRatio = 0;  // sum of the estimates / sum of the timings (kept samples)
};

// Fits the parameters of each operation's model (including the per-element `asmbl` term) by
// non-negative least squares on the relative errors, sum_i ((est_i - t_i) / t_i)^2, or with
// `timeWeighted` on sum_i (est_i - t_i)^2 / t_i (relative errors weighted by the time, as the
// factor time is a sum). Non-negative coefficients keep every estimate positive and increasing
// in each size, also outside the range of the samples, which the optimization of the merges
// would otherwise exploit. The samples off by more than `outlierRatio` from the fit (eg. the
// process descheduled) are rejected, and the fit repeated (`rounds` times).
// `stats`, if not null: potrf, trsm, syge, asmbl.
ComputationModel fitComputationModel(const ComputationModelSamples& samples,
                                     bool timeWeighted = false, double outlierRatio = 4.0,
                                     int rounds = 4, ComputationModelFitStats* stats = nullptr);

// the model as a C++ initializer (as in ComputationModel.cpp), and as the plain list of its 21
// parameters (potrf 4, trsm 6, syge 6, asmbl 4, asmbl per element 1)
std::string formatComputationModel(const ComputationModel& model);
std::string computationModelParams(const ComputationModel& model);
ComputationModel parseComputationModelParams(const std::string& params);

}  // end namespace BaSpaCho
