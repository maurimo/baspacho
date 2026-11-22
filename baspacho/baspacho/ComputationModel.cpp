/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "baspacho/baspacho/ComputationModel.h"

namespace BaSpaCho {

const ComputationModel ComputationModel::model_OpenBlas_i7_1185g7{
    {3.527141723946874224e-07, -5.382557351808083451e-08, 4.677984682984275924e-09,
     7.384424667338682676e-12},
    {1.101115592925888909e-06, 6.936563076265144074e-07, -1.827661167503034051e-09,
     1.959826916788009885e-09, 1.079857543323972179e-09, 2.963338652996178598e-11},
    {6.14190596709488416e-07, -4.489948374364910256e-09, 5.943145978912038475e-10,
     -1.201283634136652872e-08, 1.266858215451465993e-09, 2.624001993284897048e-11},
    {3.069607518266660019e-07, 3.763778311956422235e-08, 1.991443920635728855e-07,
     3.788938150548870089e-09}};

const ComputationModel ComputationModel::model_Cuda117_2080Ti{
    {1.826148710881364949e-05, 4.44453427795126131e-06, 4.824038596816424795e-09,
     2.616891884279722513e-13},
    {2.252999933361577123e-06, -3.090170544094683306e-08, 5.719994050518756252e-10,
     -1.007960791939421421e-09, 1.167012989682758675e-10, -4.025574444690399163e-13},
    {4.55174422740625352e-06, 8.155647192215145375e-11, -1.76305533240800307e-14,
     1.054095249414374123e-08, -2.491863357081539836e-12, -1.095648600879501635e-16},
    {1.975089750288875748e-06, -1.339369810950508464e-10, -3.758728373628488434e-10,
     1.745285595679570848e-13}};

// The models above were fitted on measured operations of limited size, and with unconstrained
// least squares: outside of that range they can predict negative times, or times decreasing with
// a size (model_OpenBlas_i7_1185g7 for thin syrk/gemm, eg. m = n = 3), or impossible flop rates
// (model_Cuda117_2080Ti: 31 TFlop/s for a 512^3 gemm, ie. large operations almost free). The
// optimization of the merges (Settings::analysis) exploits such artifacts, so the defaults are:

// model_OpenBlas_i7_1185g7 refitted with non-negative coefficients on its own estimates for the
// operation shapes of the factorization of a suite of problems (within 1-2% of the original for
// trsm/syge, 10% for potrf, on those shapes)
const ComputationModel ComputationModel::model_OpenBlas_i7_1185g7_nonneg{
    {6.384388371331e-08, 0.000000000000e+00, 3.374979883305e-09, 9.640686449177e-12},
    {1.298557946746e-06, 6.345303896130e-07, 0.000000000000e+00, 2.147148800184e-09,
     1.120748012072e-09, 2.709246835056e-11},
    {3.514174468363e-07, 9.535069066619e-10, 5.405521756144e-10, 0.000000000000e+00,
     9.951331270303e-10, 2.932205754931e-11},
    {3.069607518267e-07, 3.763778311956e-08, 1.991443920636e-07, 3.788938150549e-09}};

// model_Cuda117_2080Ti with the negative coefficients set to zero, and the flop terms bounded by
// the FP64 rate of a 2080Ti (0.4 TFlop/s for gemm, 0.3 for trsm and potrf: an assumption, not
// a fit)
const ComputationModel ComputationModel::model_Cuda117_2080Ti_bounded{
    {1.826148710881364949e-05, 4.44453427795126131e-06, 4.824038596816424795e-09, 1.1e-12},
    {2.252999933361577123e-06, 0, 5.719994050518756252e-10, 0, 1.167012989682758675e-10, 3.3e-12},
    {4.55174422740625352e-06, 8.155647192215145375e-11, 0, 1.054095249414374123e-08, 0, 5e-12},
    {1.975089750288875748e-06, 0, 0, 1.745285595679570848e-13}};

// fitted by MetalComputationModelTest (non-negative least squares) on a GitHub macos-15 runner
// (paravirtualized Apple GPU), without the fixed cost of a command buffer (~0.5ms there, shared by
// all the ops normally). Median relative errors 0.11-0.38, operations sampled up to syrk/gemm
// n = 1773, k = 618
const ComputationModel ComputationModel::model_Metal_CI_paravirtual{
    {1.868697958226902002e-05, 0, 3.989509855614285503e-09, 6.369721816492629040e-12},
    {1.653728738907606942e-05, 0, 1.168337037601280787e-08, 0, 1.399730557338805025e-10,
     5.920569735506544175e-11},
    {1.869411855161148446e-05, 0, 0, 1.795640974254546192e-09, 0, 3.885100081301434965e-13},
    {1.973003199212342371e-05, 0, 0, 2.847089426771100564e-10}};

}  // namespace BaSpaCho
