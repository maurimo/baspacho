# Symbolic analysis and computation models

`createSolver` runs the symbolic analysis of a sparse block matrix: it orders the parameters
(spans), finds the ranges that can be eliminated with sparse operations, and groups the spans into
supernodes (lumps). The way spans are merged into lumps decides which dense operations the
factorization runs. BaSpaCho chooses the merges using a **computation model**, an estimate of the
time of each operation on the target machine. This document describes:

* the cost of a factorization;
* the model and how to fit it to a machine;
* the analyses (`Settings::analysis`) and how they use the model.

## 1. What the factorization costs

`Solver::factor` first runs the sparse elimination ranges, then processes the remaining lumps in
order. For each lump `L` with `n` columns, it eliminates each *board* of `L` into its target lump:

* `potrf(n)`, the Cholesky factorization of the diagonal block;
* `trsm(n, R)`, where `R` is the number of scalar rows below the diagonal (skipped when `R = 0`);
* for each *board*, one `syge(m_T, M_T, n)` (syrk/gemm into a temporary buffer) followed by one
  `asmbl(br_T, bc_T)` (adding the buffer into the factor). A board is the group of rows of `L`
  that belong to one target lump `T` below the diagonal. In these arguments:
  * `m_T` is the number of rows of `L` in `T`, and `M_T` the number of rows from `T` to the end
    of the column;
  * `bc_T` is the number of blocks (chains, one per span) in `T`, and `br_T` the number of blocks
    from `T` to the end.

So merging two nodes changes more than the two columns involved. It also changes every column
that has rows in both, because their two boards become one: one `syge` and one `asmbl` fewer,
each slightly larger. When the fixed cost per operation is high, this is where most of the time
goes.

`estimateFactorTime(skel, sparseElimRanges, model)` and `Solver::estimatedFactorTime(model)` run
that same loop on the skeleton with a backend that only adds up `model`'s estimates, without
allocating the factor. They return the supernodal total, a per-operation breakdown, and a rough
estimate of the sparse elimination part, which the model does not describe.

## 2. The computation model

`ComputationModel` estimates each operation with a polynomial that is linear in its parameters:

| operation | estimate |
|---|---|
| `potrf(n)` | `a + b n + c n² + d n³` |
| `trsm(n, k)` | `a + b n + c n² + k (d + e n + f n²)` |
| `syge(m, n, k)` | `a + b (m+n) + c mn + k (d + e (m+n) + f mn)` (symmetric in m, n) |
| `asmbl(br, bc)` | `a + b br + c bc + d br bc`, plus `asmblElemParam · m · M` |

The per-element term counts the scalars that the assembly moves: the `m × M` rectangle of the
board. Without it, large boards with few blocks look free.

**The coefficients must be non-negative.** The analysis does not just evaluate the model on
typical operations, it *optimizes* against it. Any region where the model predicts negative
times, times that decrease with a size, or impossible flop rates gets exploited. For example, the
original `model_Cuda117_2080Ti` makes a 512³ gemm run at 31 TFlop/s, and the analysis
consequently builds huge supernodes. Polynomials fitted with unconstrained least squares on a
limited range of sizes have such regions.

Pre-built models (`ComputationModel.cpp`):

* `model_OpenBlas_i7_1185g7_nonneg`, the CPU default: the original i7 model refitted with
  non-negative coefficients;
* `model_Cuda117_2080Ti_bounded`, the CUDA default: the original CUDA model with its negative
  coefficients zeroed and the flop terms set to assumed FP64 rates. This is an assumption, not a
  fit, so refit it on real hardware;
* `model_Metal_CI_paravirtual`, the Metal default: fitted by `MetalComputationModelTest` on the
  paravirtualized GPU of the GitHub macOS runners;
* `model_OpenBlas_i7_1185g7` and `model_Cuda117_2080Ti`, the original fits, kept for reference.

The model is selected with `Settings::computationModel`; when unset, the backend's default is used.

## 3. Fitting a model to a machine

Timings vary a lot between machines (cache sizes, memory bandwidth, BLAS library, threads), so a
model is only valid on the machine it was fitted on.

1. **Collect** the timings of the operations of real factorizations, using single-threaded BLAS
   if that is how you will run:
   ```
   OPENBLAS_NUM_THREADS=1 build/baspacho/benchmarking/InitBench -mtx problem.mtx -t 1 -reps 2 \
       -collect samples.txt
   ```
   Repeat on problems representative of your workload, and with each analysis (`-analysis
   fast|exact|deep`), so that the samples cover the operation shapes the analysis will produce.
   `bench -Z` also writes samples (`stats_<config>.txt`). Each line of the file is one operation:
   `potrf n t`, `trsm n k t`, `syge m n k t` or `asmbl br bc m M t`. The `m` and `M` of an
   `asmbl` line come from the `syge` of the same board. On Metal, the operations are timed by
   their GPU execution time.
2. **Fit**:
   ```
   build/baspacho/benchmarking/FitModel samples.txt [more.txt ...] -out model.txt
   ```
   This calls `fitComputationModel`:
   * non-negative least squares (Lawson–Hanson) on the relative errors
     `Σ ((est − t) / t)²`, or with `-time` on `Σ (est − t)² / t`, which weights each sample by
     its time, since the factorization time is a sum;
   * the columns are scaled for conditioning;
   * samples off by more than a factor 4 from the fit, for instance because the process was
     descheduled, are rejected, and the fit is repeated.

   It prints, for each operation:
   * the median relative error;
   * the ratio of estimated to measured total time;
   * the model as a C++ initializer.

   The `-out` file can be loaded by `parseComputationModelParams` and InitBench (`-cm file:model.txt`).
3. **Validate**. `InitBench -cm file:model.txt` prints the model's estimate of the factorization
   next to the measured time:
   ```
   InitBench -mtx problem.mtx -t 1 -analysis fast -cm file:model.txt
   ```
   It is worth iterating: refit with samples collected from the partitions the analysis produced
   with the new model, then check that the estimate/measured ratio is similar across analyses.

`MetalComputationModelTest`, which runs in the Metal CI, does this on the CI runners' GPU. It
subtracts the fixed cost of a command buffer, measured on trivial operations, because in a normal
factorization the operations share command buffers. It prints the fitted parameters as CI
annotations.

## 4. The analyses

All the analyses start from the same elimination tree of the (AMD-ordered) parameters. Its
column counts give the number of rows, in blocks and in scalars, of each column of the factor.
They are computed in `O(nnz(A) α)` (Gilbert–Ng–Peyton), without building the factor's structure.

### Relaxed (`AnalysisRelaxed`)
Relaxed amalgamation in the style of CHOLMOD, with larger sizes. A child is merged into its parent
if:
* no explicit zeros are added;
* or the merged node has at most 8 columns;
* or it has at most 32 columns and fewer than 80% zeros;
* or at most 96 columns and fewer than 10% zeros;
* or fewer than 5% zeros.

It does not use the model, and it is the fastest analysis.

### Fast (`AnalysisFast`, `AnalysisAuto`)
The merges minimize the factorization time estimated by the model, using the exact cost of
section 1 (`MergeOptimizer`):

* **Nodes.** The optimization works on small supernodes: the intersection of the fundamental
  supernodes (fill-free merges), capped at 32 columns, and the relaxed supernodes. The cap
  matters: in a large dense supernode, `potrf` is less efficient than the `trsm`/`syge` of a
  split one, so supernodes must remain splittable.
* **Structure.** The structure of each node is computed from the entries of A and the structures
  of its children. Its rows are grouped into boards by target node, along the path to the root.
* **Cost.** The cost of a partition is the sum, over the lumps, of the estimates of their
  operations. Merging node `k` into its parent's lump `p` changes only:
  * the column costs of `k` and `p`: `k`'s column disappears, and `p`'s grows to `n_k + n_p`
    columns;
  * the boards, in every column with rows in both `k` and `p`, which become one board.

  So the change in cost of each candidate merge is computed exactly and cheaply, from the per-row
  lists of boards.
* **Search.** Steepest descent always applies the merge with the largest decrease. Candidates are
  re-evaluated lazily, and all of them are refreshed before stopping. The descent starts twice,
  from the fundamental and from the relaxed supernodes, and the better result is kept.

The descent only merges, and merges show synergy: a chain of merges can pay off when no single
step of it does. That is why the starting points matter.

### Deep (`AnalysisDeep`)
This is the same optimization, but on single spans and from more starting points:
* the merges of Exact;
* the fundamental supernodes, both uncapped and capped;
* the relaxed supernodes;
* no merges.

Because one starting point is Exact's result, Deep is never worse than Exact according to the
model. `Settings::deepSearchIterations` adds rounds of iterated local search: dissolve 20% of the
lumps at random, re-merge with a randomized descent, and keep the result if it is better. Deep
needs the structure of the factor at the level of single spans, so it is several times slower
than Exact and uses more memory.

### Exact (`AnalysisExact`)
The previous analysis. It computes the structure of the factor and merges greedily, in the order
of increasing fill, when the merge decreases its own estimate. That estimate is not the cost of
section 1:
* it charges each lump a fake diagonal board;
* it uses the column's size where the assembly cost needs the target's blocks;
* the merge decision ignores the boards merged in the other columns.

Its estimate of the factorization time is 1.1–1.7x the actual cost of its own partitions
(geometric mean over 38 problems, depending on the model). Using the correct cost in its greedy
procedure is no better: it then refuses the steps of the merge chains and under-merges.

## 5. How well this works

The figures below are for a suite of 39–40 problems: BAL bundle adjustment, SE-Sync pose graphs,
SuiteSparse FEM matrices, and synthetic 2D/3D grids, kNN meshes, SLAM, power-law, banded and
random graphs.

According to the models, relative to Exact (geometric means):

| model | Fast, modelled factor time | Fast, analysis time | Deep, modelled factor time | Deep, analysis time |
|---|---|---|---|---|
| CPU default | 0.87 | 0.8 | 0.87 | 4 |
| CPU fitted on the test machine | 0.96 | 0.8 | 0.96 | 4 |
| CUDA bounded | 0.97 | | 0.97 | |
| Metal CI fit | 0.88 | | 0.88 | |

Fast and Deep find the same partitions within 0.6%.

**Measured on a CPU** (a 4-core VPS, single thread), with a model fitted on that machine, the real
factorization times of Fast, Deep and Exact are the same within noise (±5%).

* The model's error on a whole partition is ±10–25%, and it depends on the shapes of the
  operations the partition uses. For example, thin syrk/gemm and large assemblies are
  underestimated in some regimes.
* That error is larger than the gains, and the optimizer selects exactly the shapes where the
  model is optimistic.

Model-driven merges therefore pay off only as far as the model is accurate on the machine.
Collect samples from the partitions the analysis produces, refit, and check the
estimate/measured ratio as in section 3.
