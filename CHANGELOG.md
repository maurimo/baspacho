Unreleased (maurimo/baspacho, November 2026)

### Metal backend (Apple Silicon GPUs, float)
- New `BackendMetal`, based on facebookresearch/baspacho#6 by Rob Taylor: sparse elimination, assembly,
  dense Cholesky, triangular solves and products (Metal Performance Shaders for large ones) on the GPU,
  data in `MetalMirror` buffers; the shader library is embedded in the library
- Solves by levels of the elimination tree (3d grid solve ~1.7x faster than lump by lump on the CI runners)
- Thread-safe across independent solvers; GPU errors thrown from the `Solver` calls
- `BackendAuto`/`detectBestBackend()` check at runtime that a CUDA/Metal device is available

### Performance
- `createSolver` (symbolic analysis) 1.5-3x faster with an identical factor structure; `InitBench` tool

### Fixes and behavior changes
- A block that is not positive definite gives NaN in the factor on every backend (CPU backends left
  finite values)
- `Settings::numThreads` is honored by the analysis (no thread pool with 1 thread)
- Statistics are fully opt-in (`enableStats`), `resetStats` clears all of them
- CUDA: errors are thrown instead of `abort()`; empty batches are no-ops; no leaks on failed initialization
- Sparse elimination ranges are validated; `extractRightBottom` fixed on non lower-triangular input
- CMake: bundling works with GNU ar, versioned llvm-ar and Apple libtool and tracks its inputs; re-configuring
  an existing build keeps the gtest targets

### CI
- GitHub Actions: CPU tests on Linux and macOS, CUDA compile checks, Metal tests on the macOS runners

1.0 (August 07, 2022)

### BaSpaCho initial release
