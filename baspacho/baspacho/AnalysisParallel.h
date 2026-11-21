/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <dispenso/parallel_for.h>
#include <algorithm>
#include <cstdint>

namespace BaSpaCho {

// Parallelism of the symbolic analysis (createSolver and the structure computations it runs):
// at most `analysisThreads()` threads. createSolver sets it to Settings::numThreads for its
// duration (AnalysisThreadsScope), outside of it the analysis is serial. With 1 thread the
// analysis never touches dispenso's global thread pool, which would start one thread per core.
inline thread_local int64_t analysisThreadsValue = 1;

inline int64_t analysisThreads() { return analysisThreadsValue; }

struct AnalysisThreadsScope {
  explicit AnalysisThreadsScope(int64_t numThreads) : prev(analysisThreadsValue) {
    analysisThreadsValue = std::max<int64_t>(numThreads, 1);
  }
  ~AnalysisThreadsScope() { analysisThreadsValue = prev; }
  AnalysisThreadsScope(const AnalysisThreadsScope&) = delete;
  AnalysisThreadsScope& operator=(const AnalysisThreadsScope&) = delete;
  int64_t prev;
};

// Calls f(chunkBegin, chunkEnd) on chunks of [begin, end) of up to `chunk` items, in parallel if
// allowed by analysisThreads() and if there are at least 2 chunks, else serially in order
template <typename F>
void analysisParallelFor(int64_t begin, int64_t end, int64_t chunk, F&& f) {
  int64_t numChunks = (end - begin + chunk - 1) / chunk;
  int64_t numThreads = analysisThreads();
  if (numThreads <= 1 || numChunks <= 1) {
    for (int64_t b = begin; b < end; b += chunk) {
      f(b, std::min(end, b + chunk));
    }
    return;
  }
  dispenso::ParForOptions options;
  options.maxThreads = (uint32_t)std::min<int64_t>(numThreads, 1 << 30);
  dispenso::parallel_for(dispenso::makeChunkedRange(begin, end, chunk), std::forward<F>(f),
                         options);
}

}  // end namespace BaSpaCho
