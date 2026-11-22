/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Fits a ComputationModel to the timings of the factor operations collected on this machine
// (InitBench -collect, bench -Z), see doc/analysis.md.

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "baspacho/baspacho/ComputationModelFit.h"

using namespace BaSpaCho;
using namespace std;

static void help() {
  cout << "FitModel [options] samples.txt [more samples files...]\n"
       << "  -time       weight the relative errors by the time (default: plain relative errors)\n"
       << "  -outlier r  reject the samples off by more than a factor r from the fit (default 4)\n"
       << "  -out f      write the parameters to f (use with InitBench -cm file:f)\n";
}

int main(int argc, char* argv[]) {
  bool timeWeighted = false;
  double outlier = 4.0;
  string out;
  vector<string> inputs;
  for (int i = 1; i < argc; i++) {
    string a = argv[i];
    if (a == "-time") {
      timeWeighted = true;
    } else if (a == "-outlier" && i + 1 < argc) {
      outlier = stod(argv[++i]);
    } else if (a == "-out" && i + 1 < argc) {
      out = argv[++i];
    } else if (a == "-h" || a == "--help") {
      help();
      return 0;
    } else if (!a.empty() && a[0] == '-') {
      cerr << "unknown option " << a << endl;
      help();
      return 1;
    } else {
      inputs.push_back(a);
    }
  }
  if (inputs.empty()) {
    help();
    return 1;
  }
  ComputationModelSamples samples;
  for (auto& f : inputs) {
    samples.load(f);
  }
  ComputationModelFitStats stats[4];
  ComputationModel model = fitComputationModel(samples, timeWeighted, outlier, 4, stats);
  const char* names[4] = {"potrf", "trsm", "syge", "asmbl"};
  for (int i = 0; i < 4; i++) {
    printf("%-6s samples %8lld kept %8lld  median relative error %.3f  "
           "estimated/measured total %.3f\n",
           names[i], (long long)stats[i].samples, (long long)stats[i].kept, stats[i].medianRelErr,
           stats[i].totalRatio);
  }
  printf("model (ComputationModel.cpp initializer):\n%s\n", formatComputationModel(model).c_str());
  if (!out.empty()) {
    ofstream o(out);
    o << computationModelParams(model) << "\n";
    printf("parameters written to %s\n", out.c_str());
  }
  return 0;
}
