/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "baspacho/baspacho/ComputationModelFit.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "baspacho/baspacho/Solver.h"

namespace BaSpaCho {

using namespace std;

void ComputationModelSamples::collectFrom(Solver& solver) {
  auto& sym = solver.internalSymbolicContext();
  sym.potrfStat.callBack = [this](double t, int, int n) { potrf.push_back({(double)n, t}); };
  sym.trsmStat.callBack = [this](double t, int, int n, int k) {
    trsm.push_back({(double)n, (double)k, t});
  };
  sym.sygeStat.callBack = [this](double t, int, int m, int n, int k) {
    syge.push_back({(double)m, (double)n, (double)k, t});
    lastSygeM = m;
    lastSygeN = n;
  };
  // in Solver::factor every assemble follows the syrk/gemm of the same board
  sym.asmblStat.callBack = [this](double t, int, int br, int bc) {
    if (lastSygeM >= 0) {
      asmbl.push_back({(double)br, (double)bc, lastSygeM, lastSygeN, t});
    }
    lastSygeM = lastSygeN = -1;
  };
}

void ComputationModelSamples::save(const string& path, bool append) const {
  FILE* f = fopen(path.c_str(), append ? "a" : "w");
  if (!f) {
    throw runtime_error("cannot write " + path);
  }
  for (auto& s : potrf) fprintf(f, "potrf %g %.9g\n", s[0], s[1]);
  for (auto& s : trsm) fprintf(f, "trsm %g %g %.9g\n", s[0], s[1], s[2]);
  for (auto& s : syge) fprintf(f, "syge %g %g %g %.9g\n", s[0], s[1], s[2], s[3]);
  for (auto& s : asmbl) fprintf(f, "asmbl %g %g %g %g %.9g\n", s[0], s[1], s[2], s[3], s[4]);
  fclose(f);
}

void ComputationModelSamples::load(const string& path) {
  ifstream in(path);
  if (!in) {
    throw runtime_error("cannot read " + path);
  }
  string line, op;
  while (getline(in, line)) {
    istringstream ls(line);
    if (!(ls >> op)) continue;
    vector<double> v;
    double x;
    while (ls >> x) v.push_back(x);
    if (op == "potrf" && v.size() == 2) {
      potrf.push_back({v[0], v[1]});
    } else if (op == "trsm" && v.size() == 3) {
      trsm.push_back({v[0], v[1], v[2]});
    } else if (op == "syge" && v.size() == 4) {
      syge.push_back({v[0], v[1], v[2], v[3]});
    } else if (op == "asmbl" && v.size() == 5) {
      asmbl.push_back({v[0], v[1], v[2], v[3], v[4]});
    } else {
      throw runtime_error("unexpected line in " + path + ": " + line);
    }
  }
}

namespace {

// non-negative least squares (Lawson-Hanson active set): min |A x - b|, x >= 0
Eigen::VectorXd nnls(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
  int64_t n = A.cols();
  Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
  vector<bool> passive(n, false);
  auto lsOnPassive = [&]() {
    vector<int64_t> idx;
    for (int64_t j = 0; j < n; j++) {
      if (passive[j]) idx.push_back(j);
    }
    Eigen::MatrixXd Ap(A.rows(), idx.size());
    for (size_t q = 0; q < idx.size(); q++) Ap.col(q) = A.col(idx[q]);
    Eigen::VectorXd zp = Ap.colPivHouseholderQr().solve(b);
    Eigen::VectorXd z = Eigen::VectorXd::Zero(n);
    for (size_t q = 0; q < idx.size(); q++) z[idx[q]] = zp[q];
    return z;
  };
  for (int iter = 0; iter < 3 * n; iter++) {
    Eigen::VectorXd w = A.transpose() * (b - A * x);
    int64_t best = -1;
    for (int64_t j = 0; j < n; j++) {
      if (!passive[j] && w[j] > 1e-12 * b.norm() && (best == -1 || w[j] > w[best])) best = j;
    }
    if (best == -1) break;
    passive[best] = true;
    while (true) {
      Eigen::VectorXd z = lsOnPassive();
      bool allPos = true;
      double alpha = 1.0;
      for (int64_t j = 0; j < n; j++) {
        if (passive[j] && z[j] <= 0) {
          allPos = false;
          alpha = min(alpha, x[j] / (x[j] - z[j]));
        }
      }
      if (allPos) {
        x = z;
        break;
      }
      x += alpha * (z - x);
      for (int64_t j = 0; j < n; j++) {
        if (passive[j] && x[j] <= 1e-15) {
          passive[j] = false;
          x[j] = 0;
        }
      }
    }
  }
  return x;
}

// fit of one operation: rows of `features` (the model is linear in its parameters) and times
Eigen::VectorXd fitOp(const Eigen::MatrixXd& features, const Eigen::VectorXd& times,
                      bool timeWeighted, double outlierRatio, int rounds,
                      ComputationModelFitStats* stats) {
  int64_t n = features.cols();
  vector<int64_t> keep;
  for (int64_t i = 0; i < times.size(); i++) {
    if (times[i] > 0) keep.push_back(i);
  }
  int64_t numPositive = keep.size();
  Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
  for (int round = 0; round < max(rounds, 1) && !keep.empty(); round++) {
    Eigen::MatrixXd A(keep.size(), n);
    Eigen::VectorXd b(keep.size());
    for (size_t q = 0; q < keep.size(); q++) {
      double t = times[keep[q]];
      double w = timeWeighted ? sqrt(t) : 1.0;
      A.row(q) = features.row(keep[q]) * (w / t);
      b[q] = w;
    }
    Eigen::VectorXd scale = A.colwise().norm().transpose().cwiseMax(1e-300);
    for (int64_t j = 0; j < n; j++) A.col(j) /= scale[j];
    x = nnls(A, b).cwiseQuotient(scale);
    if (round + 1 < rounds) {  // reject the outliers, from all the samples
      keep.clear();
      for (int64_t i = 0; i < times.size(); i++) {
        double t = times[i], est = features.row(i).dot(x);
        if (t > 0 && est > 0 && t < est * outlierRatio && t * outlierRatio > est) keep.push_back(i);
      }
    }
  }
  if (stats) {
    vector<double> errs;
    double sumEst = 0, sumT = 0;
    for (int64_t i : keep) {
      double t = times[i], est = features.row(i).dot(x);
      errs.push_back(fabs(est - t) / t);
      sumEst += est;
      sumT += t;
    }
    sort(errs.begin(), errs.end());
    stats->samples = numPositive;
    stats->kept = keep.size();
    stats->medianRelErr = errs.empty() ? 0 : errs[errs.size() / 2];
    stats->totalRatio = sumT > 0 ? sumEst / sumT : 0;
  }
  return x;
}

}  // namespace

ComputationModel fitComputationModel(const ComputationModelSamples& s, bool timeWeighted,
                                     double outlierRatio, int rounds,
                                     ComputationModelFitStats* stats) {
  auto fit = [&](const Eigen::MatrixXd& f, const Eigen::VectorXd& t, int op) {
    return fitOp(f, t, timeWeighted, outlierRatio, rounds, stats ? stats + op : nullptr);
  };
  Eigen::MatrixXd f;
  Eigen::VectorXd t;

  f.resize(s.potrf.size(), 4);
  t.resize(s.potrf.size());
  for (size_t i = 0; i < s.potrf.size(); i++) {
    f.row(i) = ComputationModel::dPotrfModel(s.potrf[i][0]);
    t[i] = s.potrf[i][1];
  }
  Eigen::Vector<double, 4> potrfParams = fit(f, t, 0);

  f.resize(s.trsm.size(), 6);
  t.resize(s.trsm.size());
  for (size_t i = 0; i < s.trsm.size(); i++) {
    f.row(i) = ComputationModel::dTrsmModel(s.trsm[i][0], s.trsm[i][1]);
    t[i] = s.trsm[i][2];
  }
  Eigen::Vector<double, 6> trsmParams = fit(f, t, 1);

  f.resize(s.syge.size(), 6);
  t.resize(s.syge.size());
  for (size_t i = 0; i < s.syge.size(); i++) {
    f.row(i) = ComputationModel::dSygeModel(s.syge[i][0], s.syge[i][1], s.syge[i][2]);
    t[i] = s.syge[i][3];
  }
  Eigen::Vector<double, 6> sygeParams = fit(f, t, 2);

  f.resize(s.asmbl.size(), 5);  // asmbl model, and the per-element term
  t.resize(s.asmbl.size());
  for (size_t i = 0; i < s.asmbl.size(); i++) {
    const auto& a = s.asmbl[i];
    f.row(i).head<4>() = ComputationModel::dAsmblModel(a[0], a[1]);
    f(i, 4) = a[2] * a[3];
    t[i] = a[4];
  }
  Eigen::VectorXd asmbl = fit(f, t, 3);
  if (asmbl.size() != 5) asmbl = Eigen::VectorXd::Zero(5);

  return ComputationModel(potrfParams, trsmParams, sygeParams, asmbl.head<4>(), asmbl[4]);
}

string formatComputationModel(const ComputationModel& m) {
  auto vec = [](const auto& v) {
    ostringstream o;
    o.precision(17);
    o << "{";
    for (int i = 0; i < v.size(); i++) o << (i ? ", " : "") << v[i];
    o << "}";
    return o.str();
  };
  ostringstream o;
  o.precision(17);
  o << "{" << vec(m.potrfParams) << ",\n " << vec(m.trsmParams) << ",\n " << vec(m.sygeParams)
    << ",\n " << vec(m.asmblParams) << ",\n " << m.asmblElemParam << "}";
  return o.str();
}

string computationModelParams(const ComputationModel& m) {
  ostringstream o;
  o.precision(17);
  for (int i = 0; i < 4; i++) o << m.potrfParams[i] << " ";
  for (int i = 0; i < 6; i++) o << m.trsmParams[i] << " ";
  for (int i = 0; i < 6; i++) o << m.sygeParams[i] << " ";
  for (int i = 0; i < 4; i++) o << m.asmblParams[i] << " ";
  o << m.asmblElemParam;
  return o.str();
}

ComputationModel parseComputationModelParams(const string& params) {
  istringstream in(params);
  vector<double> v;
  double x;
  while (in >> x) v.push_back(x);
  if (v.size() != 20 && v.size() != 21) {
    throw runtime_error("computation model: expected 20 or 21 parameters, got " +
                        to_string(v.size()));
  }
  return ComputationModel(Eigen::Vector4d(v[0], v[1], v[2], v[3]),
                          Eigen::Map<Eigen::Matrix<double, 6, 1>>(&v[4]),
                          Eigen::Map<Eigen::Matrix<double, 6, 1>>(&v[10]),
                          Eigen::Vector4d(v[16], v[17], v[18], v[19]), v.size() > 20 ? v[20] : 0.0);
}

}  // end namespace BaSpaCho
