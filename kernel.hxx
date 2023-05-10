#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>

struct EvalDouble {
  virtual double operator()(double d) const = 0;
};

struct Laplace2D : public EvalDouble {
  double singularity;
  double diag_shift;
  Laplace2D (double s, double diag_shift = 0.) : singularity(1. / s), diag_shift(diag_shift) {}
  double operator()(double d) const override {
    double x = d == 0. ? singularity : std::log(d);
    if (d == 0.) return x + diag_shift;
    else return x;
  }
};

struct Laplace3D : public EvalDouble {
  double singularity;
  double diag_shift;
  Laplace3D (double s, double diag_shift = 0.) : singularity(1. / s), diag_shift(diag_shift) {}
  double operator()(double d) const override {
    double x = d == 0. ? singularity : (1. / d);
    if (d == 0.) return x + diag_shift;
    else return x;
  }
};

struct Yukawa3D : public EvalDouble {
  double singularity, alpha;
  double diag_shift;
  Yukawa3D (double s, double a, double diag_shift = 0.) : singularity(1. / s), alpha(a), diag_shift(diag_shift) {}
  double operator()(double d) const override {
    double x = d == 0. ? singularity : (std::exp(-alpha * d) / d);
    if (d == 0.) return x + diag_shift;
    else return x;
  }
};

struct Gaussian : public EvalDouble {
  double alpha;
  double diag_shift;
  Gaussian (double a, double diag_shift = 0.) : alpha(1. / (a * a)), diag_shift(diag_shift) {}
  double operator()(double d) const override {
    double x = std::exp(- alpha * d * d);
    if (d == 0.) return x + diag_shift;
    else return x;
  }
};

struct Toeplitz : public EvalDouble {
  double p;
  double diag_shift;
  Toeplitz (double p, double diag_shift = 0.) : p(p), diag_shift(diag_shift) {}
  double operator()(double d) const override {
    double x = std::pow(p, d);
    if (d == 0.) return x + diag_shift;
    else return x;
  }
};
