
#include "kernel.hxx"
#include "build_tree.hxx"

#include "stdio.h"
#include <cmath>
#include <random>

using namespace nbd;

double _singularity = 1.e-8;
double _alpha = 1.;

void nbd::laplace3d(double* r2) {
  double _r2 = *r2;
  double r = std::sqrt(_r2) + _singularity;
  *r2 = 1. / r;
}

void nbd::yukawa3d(double* r2) {
  double _r2 = *r2;
  double r = std::sqrt(_r2) + _singularity;
  *r2 = std::exp(_alpha * -r) / r;
}

void nbd::set_kernel_constants(double singularity, double alpha) {
  _singularity = singularity;
  _alpha = alpha;
}

void nbd::eval(eval_func_t ef, const Body* bi, const Body* bj, int64_t dim, double* out) {
  double& r2 = *out;
  r2 = 0.;
  for (int64_t i = 0; i < dim; i++) {
    double dX = bi->X[i] - bj->X[i];
    r2 += dX * dX;
  }
  ef(out);
}


void nbd::P2Pmat(eval_func_t ef, int64_t m, int64_t n, const Body bi[], const Body bj[], int64_t dim, Matrix& a, const int64_t sel_i[], const int64_t sel_j[]) {
  for (int64_t i = 0; i < m * n; i++) {
    int64_t x = i / m;
    int64_t bx = sel_j == NULL ? x : sel_j[x];
    int64_t y = i - x * m;
    int64_t by = sel_i == NULL ? y : sel_i[y];

    double r2;
    eval(ef, bi + by, bj + bx, dim, &r2);
    a.A[x * a.M + y] = r2;
  }
}


