
#include "kernel.h"

#include <cmath>

using namespace nbd;

eval_func_t nbd::r2() {
  return [](real_t& r2) -> void {};
}

eval_func_t nbd::l3d() {
  return [](real_t& r2) -> void {
    r2 = r2 == 0 ? 1.e5 : 1. / std::sqrt(r2);
  };
}


void nbd::eval(eval_func_t r2f, const Body* bi, const Body* bj, int dim, real_t* out) {
  real_t& r2 = *out;
  r2 = 0.;
  for (int i = 0; i < dim; i++) {
    real_t dX = bi->X[i] - bj->X[i];
    r2 += dX * dX;
  }
  r2f(r2);
}


void nbd::dense_kernel(eval_func_t r2f, const Cell* ci, const Cell* cj, int dim, real_t* a, int lda) {
  int m = ci->NBODY, n = cj->NBODY;

  for (int i = 0; i < m * n; i++) {
    int x = i / m, y = i - x * m;
    real_t r2;
    eval(r2f, ci->BODY + y, cj->BODY + x, dim, &r2);
    a[x * lda + y] = r2;
  }
}


void nbd::mvec_kernel(eval_func_t r2f, const Cell* ci, const Cell* cj, int dim, const real_t* x_vec, real_t* b_vec) {
  int m = ci->NBODY, n = cj->NBODY;

  for (int y = 0; y < m; y++) {
    real_t sum = 0.;
    for (int x = 0; x < n; x++) {
      real_t r2;
      eval(r2f, ci->BODY + y, cj->BODY + x, dim, &r2);
      sum += r2 * x_vec[x];
    }
    b_vec[y] = sum;
  }
}
