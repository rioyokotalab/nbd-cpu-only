
#include "kernel.h"
#include "build_tree.h"
#include "aca.h"
#include "svd.h"

#include <cmath>

using namespace nbd;

eval_func_t nbd::r2() {
  return [](real_t& r2) -> void {};
}

eval_func_t nbd::l2d() {
  return [](real_t& r2) -> void {
    r2 = r2 == 0 ? 1.e5 : std::log(std::sqrt(r2));
  };
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


void nbd::P2P(eval_func_t r2f, const Cell* ci, const Cell* cj, int dim, Matrix& a) {
  int m = ci->NBODY, n = cj->NBODY;
  a.A.resize((size_t)m * n);
  a.M = m;
  a.N = n;
  a.LDA = m;

  for (int i = 0; i < m * n; i++) {
    int x = i / m, y = i - x * m;
    real_t r2;
    eval(r2f, ci->BODY + y, cj->BODY + x, dim, &r2);
    a[(size_t)x * a.LDA + y] = r2;
  }
}

void nbd::M2L(eval_func_t r2f, const Cell* ci, const Cell* cj, int dim, Matrix& u, Matrix& v, int rank) {
  int m = ci->NBODY, n = cj->NBODY;
  u.A.resize((size_t)m * rank);
  u.M = u.LDA = m;

  v.A.resize((size_t)n * rank);
  v.M = v.LDA = n;
  u.N = v.N = rank;

  int iters;
  daca_cells(r2f, ci, cj, dim, rank, u, u.LDA, v, v.LDA, &iters);
  u.A.resize((size_t)m * iters);
  v.A.resize((size_t)n * iters);
  u.N = v.N = iters;
}
