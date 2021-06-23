
#include "kernel.h"
#include "build_tree.h"
#include "aca.h"
#include "svd.h"

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
    a[(size_t)x * lda + y] = r2;
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


void nbd::P2M_L2P(eval_func_t r2f, int p, Cell* c, int dim) {
  Bodies outer(p);
  getBoundBox(p, c, outer, dim);

  int m = c->NBODY;
  Matrix& c2o = c->V;
  c2o.resize((size_t)m * p);

  for (int i = 0; i < m * p; i++) {
    int x = i / m, y = i - x * m;
    real_t r2;
    eval(r2f, c->BODY + y, outer.data() + x, dim, &r2);
    c2o[(size_t)x * m + y] = r2;
  }

  Matrix i2o((size_t)p * p);

  for (int i = 0; i < p * p; i++) {
    int x = i / p, y = i - x * p;
    real_t r2;
    eval(r2f, c->inner.data() + y, outer.data() + x, dim, &r2);
    i2o[(size_t)x * p + y] = r2;
  }

  Matrix u((size_t)p * p), v((size_t)p * p), s(p);
  int iters;
  daca(p, p, p, i2o.data(), p, u.data(), p, v.data(), p, &iters);
  dlr_svd(p, p, iters, u.data(), p, v.data(), p, s.data());
  dtsvd(p, p, iters, s.data(), &iters);

  dpinvr_svd(p, p, iters, s.data(), u.data(), p, v.data(), p, c2o.data(), m, m);

}


void nbd::M2L(eval_func_t r2f, const Cell* ci, const Cell* cj, int dim, Matrix& s) {
  int m = (int)ci->inner.size(), n = (int)cj->inner.size();
  s.resize((size_t)m * n);

  for (int i = 0; i < m * n; i++) {
    int x = i / m, y = i - x * m;
    real_t r2;
    eval(r2f, ci->inner.data() + y, cj->inner.data() + x, dim, &r2);
    s[(size_t)x * m + y] = r2;
  }
}