
#include "aca.h"
#include "kernel.h"

#include <stdio.h>
#include <algorithm>
#include <cmath>

using namespace nbd;

void ddot(int64_t n, real_t* x, int64_t incx, real_t* y, int64_t incy, real_t* res) {
  real_t s = 0.;
  for (int64_t i = 0; i < n; i++)
    s += x[i * incx] * y[i * incy];
  *res = s;
}

void dscal(int64_t n, real_t a, real_t* x, int64_t incx) {
  for (int64_t i = 0; i < n; i++)
    x[i * incx] = a * x[i * incx];
}

void nbd::raca(eval_func_t r2f, const Cell* ci, const Cell* cj, int dim, int64_t max_iters, real_t* u, int64_t ldu, real_t* v, int64_t ldv, int* info) {

  int64_t y = 0, x, iter = 1, m = ci->NBODY, n = cj->NBODY, piv_i = 0;
  real_t piv = 0;
  std::vector<int64_t> p_u, p_v;

  p_u.reserve(max_iters);
  p_v.reserve(max_iters);

#ifdef ACA_USE_NORM
  real_t my_norm_u, my_norm_v, epi2 = ACA_EPI * ACA_EPI;
  real_t norm;
#endif

  for (int64_t i = 0; i < n; i++) {
    real_t s;
    eval(r2f, ci->BODY + y, cj->BODY + i, dim, &s);
    v[i] = s;

    if (std::abs(s) > std::abs(piv))
    { piv_i = i; piv = s; }
  }

  x = piv_i;
  dscal(n, 1. / piv, v, 1);

  piv = 0.;
  for (int64_t i = 0; i < m; i++) {
    real_t s;
    eval(r2f, ci->BODY + i, cj->BODY + x, dim, &s);
    u[i] = s;

    if (i > 0 && std::abs(s) > std::abs(piv))
    { piv_i = i; piv = s; }
  }

  p_u.push_back(0);
  p_v.push_back(x);

#ifdef ACA_USE_NORM
  ddot(m, u, 1, u, 1, &my_norm_u);
  ddot(n, v, 1, v, 1, &my_norm_v);

  real_t n2 = my_norm_u * my_norm_v;
  if (n2 <= epi2 * (norm = n2))
    piv_i = -1;
#endif

  while (piv_i != -1 && iter < max_iters) {
    real_t* curr_u = u + iter * ldu;
    real_t* curr_v = v + iter * ldv;
    y = piv_i;

    piv = 0.;
    for (int64_t i = 0; i < n; i++) {
      real_t vi;
      eval(r2f, ci->BODY + y, cj->BODY + i, dim, &vi);

      real_t s;
      ddot(iter, u + y, ldu, v + i, ldv, &s);
      vi -= s;
      curr_v[i] = vi;

      auto p = std::find(p_v.begin(), p_v.end(), i);

      if (p == p_v.end() && std::abs(vi) > std::abs(piv))
      { piv_i = i; piv = vi; }
    }

    x = piv_i;
    dscal(n, 1. / piv, curr_v, 1);

    p_u.push_back(y);
    p_v.push_back(x);

    piv = 0.;
    for (int64_t i = 0; i < m; i++) {
      real_t ui;
      eval(r2f, ci->BODY + i, cj->BODY + x, dim, &ui);

      real_t s;
      ddot(iter, v + x, ldv, u + i, ldu, &s);
      ui -= s;
      curr_u[i] = ui;

      auto p = std::find(p_u.begin(), p_u.end(), i);

      if (p == p_u.end() && std::abs(ui) > std::abs(piv))
      { piv_i = i; piv = ui; }
    }

#ifdef ACA_USE_NORM
    for (int64_t j = 0; j < iter; j++) {
      ddot(m, u + j * ldu, 1, curr_u, 1, &my_norm_u);
      ddot(n, v + j * ldv, 1, curr_v, 1, &my_norm_v);

      norm += 2. * my_norm_u * my_norm_v;
    }
    ddot(m, curr_u, 1, curr_u, 1, &my_norm_u);
    ddot(n, curr_v, 1, curr_v, 1, &my_norm_v);

    real_t n2 = my_norm_u * my_norm_v;
    if (n2 <= epi2 * (norm += n2))
      piv_i = -1;
#endif
    iter++;
  }

  if (info != nullptr)
    *info = (int)iter;
}


void nbd::raca(int64_t m, int64_t n, int64_t max_iters, const real_t* a, int64_t lda, real_t* u, int64_t ldu, real_t* v, int64_t ldv, int* info) {
  int64_t y = 0, x, iter = 1, piv_i = 0;
  real_t piv = 0;
  std::vector<int64_t> p_u, p_v;

  p_u.reserve(max_iters);
  p_v.reserve(max_iters);

#ifdef ACA_USE_NORM
  real_t my_norm_u, my_norm_v, epi2 = ACA_EPI * ACA_EPI;
  real_t norm;
#endif

  for (int64_t i = 0; i < n; i++) {
    real_t s = a[i * lda + y];
    v[i] = s;

    if (std::abs(s) > std::abs(piv))
    { piv_i = i; piv = s; }
  }

  x = piv_i;
  dscal(n, 1. / piv, v, 1);

  piv = 0.;
  for (int64_t i = 0; i < m; i++) {
    real_t s = a[x * lda + i];
    u[i] = s;

    if (i > 0 && std::abs(s) > std::abs(piv))
    { piv_i = i; piv = s; }
  }

  p_u.push_back(0);
  p_v.push_back(x);

#ifdef ACA_USE_NORM
  ddot(m, u, 1, u, 1, &my_norm_u);
  ddot(n, v, 1, v, 1, &my_norm_v);

  real_t n2 = my_norm_u * my_norm_v;
  if (n2 <= epi2 * (norm = n2))
    piv_i = -1;
#endif

  while (piv_i != -1 && iter < max_iters) {
    real_t* curr_u = u + iter * ldu;
    real_t* curr_v = v + iter * ldv;
    y = piv_i;

    piv = 0.;
    for (int64_t i = 0; i < n; i++) {
      real_t vi = a[i * lda + y];

      real_t s;
      ddot(iter, u + y, ldu, v + i, ldv, &s);
      vi -= s;
      curr_v[i] = vi;

      auto p = std::find(p_v.begin(), p_v.end(), i);

      if (p == p_v.end() && std::abs(vi) > std::abs(piv))
      { piv_i = i; piv = vi; }
    }

    x = piv_i;
    dscal(n, 1. / piv, curr_v, 1);

    p_u.push_back(y);
    p_v.push_back(x);

    piv = 0.;
    for (int64_t i = 0; i < m; i++) {
      real_t ui = a[x * lda + i];

      real_t s;
      ddot(iter, v + x, ldv, u + i, ldu, &s);
      ui -= s;
      curr_u[i] = ui;

      auto p = std::find(p_u.begin(), p_u.end(), i);

      if (p == p_u.end() && std::abs(ui) > std::abs(piv))
      { piv_i = i; piv = ui; }
    }

#ifdef ACA_USE_NORM
    for (int64_t j = 0; j < iter; j++) {
      ddot(m, u + j * ldu, 1, curr_u, 1, &my_norm_u);
      ddot(n, v + j * ldv, 1, curr_v, 1, &my_norm_v);

      norm += 2. * my_norm_u * my_norm_v;
    }
    ddot(m, curr_u, 1, curr_u, 1, &my_norm_u);
    ddot(n, curr_v, 1, curr_v, 1, &my_norm_v);

    real_t n2 = my_norm_u * my_norm_v;
    if (n2 <= epi2 * (norm += n2))
      piv_i = -1;
#endif
    iter++;
  }

  if (info != nullptr)
    *info = (int)iter;
}


#include <cstdio>
#include <cstdlib>
#include <random>
#include <chrono>

int main(int argc, char* argv[]) {

  int rank = argc > 1 ? atoi(argv[1]) : 16;
  int m = argc > 2 ? atoi(argv[2]) : 32;
  int n = argc > 3 ? atoi(argv[3]) : m;

  std::srand(199);
  std::vector<real_t> left(m, rank), right(n, rank);

  for(auto& i : left)
    i = ((real_t)std::rand() / RAND_MAX) * 100;
  for(auto& i : right)
    i = ((real_t)std::rand() / RAND_MAX) * 100;

  std::vector<real_t> a(m, n), u(m, rank), v(n, rank);

  for(int j = 0; j < n; j++) {
    for(int i = 0; i < m; i++) {
      real_t e = 0.;
      for(int k = 0; k < rank; k++)
        e += left[i + k * m] * right[j + k * m];
      a[i + j * m] = e;
    }
  }

  using namespace nbd;

  int iters;
  raca(m, n, rank, a.data(), m, u.data(), m, v.data(), n, &iters);

  double err = 0.;
  for(int j = 0; j < n; j++) {
    for(int i = 0; i < m; i++) {
      real_t e = 0.;
      for(int k = 0; k < iters; k++)
        e += u[i + k * m] * v[j + k * m];
      e -= a[i + j * m];
      err += e * e;
    }
  }

  printf("abs err: %e, aca iters %d\n", err, iters);

  return 0;
}
