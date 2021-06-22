
#include "aca.h"
#include "kernel.h"

#include "mkl.h"
#include <algorithm>
#include <cmath>
#include <cstddef>

using namespace nbd;

void nbd::daca(eval_func_t r2f, const Cell* ci, const Cell* cj, int dim, int max_iters, double* u, int ldu, double* v, int ldv, int* info) {

  int y = 0, x, iter = 1, m = ci->NBODY, n = cj->NBODY, piv_i = 0;
  double piv = 0;
  std::vector<int> p_u, p_v;

  p_u.reserve(max_iters);
  p_v.reserve(max_iters);

#ifdef ACA_USE_NORM
  double my_norm_u, my_norm_v, epi2 = ACA_EPI * ACA_EPI;
  double norm;
#endif

  for (int i = 0; i < n; i++) {
    real_t s;
    eval(r2f, ci->BODY + y, cj->BODY + i, dim, &s);
    v[i] = s;

    if (std::abs(s) > std::abs(piv))
    { piv_i = i; piv = s; }
  }

  x = piv_i;
  cblas_dscal(n, 1. / piv, v, 1);

  piv = 0.;
  for (int i = 0; i < m; i++) {
    real_t s;
    eval(r2f, ci->BODY + i, cj->BODY + x, dim, &s);
    u[i] = s;

    if (i > 0 && std::abs(s) > std::abs(piv))
    { piv_i = i; piv = s; }
  }

  p_u.push_back(0);
  p_v.push_back(x);

#ifdef ACA_USE_NORM
  my_norm_u = cblas_ddot(m, u, 1, u, 1);
  my_norm_v = cblas_ddot(n, v, 1, v, 1);

  double n2 = my_norm_u * my_norm_v;
  if (n2 <= epi2 * (norm = n2))
    piv_i = -1;
#endif

  while (piv_i != -1 && iter < max_iters) {
    double* curr_u = u + (size_t)iter * ldu;
    double* curr_v = v + (size_t)iter * ldv;
    y = piv_i;

    piv = 0.;
    for (int i = 0; i < n; i++) {
      real_t vi;
      eval(r2f, ci->BODY + y, cj->BODY + i, dim, &vi);

      double s = cblas_ddot(iter, u + y, ldu, v + i, ldv);
      vi -= s;
      curr_v[i] = vi;

      auto p = std::find(p_v.begin(), p_v.end(), i);

      if (p == p_v.end() && std::abs(vi) > std::abs(piv))
      { piv_i = i; piv = vi; }
    }

    x = piv_i;
    cblas_dscal(n, 1. / piv, curr_v, 1);

    p_u.push_back(y);
    p_v.push_back(x);

    piv = 0.;
    for (int i = 0; i < m; i++) {
      real_t ui;
      eval(r2f, ci->BODY + i, cj->BODY + x, dim, &ui);

      double s = cblas_ddot(iter, v + x, ldv, u + i, ldu);
      ui -= s;
      curr_u[i] = ui;

      auto p = std::find(p_u.begin(), p_u.end(), i);

      if (p == p_u.end() && std::abs(ui) > std::abs(piv))
      { piv_i = i; piv = ui; }
    }

#ifdef ACA_USE_NORM
    for (int j = 0; j < iter; j++) {
      my_norm_u = cblas_ddot(m, u + (size_t)j * ldu, 1, curr_u, 1);
      my_norm_v = cblas_ddot(n, v + (size_t)j * ldv, 1, curr_v, 1);

      norm += 2. * my_norm_u * my_norm_v;
    }
    my_norm_u = cblas_ddot(m, curr_u, 1, curr_u, 1);
    my_norm_v = cblas_ddot(n, curr_v, 1, curr_v, 1);

    double n2 = my_norm_u * my_norm_v;
    if (n2 <= epi2 * (norm += n2))
      piv_i = -1;
#endif
    iter++;
  }

  if (info != nullptr)
    *info = (int)iter;
}


void nbd::daca(int m, int n, int max_iters, const double* a, int lda, double* u, int ldu, double* v, int ldv, int* info) {
  int y = 0, x, iter = 1, piv_i = 0;
  double piv = 0;
  std::vector<int> p_u, p_v;

  p_u.reserve(max_iters);
  p_v.reserve(max_iters);

#ifdef ACA_USE_NORM
  double my_norm_u, my_norm_v, epi2 = ACA_EPI * ACA_EPI;
  double norm;
#endif

  for (int i = 0; i < n; i++) {
    double s = a[i * lda + y];
    v[i] = s;

    if (std::abs(s) > std::abs(piv))
    { piv_i = i; piv = s; }
  }

  x = piv_i;
  cblas_dscal(n, 1. / piv, v, 1);

  piv = 0.;
  for (int i = 0; i < m; i++) {
    double s = a[x * lda + i];
    u[i] = s;

    if (i > 0 && std::abs(s) > std::abs(piv))
    { piv_i = i; piv = s; }
  }

  p_u.push_back(0);
  p_v.push_back(x);

#ifdef ACA_USE_NORM
  my_norm_u = cblas_ddot(m, u, 1, u, 1);
  my_norm_v = cblas_ddot(n, v, 1, v, 1);

  double n2 = my_norm_u * my_norm_v;
  if (n2 <= epi2 * (norm = n2))
    piv_i = -1;
#endif

  while (piv_i != -1 && iter < max_iters) {
    double* curr_u = u + (size_t)iter * ldu;
    double* curr_v = v + (size_t)iter * ldv;
    y = piv_i;

    piv = 0.;
    for (int i = 0; i < n; i++) {
      double vi = a[i * lda + y];

      double s = cblas_ddot(iter, u + y, ldu, v + i, ldv);
      vi -= s;
      curr_v[i] = vi;

      auto p = std::find(p_v.begin(), p_v.end(), i);

      if (p == p_v.end() && std::abs(vi) > std::abs(piv))
      { piv_i = i; piv = vi; }
    }

    x = piv_i;
    cblas_dscal(n, 1. / piv, curr_v, 1);

    p_u.push_back(y);
    p_v.push_back(x);

    piv = 0.;
    for (int i = 0; i < m; i++) {
      double ui = a[x * lda + i];

      double s = cblas_ddot(iter, v + x, ldv, u + i, ldu);
      ui -= s;
      curr_u[i] = ui;

      auto p = std::find(p_u.begin(), p_u.end(), i);

      if (p == p_u.end() && std::abs(ui) > std::abs(piv))
      { piv_i = i; piv = ui; }
    }

#ifdef ACA_USE_NORM
    for (int j = 0; j < iter; j++) {
      my_norm_u = cblas_ddot(m, u + (size_t)j * ldu, 1, curr_u, 1);
      my_norm_v = cblas_ddot(n, v + (size_t)j * ldv, 1, curr_v, 1);

      norm += 2. * my_norm_u * my_norm_v;
    }
    my_norm_u = cblas_ddot(m, curr_u, 1, curr_u, 1);
    my_norm_v = cblas_ddot(n, curr_v, 1, curr_v, 1);

    double n2 = my_norm_u * my_norm_v;
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

  rank = std::min(rank, m);
  rank = std::min(rank, n);

  int rp = rank + 8;

  std::srand(199);
  std::vector<double> left(m * rank), right(n * rank);

  for(auto& i : left)
    i = ((double)std::rand() / RAND_MAX) * 100;
  for(auto& i : right)
    i = ((double)std::rand() / RAND_MAX) * 100;

  std::vector<double> a(m * n), u(m * rp), v(n * rp);

  for(int j = 0; j < n; j++) {
    for(int i = 0; i < m; i++) {
      double e = 0.;
      for(int k = 0; k < rank; k++)
        e += left[i + k * m] * right[j + k * n];
      a[i + j * m] = e;
    }
  }

  using namespace nbd;

  int iters;
  daca(m, n, rp, a.data(), m, u.data(), m, v.data(), n, &iters);

  double err = 0., nrm = 0.;
  for(int j = 0; j < n; j++) {
    for(int i = 0; i < m; i++) {
      double e = 0.;
      for(int k = 0; k < iters; k++)
        e += u[i + k * m] * v[j + k * n];
      e -= a[i + j * m];
      err += e * e;
      nrm += a[i + j * m];
    }
  }

  printf("rel err: %e, aca iters %d\n", std::sqrt(err / nrm), iters);

  return 0;
}
