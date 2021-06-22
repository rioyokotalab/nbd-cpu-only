
#include "aca.h"
#include "kernel.h"

#include "mkl.h"
#include <algorithm>
#include <cmath>
#include <cstddef>

using namespace nbd;

struct InputMat {
  int m;
  int n;
  const double* a;
  int lda;

  eval_func_t r2f;
  const Cell* ci;
  const Cell* cj; 
  int dim;
};

inline void _eval(const InputMat* M, int y, int x, real_t* out) {
  if (M->a != NULL)
    *out = M->a[y + (size_t)M->lda * x];
  else
    eval(M->r2f, M->ci->BODY + y, M->cj->BODY + x, M->dim, out);
}

void _daca(const InputMat* M, int max_iters, double* u, int ldu, double* v, int ldv, int* info) {

  int y = 0, x, iter = 1, m = M->m, n = M->n, piv_i = 0;
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
    _eval(M, y, i, &s);
    v[i] = s;

    if (std::abs(s) > std::abs(piv))
    { piv_i = i; piv = s; }
  }

  x = piv_i;
  cblas_dscal(n, 1. / piv, v, 1);

  piv = 0.;
  for (int i = 0; i < m; i++) {
    real_t s;
    _eval(M, i, x, &s);
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
      _eval(M, y, i, &vi);

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
      _eval(M, i, x, &ui);

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

  if (info)
    *info = (int)iter;
}

void nbd::daca(eval_func_t r2f, const Cell* ci, const Cell* cj, int dim, int max_iters, double* u, int ldu, double* v, int ldv, int* info) {
  InputMat M{ ci->NBODY, cj->NBODY, NULL, 0, r2f, ci, cj, dim };
  _daca(&M, max_iters, u, ldu, v, ldv, info);
}

void nbd::daca(int m, int n, int max_iters, const double* a, int lda, double* u, int ldu, double* v, int ldv, int* info) {
  InputMat M{ m, n, a, lda, NULL, NULL, NULL, 0 };
  _daca(&M, max_iters, u, ldu, v, ldv, info);
}
