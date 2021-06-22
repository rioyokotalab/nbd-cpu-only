
#include "svd.h"
#include "kernel.h"

#include "mkl.h"
#include <algorithm>
#include <cmath>
#include <cstddef>

using namespace nbd;

void nbd::dsvd(eval_func_t r2f, const Cell* ci, const Cell* cj, int dim, double* s, double* u, int ldu, double* v, int ldv, int* info) {
  int m = ci->NBODY, n = cj->NBODY;
  std::vector<double> a((size_t)m * n);
  dense_kernel(r2f, ci, cj, dim, &a[0], m);
  dsvd(m, n, &a[0], m, s, u, ldu, v, ldv, info);
}

void nbd::dsvd(int m, int n, double* a, int lda, double* s, double* u, int ldu, double* v, int ldv, int* info) {
  int r = std::min(m, n);
  std::vector<double> superb((size_t)r - 1), vt((size_t)r * n);
  int i = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'S', m, n, a, lda, s, u, ldu, &vt[0], r, &superb[0]);
  mkl_domatcopy('C', 'T', r, n, 1., &vt[0], r, v, ldv);
  if (info)
    *info = i;
}

void nbd::dlr_svd(int m, int n, int r, double* u, int ldu, double* v, int ldv, double* s, int* info) {
  std::vector<double> a((size_t)r * r, 0.);
  std::vector<double> left((size_t)m * r);
  std::vector<double> right((size_t)n * r);
  std::vector<double> Ltau(r);
  std::vector<double> Rtau(r);

  LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, r, u, ldu, &Ltau[0]);
  LAPACKE_dgeqrf(LAPACK_COL_MAJOR, n, r, v, ldv, &Rtau[0]);

  LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', r, r, u, ldu, &a[0], r);
  cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit, r, r, 1., v, ldv, &a[0], r);

  dsvd(r, r, &a[0], r, s, &left[0], m, &right[0], n, info);

  LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'N', m, r, r, u, ldu, &Ltau[0], &left[0], m);
  LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'N', n, r, r, v, ldv, &Rtau[0], &right[0], n);

  mkl_domatcopy('C', 'N', m, r, 1., &left[0], m, u, ldu);
  mkl_domatcopy('C', 'N', n, r, 1., &right[0], n, v, ldv);
}


void nbd::dtsvd(int m, int n, int r, double* s, int* out) {
  double t = std::numeric_limits<double>::epsilon() * std::max(m, n) * s[0];
  auto p = std::find_if(s, s + r, [t](double si) { return si < t; });
  *out = (int)(p - s);
}


void nbd::dpinvl_svd(int m, int n, int r, const double* s, const double* u, int ldu, const double* v, int ldv, double* a, int na, int lda) {
  std::vector<double> work((size_t)r * na);
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, r, na, m, 1., u, ldu, a, lda, 0., &work[0], r);

  for (int i = 0; i < r; i++)
    cblas_dscal(na, 1. / s[i], &work[i], r);

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, na, r, 1., v, ldv, &work[0], r, 0., a, lda);
}

void nbd::dpinvr_svd(int m, int n, int r, const double* s, const double* u, int ldu, const double* v, int ldv, double* a, int ma, int lda) {
  std::vector<double> work((size_t)ma * r);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ma, r, n, 1., a, lda, v, ldv, 0., &work[0], ma);

  for (int i = 0; i < r; i++)
    cblas_dscal(ma, 1. / s[i], &work[(size_t)i * ma], 1);

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, ma, m, r, 1., &work[0], ma, u, ldu, 0., a, lda);
}
