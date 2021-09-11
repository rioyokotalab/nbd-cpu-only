
#include "close_invert.h"

#include "kernel.h"
#include "build_tree.h"
#include "lapacke.h"
#include "cblas.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <algorithm>

using namespace nbd;

void nbd::dgetf2np(int m, int n, double* a, int lda) {
  int k = m < n ? m : n;
  for (int i = 0; i < k; i++) {
    double p = 1. / a[i + (size_t)i * lda];
    int mi = m - i - 1;
    int ni = n - i - 1;

    double* ax = a + i + (size_t)i * lda + 1;
    double* ay = a + i + (size_t)i * lda + lda;
    double* an = ay + 1;

    cblas_dscal(mi, p, ax, 1);
    cblas_dger(CblasColMajor, mi, ni, -1., ax, 1, ay, lda, an, lda);
  }
}

void nbd::dtrsml(int m, int n, const double* a, double* b) {
  if (m > 0 && n > 0 && a != nullptr && b != nullptr)
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, m, n, 1., a, m, b, m);
}

void nbd::dtrsmc(int m, int n, const double* a, double* b) {
  if (m > 0 && n > 0 && a != nullptr && b != nullptr)
    cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, m, n, 1., a, n, b, m);
}

void nbd::dschur(int m, int n, int k, const double* a, const double* b, double* c) {
  if (m > 0 && n > 0 && k > 0 && a != nullptr && b != nullptr) {
    double beta = 1.;
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, -1., a, m, b, k, beta, c, m);
  }
}


void nbd::dtrsvf(int n, const double* a, double* x) {
  cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit, n, a, n, x, 1);
}

void nbd::dtrsvb(int n, const double* a, double* x) {
  cblas_dtrsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, a, n, x, 1);
}

void nbd::dschurv(int m, int n, const double* a, const double* x, double* y) {
  if (a != nullptr)
    cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, -1., a, m, x, 1, 1., y, 1);
}


Matrix nbd::near(const Cells& icells, const Cells& jcells, const Matrices& d) {
  Matrix A(icells[0].NBODY, jcells[0].NBODY, icells[0].NBODY);

  double* a = A.A.data();
  int lda = A.LDA;
  memset(a, 0, sizeof(double) * icells[0].NBODY * jcells[0].NBODY);

  auto j_begin = jcells[0].BODY;
  auto i_begin = icells[0].BODY;
  int ld = (int)icells.size();

#pragma omp parallel for
  for (int y = 0; y < icells.size(); y++) {
    auto i = icells[y];
    auto yi = i.BODY - i_begin;
    for (auto& j : i.listNear) {
      auto _x = j - &jcells[0];
      auto xi = j->BODY - j_begin;
      const Matrix& m = d[y + (size_t)_x * ld];
      for (int jj = 0; jj < m.N; jj++)
        for (int ii = 0; ii < m.M; ii++)
          a[ii + yi + (jj + xi) * lda] = m.A[ii + (size_t)jj * m.LDA];
    }
  }

  dgetf2np(A.M, A.N, a, lda);
  return A;
}


void near_solve(const Matrix& d, const double* b, double* x) {
  if (b != x)
    cblas_dcopy(d.M, b, 1, x, 1);
  dtrsvf(d.M, d, x);
  dtrsvb(d.M, d, x);
}
