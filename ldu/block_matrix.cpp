
#include "block_matrix.h"

#include "../kernel.h"
#include "../build_tree.h"
#include "lapacke.h"
#include "cblas.h"

#include <cstddef>
#include <cstdlib>
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

void nbd::dschur(int m, int n, int k, const double* a, const double* b, double*& c) {
  if (m > 0 && n > 0 && k > 0 && a != nullptr && b != nullptr) {
    double beta = 1.;
    if (c == nullptr) {
      c = (double*)malloc(sizeof(double) * m * n);
      beta = 0.;
    }

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
