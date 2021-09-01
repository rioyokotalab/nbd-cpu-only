
#include "block_matrix.h"

#include "../kernel.h"
#include "../build_tree.h"
#include "lapacke.h"
#include "cblas.h"

#include <cstddef>
#include <cstdlib>
#include <algorithm>

using namespace irs;

void irs::matcpy(const nbd::Matrix& m1, int m, int n, double* a) {
  if (m > 0 && n > 0 && m <= m1.M && n <= m1.N)
    for (int i = 0; i < n; i++)
      cblas_dcopy(m, &m1.A[(size_t)i * m1.LDA], 1, &a[(size_t)i * m], 1);
}


BlockMatrix irs::build(nbd::eval_func_t r2f, int dim, const nbd::Cells& cells, double theta) {
  nbd::Cells leaf = nbd::getLeaves(cells);
  nbd::getList(&leaf[0], &leaf[0], dim, theta, true);
  
  int N = leaf[0].NCHILD;
  BlockMatrix d;
  d.N.resize(N);
  d.A.resize((size_t)N * N, nullptr);
  
  for (int x = 0; x < N; x++)
    d.N[x] = leaf[x + 1].NBODY;

#pragma omp parallel for
  for (int y = 0; y < N; y++) {
    auto i = leaf[y + 1];
    for (auto& j : i.listNear) {
      auto _x = j - &leaf[1];
      nbd::Matrix m;
      nbd::P2Pnear(r2f, &leaf[y + 1], &leaf[_x + 1], dim, m);

      double* a = (double*)malloc(sizeof(double) * d.N[y] * d.N[_x]);
      matcpy(m, d.N[y], d.N[_x], a);
      d.A[y + (size_t)N * _x] = a;
    }
  }

  return d;
}

void irs::dgetf2np(int m, int n, double* a, int lda) {
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

void irs::dtrsml(int m, int n, const double* a, double* b) {
  if (m > 0 && n > 0 && a != nullptr && b != nullptr)
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, m, n, 1., a, m, b, m);
}

void irs::dtrsmc(int m, int n, const double* a, double* b) {
  if (m > 0 && n > 0 && a != nullptr && b != nullptr)
    cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, m, n, 1., a, n, b, m);
}

void irs::dschur(int m, int n, int k, const double* a, const double* b, double*& c) {
  if (m > 0 && n > 0 && k > 0 && a != nullptr && b != nullptr) {
    double beta = 1.;
    if (c == nullptr) {
      c = (double*)malloc(sizeof(double) * m * n);
      beta = 0.;
    }

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, -1., a, m, b, k, beta, c, m);
  }
}

void irs::elim(BlockMatrix& d) {
  
  int n = d.N.size();
  for (int i = 0; i < n; i++) {
    
    double* ai = d.A[i + (size_t)i * n];
    dgetf2np(d.N[i], d.N[i], ai, d.N[i]);

    for (int j = i + 1; j < n; j++) {
      double* aij = d.A[i + (size_t)j * n];
      dtrsml(d.N[i], d.N[j], ai, aij);
    }

    for (int j = i + 1; j < n; j++) {
      double* aji = d.A[j + (size_t)i * n];
      dtrsmc(d.N[j], d.N[i], ai, aji);
      for (int k = i + 1; k < n; k++) {
        double* aik = d.A[i + (size_t)k * n];
        double*& ajk = d.A[j + (size_t)k * n];
        dschur(d.N[j], d.N[k], d.N[i], aji, aik, ajk);
      }
    }
  }

}


void irs::dtrsvf(int n, const double* a, double* x) {
  cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit, n, a, n, x, 1);
}

void irs::dtrsvb(int n, const double* a, double* x) {
  cblas_dtrsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, a, n, x, 1);
}

void irs::dschurv(int m, int n, const double* a, const double* x, double* y) {
  if (a != nullptr)
    cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, -1., a, m, x, 1, 1., y, 1);
}

void irs::solve(const BlockMatrix& d, double* x) {
  int n = d.N.size();
  std::vector<double*> xi(n);
  xi[0] = x;
  for (int i = 0; i < n - 1; i++)
    xi[i + 1] = xi[i] + d.N[i];

  for (int i = 0; i < n; i++) {
    
    const double* ai = d.A[i + (size_t)i * n];
    dtrsvf(d.N[i], ai, xi[i]);

    for (int j = i + 1; j < n; j++) {
      double* aji = d.A[j + (size_t)i * n];
      dschurv(d.N[j], d.N[i], aji, xi[i], xi[j]);
    }
  }

  for (int i = n - 1; i >= 0; i--) {

    for (int j = i + 1; j < n; j++) {
      double* aij = d.A[i + (size_t)j * n];
      dschurv(d.N[i], d.N[j], aij, xi[j], xi[i]);
    }
    
    const double* ai = d.A[i + (size_t)i * n];
    dtrsvb(d.N[i], ai, xi[i]);
  }
}


void irs::clear(BlockMatrix& d) {
  for (auto& a : d.A)
    if (a != nullptr)
      free(a);

  d.A.clear();
  d.N.clear();
}
