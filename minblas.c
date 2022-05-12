
#include "minblas.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef CBLAS
#include "cblas.h"
#include "lapacke.h"
#endif

int64_t FLOPS = 0;

void dlra(double epi, int64_t m, int64_t n, int64_t k, double* a, double* u, int64_t ldu, double* vt, int64_t ldvt, int64_t* rank, int64_t* piv) {
  double nrm = 0.;
  double epi2 = epi * epi;
  double n2 = 1.;
  double amax = 1.;

  int64_t i = 0;
  while (i < k && (n2 > epi2 * nrm) && amax > 0) {
    amax = 0.;
    int64_t ymax = 0;
    Cidamax(m * n, a, 1, &ymax);
    amax = fabs(a[ymax]);

    if (amax > 0.) {
      if (piv != NULL)
        piv[i] = ymax;
      int64_t xp = ymax / m;
      int64_t yp = ymax - xp * m;
      double ap = 1. / a[ymax];
      double* ui = &u[i * ldu];
      double* vi = &vt[i * ldvt];

      Cdcopy(n, &a[yp], m, vi, 1);
      Cdcopy(m, &a[xp * m], 1, ui, 1);
      Cdscal(m, ap, ui, 1);
      ui[yp] = 1.;

      for (int64_t x = 0; x < n; x++) {
        double ri = -vi[x];
        Cdaxpy(m, ri, ui, 1, &a[x * m], 1);
      }

      double zero = 0.;
      Cdcopy(n, &zero, 0, &a[yp], m);
      Cdcopy(m, &zero, 0, &a[xp * m], 1);

      if (epi2 > 0.) {
        double nrm_v = 0.;
        double nrm_u = 0.;
        double nrm_vi = 0.;
        double nrm_ui = 0.;

        Cddot(n, vi, 1, vi, 1, &nrm_vi);
        Cddot(m, ui, 1, ui, 1, &nrm_ui);

        for (int64_t j = 0; j < i; j++) {
          double* vj = &vt[j * ldvt];
          double* uj = &u[j * ldu];
          double nrm_vj = 0.;
          double nrm_uj = 0.;

          Cddot(n, vi, 1, vj, 1, &nrm_vj);
          Cddot(m, ui, 1, uj, 1, &nrm_uj);
          nrm_v = nrm_v + nrm_vj;
          nrm_u = nrm_u + nrm_uj;
        }

        n2 = nrm_ui * nrm_vi;
        nrm = nrm + 2. * nrm_u * nrm_v + n2;
      }
      i++;
    }
  }

  *rank = i;
}

void didrow(double epi, int64_t m, int64_t n, int64_t k, double* a, double* x, int64_t ldx, int64_t* arow, int64_t* rank) {
  int64_t rnk;
  double* u = (double*)malloc(sizeof(double) * m * k);
  double* vt = (double*)malloc(sizeof(double)* n * k);
  dlra(epi, m, n, k, a, u, m, vt, n, &rnk, arow);
  *rank = rnk;

  double* r = (double*)malloc(sizeof(double) * rnk * rnk);
  double* q = (double*)malloc(sizeof(double) * rnk * rnk);
  for (int64_t i = 0; i < rnk; i++) {
    int64_t ymax = arow[i];
    int64_t xp = ymax / m;
    int64_t yp = ymax - xp * m;
    arow[i] = yp;
    Cdcopy(rnk, u + yp, m, r + i, rnk);
  }
  dorth('F', rnk, rnk, r, rnk, q, rnk);
  dtrsmr_right(m, rnk, r, rnk, u, m);
  Cdgemm('N', 'T', m, rnk, rnk, 1., u, m, q, rnk, 0., x, ldx);

  free(u);
  free(vt);
  free(r);
  free(q);
}

void dorth(char ecoq, int64_t m, int64_t n, double* r, int64_t ldr, double* q, int64_t ldq) {
  int64_t k = m < n ? m : n;
  double* TAU = (double*)malloc(sizeof(double) * k);
  int64_t nq = (ecoq == 'F' || ecoq == 'f') ? m : k;

#ifdef CBLAS
  LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, r, ldr, TAU);
  for (int64_t i = 0; i < k; i++)
    cblas_dcopy(m, &r[i * ldr], 1, &q[i * ldq], 1);
  LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, nq, k, q, ldq, TAU);
#else
  for (int64_t x = 0; x < k; x++) {
    double nrmx = 0.;
    for (int64_t y = x; y < m; y++) {
      double e = r[y + x * ldr];
      nrmx = nrmx + e * e;
    }
    nrmx = sqrt(nrmx);

    double rx = r[x + x * ldr];
    double s = rx > 0 ? -1. : 1.;
    double u1 = rx - s * nrmx;
    double tau = -s * u1 / nrmx;
    TAU[x] = tau;

    double iu1 = 1. / u1;
    for (int64_t y = x + 1; y < m; y++)
      r[y + x * ldr] = r[y + x * ldr] * iu1;
    r[x + x * ldr] = s * nrmx;

    for (int64_t xx = x + 1; xx < n; xx++) {
      double wdr = 0.;
      for (int64_t y = x; y < m; y++) {
        double e1 = y == x ? 1. : r[y + x * ldr];
        double e2 = r[y + xx * ldr];
        wdr = wdr + e1 * e2;
      }

      wdr = wdr * tau;
      for (int64_t y = x; y < m; y++) {
        double e1 = y == x ? 1. : r[y + x * ldr];
        double e2 = r[y + xx * ldr];
        r[y + xx * ldr] = e2 - e1 * wdr;
      }
    }
  }


  for (int64_t x = 0; x < nq; x++) {
    for (int64_t y = 0; y < m; y++)
      q[y + x * ldq] = 0.;
    q[x + x * ldq] = 1.;
  }

  for (int64_t kk = k - 1; kk >= 0; kk--) {
    for (int64_t x = 0; x < nq; x++) {
      double wdq = 0.;
      for (int64_t y = kk; y < m; y++) {
        double e1 = y == kk ? 1. : r[y + kk * ldr];
        double e2 = q[y + x * ldq];
        wdq = wdq + e1 * e2;
      }

      wdq = wdq * TAU[kk];
      for (int64_t y = kk; y < m; y++) {
        double e1 = y == kk ? 1. : r[y + kk * ldr];
        double e2 = q[y + x * ldq];
        q[y + x * ldq] = e2 - e1 * wdq;
      }
    }
  }
#endif
  free(TAU);
  FLOPS = FLOPS + 2 * m * n * n;
}

void Cdpotrf(int64_t n, double* a, int64_t lda) {
#ifdef CBLAS
  LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, a, lda);
#else
  for (int64_t i = 0; i < n; i++) {
    double p = a[i + i * lda];
    if (p <= 0.)
    { fprintf(stderr, "A is not positive-definite.\n"); return; }
    p = sqrt(p);
    a[i + i * lda] = p;
    double invp = 1. / p;

    for (int64_t j = i + 1; j < n; j++)
      a[j + i * lda] = a[j + i * lda] * invp;
    
    for (int64_t k = i + 1; k < n; k++) {
      double c = a[k + i * lda];
      a[k + k * lda] = a[k + k * lda] - c * c;
      for (int64_t j = k + 1; j < n; j++) {
        double r = a[j + i * lda];
        a[j + k * lda] = a[j + k * lda] - r * c;
      }
    }
  }
#endif
  FLOPS = FLOPS + n * n * n / 3;
}

void dtrsmlt_right(int64_t m, int64_t n, const double* a, int64_t lda, double* b, int64_t ldb) {
#ifdef CBLAS
  cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, m, n, 1., a, lda, b, ldb);
#else
  for (int64_t i = 0; i < n; i++) {
    double p = a[i + i * lda];
    double invp = 1. / p;

    for (int64_t j = 0; j < m; j++)
      b[j + i * ldb] = b[j + i * ldb] * invp;
    
    for (int64_t k = i + 1; k < n; k++) {
      double c = a[k + i * lda];
      for (int64_t j = 0; j < m; j++) {
        double r = b[j + i * ldb];
        b[j + k * ldb] = b[j + k * ldb] - r * c;
      }
    }
  }
#endif
  FLOPS = FLOPS + n * m * n / 3;
}

void dtrsmr_right(int64_t m, int64_t n, const double* a, int64_t lda, double* b, int64_t ldb) {
#ifdef CBLAS
  cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, m, n, 1., a, lda, b, ldb);
#else
  for (int64_t i = 0; i < n; i++) {
    double p = a[i + i * lda];
    double invp = 1. / p;

    for (int64_t j = 0; j < m; j++)
      b[j + i * ldb] = b[j + i * ldb] * invp;

    for (int64_t k = i + 1; k < n; k++) {
      double c = a[i + k * lda];
      for (int64_t j = 0; j < m; j++) {
        double r = b[j + i * ldb];
        b[j + k * ldb] = b[j + k * ldb] - r * c;
      }
    }
  }
#endif
  FLOPS = FLOPS + n * m * n / 3;
}

void dtrsml_left(int64_t m, int64_t n, const double* a, int64_t lda, double* b, int64_t ldb) {
#ifdef CBLAS
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, m, n, 1., a, lda, b, ldb);
#else
  for (int64_t i = 0; i < m; i++) {
    double p = a[i + i * lda];
    double invp = 1. / p;

    for (int64_t j = 0; j < n; j++)
      b[i + j * ldb] = b[i + j * ldb] * invp;
    
    for (int64_t k = i + 1; k < m; k++) {
      double r = a[k + i * lda];
      for (int64_t j = 0; j < n; j++) {
        double c = b[i + j * ldb];
        b[k + j * ldb] = b[k + j * ldb] - r * c;
      }
    }
  }
#endif
  FLOPS = FLOPS + m * n * m / 3;
}

void dtrsmlt_left(int64_t m, int64_t n, const double* a, int64_t lda, double* b, int64_t ldb) {
#ifdef CBLAS
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit, m, n, 1., a, lda, b, ldb);
#else
  for (int64_t i = m - 1; i >= 0; i--) {
    double p = a[i + i * lda];
    double invp = 1. / p;

    for (int64_t j = 0; j < n; j++)
      b[i + j * ldb] = b[i + j * ldb] * invp;
    
    for (int64_t k = 0; k < i; k++) {
      double r = a[i + k * lda];
      for (int64_t j = 0; j < n; j++) {
        double c = b[i + j * ldb];
        b[k + j * ldb] = b[k + j * ldb] - r * c;
      }
    }
  }
#endif
  FLOPS = FLOPS + m * n * m / 3;
}

void dtrsmr_left(int64_t m, int64_t n, const double* a, int64_t lda, double* b, int64_t ldb) {
#ifdef CBLAS
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, m, n, 1., a, lda, b, ldb);
#else
  for (int64_t i = m - 1; i >= 0; i--) {
    double p = a[i + i * lda];
    double invp = 1. / p;

    for (int64_t j = 0; j < n; j++)
      b[i + j * ldb] = b[i + j * ldb] * invp;

    for (int64_t k = 0; k < i; k++) {
      double r = a[k + i * lda];
      for (int64_t j = 0; j < n; j++) {
        double c = b[i + j * ldb];
        b[k + j * ldb] = b[k + j * ldb] - r * c;
      }
    }
  }
#endif
  FLOPS = FLOPS + m * n * m / 3;
}

void Cdgemv(char ta, int64_t m, int64_t n, double alpha, const double* a, int64_t lda, const double* x, int64_t incx, double beta, double* y, int64_t incy) {
#ifdef CBLAS
  if (ta == 'T' || ta == 't')
    cblas_dgemv(CblasColMajor, CblasTrans, m, n, alpha, a, lda, x, incx, beta, y, incy);
  else if (ta == 'N' || ta == 'n')
    cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, alpha, a, lda, x, incx, beta, y, incy);
#else
  if (ta == 'T' || ta == 't') {
    int64_t lenx = m;
    int64_t leny = n;

    for (int64_t i = 0; i < leny; i++) {
      double e = 0.;
      if (beta == 1.)
        e = y[i * incy];
      else if (beta != 0.)
        e = beta * y[i * incy];
      
      double s = 0.;
      for (int64_t j = 0; j < lenx; j++) {
        double aji = a[j + i * lda];
        double xj = x[j * incx];
        s = s + aji * xj;
      }

      y[i * incy] = e + s * alpha;
    }
  }
  else if (ta == 'N' || ta == 'n') {
    int64_t lenx = n;
    int64_t leny = m;

    for (int64_t i = 0; i < leny; i++) {
      double e = 0.;
      if (beta == 1.)
        e = y[i * incy];
      else if (beta != 0.)
        e = beta * y[i * incy];
      
      double s = 0.;
      for (int64_t j = 0; j < lenx; j++) {
        double aij = a[i + j * lda];
        double xj = x[j * incx];
        s = s + aij * xj;
      }

      y[i * incy] = e + s * alpha;
    }
  }
#endif
  FLOPS = FLOPS + 2 * n * m;
}

void Cdgemm(char ta, char tb, int64_t m, int64_t n, int64_t k, double alpha, const double* a, int64_t lda, const double* b, int64_t ldb, double beta, double* c, int64_t ldc) {
#ifdef CBLAS
  if (ta == 'T' || ta == 't') {
    if (tb == 'T' || tb == 't')
      cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    else if (tb == 'N' || tb == 'n')
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
  else if (ta == 'N' || ta == 'n') {
    if (tb == 'T' || tb == 't')
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    else if (tb == 'N' || tb == 'n')
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
#else
  int64_t ma = k;
  int64_t na = m;
  if (ta == 'N' || ta == 'n') {
    ma = m;
    na = k;
  }

  if (tb == 'T' || tb == 't')
    for (int64_t i = 0; i < n; i++)
      Cdgemv(ta, ma, na, alpha, a, lda, b + i, ldb, beta, c + i * ldc, 1);
  else if (tb == 'N' || tb == 'n')
    for (int64_t i = 0; i < n; i++)
      Cdgemv(ta, ma, na, alpha, a, lda, b + i * ldb, 1, beta, c + i * ldc, 1);
#endif
}

void Cdcopy(int64_t n, const double* x, int64_t incx, double* y, int64_t incy) {
#ifdef CBLAS
  cblas_dcopy(n, x, incx, y, incy);
#else
  for (int64_t i = 0; i < n; i++)
    y[i * incy] = x[i * incx];
#endif
}

void Cdscal(int64_t n, double alpha, double* x, int64_t incx) {
#ifdef CBLAS
  cblas_dscal(n, alpha, x, incx);
#else
  if (alpha == 0.)
    for (int64_t i = 0; i < n; i++)
      x[i * incx] = 0.;
  else if (alpha != 1.)
    for (int64_t i = 0; i < n; i++)
      x[i * incx] = alpha * x[i * incx];
#endif
  FLOPS = FLOPS + n;
}

void Cdaxpy(int64_t n, double alpha, const double* x, int64_t incx, double* y, int64_t incy) {
#ifdef CBLAS
  cblas_daxpy(n, alpha, x, incx, y, incy);
#else
  for (int64_t i = 0; i < n; i++)
    y[i * incy] = y[i * incy] + alpha * x[i * incx];
#endif
  FLOPS = FLOPS + 2 * n;
}

void Cddot(int64_t n, const double* x, int64_t incx, const double* y, int64_t incy, double* result) {
#ifdef CBLAS
  *result = cblas_ddot(n, x, incx, y, incy);
#else
  double s = 0.;
  for (int64_t i = 0; i < n; i++)
    s = s + y[i * incy] * x[i * incx];
  *result = s;
#endif
  FLOPS = FLOPS + 2 * n;
}

void Cidamax(int64_t n, const double* x, int64_t incx, int64_t* ida) {
#ifdef CBLAS
  *ida = cblas_idamax(n, x, incx);
#else
  if (n > 0) {
    double amax = x[0];
    int64_t ymax = 0;
    for (int64_t i = 1; i < n; i++) {
      double fa = fabs(x[i * incx]);
      if (fa > amax) {
        amax = fa;
        ymax = i;
      }
    }
    *ida = ymax;
  }
#endif
  FLOPS = FLOPS + n;
}

void Cdnrm2(int64_t n, const double* x, int64_t incx, double* nrm_out) {
#ifdef CBLAS
  *nrm_out = cblas_dnrm2(n, x, incx);
#else
  double nrm = 0.;
  for (int64_t i = 0; i < n; i++) {
    double e = x[i * incx];
    nrm = nrm + e * e;
  }
  nrm = sqrt(nrm);
  *nrm_out = nrm;
#endif
  FLOPS = FLOPS + 2 * n;
}

int64_t* getFLOPS() {
  return &FLOPS;
}
