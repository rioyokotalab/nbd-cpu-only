
#include "minblas.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int64_t FLOPS = 0;

void dlra(double epi, int64_t m, int64_t n, int64_t k, double* a, int64_t lda, double* u, int64_t ldu, double* vt, int64_t ldvt, int64_t* rank, int64_t* piv) {
  double nrm = 0.;
  double epi2 = epi * epi;
  double n2 = 1.;
  double amax = 1.;

  int64_t i = 0;
  while (i < k && (n2 > epi2 * nrm) && amax > 0) {
    amax = 0.;
    int64_t ymax = 0;
    for (int64_t x = 0; x < n; x++) {
      int64_t ybegin = x * lda;
      int64_t yend = m + x * lda;
      for (int64_t y = ybegin; y < yend; y++) {
        double fa = fabs(a[y]);
        if (fa > amax) {
          amax = fa;
          ymax = y;
        }
      }
    }

    if (amax > 0.) {
      if (piv != NULL)
        piv[i] = ymax;
      int64_t xp = ymax / lda;
      int64_t yp = ymax - xp * lda;
      double ap = 1. / a[ymax];
      double* ui = u + i * ldu;
      double* vi = vt + i * ldvt;

      for (int64_t x = 0; x < n; x++) {
        double ax = a[yp + x * lda];
        vi[x] = ax;
        a[yp + x * lda] = 0.;
      }

      for (int64_t y = 0; y < m; y++) {
        double ay = a[y + xp * lda];
        ui[y] = ay * ap;
        a[y + xp * lda] = 0.;
      }
      ui[yp] = 1.;

      for (int64_t x = 0; x < n; x++) {
        if (x == xp)
          continue;
        double ri = vi[x];
        for (int64_t y = 0; y < m; y++) {
          if (y == yp)
            continue;
          double lf = ui[y];
          double e = a[y + x * lda];
          e = e - lf * ri;
          a[y + x * lda] = e;
        }
      }

      if (epi2 > 0.) {
        double nrm_v = 0.;
        double nrm_vi = 0.;
        for (int64_t x = 0; x < n; x++) {
          double vx = vi[x];
          nrm_vi = nrm_vi + vx * vx;
          for (int64_t j = 0; j < i; j++) {
            double vj = vt[x + j * ldvt];
            nrm_v = nrm_v + vx * vj;
          }
        }

        double nrm_u = 0.;
        double nrm_ui = 0.;
        for (int64_t y = 0; y < m; y++) {
          double uy = ui[y];
          nrm_ui = nrm_ui + uy * uy;
          for (int64_t j = 0; j < i; j++) {
            double uj = u[y + j * ldu];
            nrm_u = nrm_u + uy * uj;
          }
        }

        n2 = nrm_ui * nrm_vi;
        nrm = nrm + 2. * nrm_u * nrm_v + n2;
      }
      i++;
    }
  }

  *rank = i;
  FLOPS = FLOPS + 2 * m * n * i;
  if (epi2 > 0 && i == k)
    fprintf(stderr, "LRA reached full iterations.\n");
}

void didrow(double epi, int64_t m, int64_t n, int64_t k, double* a, int64_t lda, double* x, int64_t ldx, int64_t* arow, int64_t* rank) {
  int64_t rnk;
  double* u = (double*)malloc(sizeof(double) * m * k);
  double* vt = (double*)malloc(sizeof(double)* n * k);
  dlra(epi, m, n, k, a, lda, u, m, vt, n, &rnk, arow);
  *rank = rnk;

  double* r = (double*)malloc(sizeof(double) * rnk * rnk);
  double* q = (double*)malloc(sizeof(double) * rnk * rnk);
  for (int64_t i = 0; i < rnk; i++) {
    int64_t ymax = arow[i];
    int64_t xp = ymax / lda;
    int64_t yp = ymax - xp * lda;
    arow[i] = yp;
    dcopy(rnk, u + yp, m, r + i, rnk);
  }
  dorth('F', rnk, rnk, r, rnk, q, rnk);
  dtrsmr_right(m, rnk, r, rnk, u, m);
  dgemm('N', 'T', m, rnk, rnk, 1., u, m, q, rnk, 0., x, ldx);

  free(u);
  free(vt);
  free(r);
  free(q);
}

void dorth(char ecoq, int64_t m, int64_t n, double* r, int64_t ldr, double* q, int64_t ldq) {
  int64_t k = m < n ? m : n;
  double* TAU = (double*)malloc(sizeof(double) * k);

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

  int64_t nq = (ecoq == 'F' || ecoq == 'f') ? m : k;

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

  free(TAU);
  FLOPS = FLOPS + 2 * m * n * n;
}

void dpotrf(int64_t n, double* a, int64_t lda) {
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
  FLOPS = FLOPS + n * n * n / 3;
}

void dtrsmlt_right(int64_t m, int64_t n, const double* a, int64_t lda, double* b, int64_t ldb) {
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
  FLOPS = FLOPS + n * m * n / 3;
}

void dtrsmr_right(int64_t m, int64_t n, const double* a, int64_t lda, double* b, int64_t ldb) {
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
  FLOPS = FLOPS + n * m * n / 3;
}

void dtrsml_left(int64_t m, int64_t n, const double* a, int64_t lda, double* b, int64_t ldb) {
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
  FLOPS = FLOPS + m * n * m / 3;
}

void dtrsmlt_left(int64_t m, int64_t n, const double* a, int64_t lda, double* b, int64_t ldb) {
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
  FLOPS = FLOPS + m * n * m / 3;
}

void dtrsmr_left(int64_t m, int64_t n, const double* a, int64_t lda, double* b, int64_t ldb) {
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
  FLOPS = FLOPS + m * n * m / 3;
}

void dgemv(char ta, int64_t m, int64_t n, double alpha, const double* a, int64_t lda, const double* x, int64_t incx, double beta, double* y, int64_t incy) {
  
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

  FLOPS = FLOPS + 2 * n * m;
}

void dgemm(char ta, char tb, int64_t m, int64_t n, int64_t k, double alpha, const double* a, int64_t lda, const double* b, int64_t ldb, double beta, double* c, int64_t ldc) {
  int64_t ma = k;
  int64_t na = m;
  if (ta == 'N' || ta == 'n') {
    ma = m;
    na = k;
  }

  if (tb == 'T' || tb == 't')
    for (int64_t i = 0; i < n; i++)
      dgemv(ta, ma, na, alpha, a, lda, b + i, ldb, beta, c + i * ldc, 1);
  else if (tb == 'N' || tb == 'n')
    for (int64_t i = 0; i < n; i++)
      dgemv(ta, ma, na, alpha, a, lda, b + i * ldb, 1, beta, c + i * ldc, 1);
}

void dcopy(int64_t n, const double* x, int64_t incx, double* y, int64_t incy) {
  for (int64_t i = 0; i < n; i++)
    y[i * incy] = x[i * incx];
}

void dscal(int64_t n, double alpha, double* x, int64_t incx) {
  if (alpha == 0.)
    for (int64_t i = 0; i < n; i++)
      x[i * incx] = 0.;
  else if (alpha != 1.)
    for (int64_t i = 0; i < n; i++)
      x[i * incx] = alpha * x[i * incx];
  FLOPS = FLOPS + n;
}

void daxpy(int64_t n, double alpha, const double* x, int64_t incx, double* y, int64_t incy) {
  for (int64_t i = 0; i < n; i++)
    y[i * incy] = y[i * incy] + alpha * x[i * incx];
  FLOPS = FLOPS + 2 * n;
}

void ddot(int64_t n, const double* x, int64_t incx, const double* y, int64_t incy, double* result) {
  double s = 0.;
  for (int64_t i = 0; i < n; i++)
    s = s + y[i * incy] * x[i * incx];
  *result = s;
  FLOPS = FLOPS + 2 * n;
}

void dnrm2(int64_t n, const double* x, int64_t incx, double* nrm_out) {
  double nrm = 0.;
  for (int64_t i = 0; i < n; i++) {
    double e = x[i * incx];
    nrm = nrm + e * e;
  }
  nrm = sqrt(nrm);
  *nrm_out = nrm;
  FLOPS = FLOPS + 2 * n;
}

int64_t* getFLOPS() {
  return &FLOPS;
}
