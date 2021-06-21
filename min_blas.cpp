
#include "min_blas.h"

#include <cstdio>
#include <algorithm>
#include <cmath>
#include <vector>

using namespace nbd;

void nbd::drotg(double& a, double& b, double& c, double& s) {
  double scale = std::fabs(a) + std::fabs(b), r, z;
  if (scale == 0.) {
    c = 1.;
    s = 0.;
    r = 0.;
    z = 0.;
  }
  else {
    double roe = b;
    if (std::fabs(a) > std::fabs(b)) roe = a;
    r = scale * std::hypot(a / scale, b / scale);
    r = std::copysign(r, roe);
    c = a / r;
    s = b / r;
    z = 1.;
    if (std::fabs(a) > std::fabs(b)) z = s;
    if (std::fabs(b) >= std::fabs(a) && c != 0.) z = 1. / c;
  }
  a = r;
  b = z;
}

void nbd::drot(int64_t n, double* x, int64_t incx, double* y, int64_t incy, double c, double s) {
  for (int64_t i = 0; i < n; i++) {
    double a = x[i * incx], b = y[i * incy];
    x[i * incx] = c * a + s * b;
    y[i * incy] = -s * a + c * b;
  }
}

void nbd::dztocs(double z, double& c, double& s) {
  if (z == 1) {
    c = 0.;
    s = 1.;
  }
  else if (std::fabs(z) < 1) {
    c = std::sqrt(1 - z * z);
    s = z;
  }
  else {
    c = 1 / z;
    s = std::sqrt(1 - c * c);
  }
}

void nbd::dgivens(int64_t m, int64_t n, double* a, int64_t lda) {
  for (int64_t i = 0; i < n; i++) {
    for (int64_t j = i + 1; j < m; j++) {
      double c, s;
      drotg(a[i * lda + i], a[i * lda + j], c, s);
      drot(n - i - 1, a + i * lda + i + lda, lda, a + i * lda + j + lda, lda, c, s);
    }
  }
}

void nbd::dgivensm(int side, int trans, int64_t m, int64_t n, int64_t k, const double* a, int64_t lda, double* c, int64_t ldc) {
  
  int64_t inc = 1;
  if (!side) {
    inc = ldc;
    ldc = 1;
    int64_t t = n;
    n = m;
    m = t;
    trans = !trans;
  }

  if (trans)
    for (int64_t i = k - 1; i >= 0; i--) {
      for (int64_t j = n - 1; j > i; j--) {
        double c_, s_;
        dztocs(a[i * lda + j], c_, s_);
        drot(m, c + i * ldc, inc, c + j * ldc, inc, c_, -s_);
      }
    }
  else
    for (int64_t i = 0; i < k; i++) {
      for (int64_t j = i + 1; j < n; j++) {
        double c_, s_;
        dztocs(a[i * lda + j], c_, s_);
        drot(m, c + i * ldc, inc, c + j * ldc, inc, c_, s_);
      }
    }
}

void nbd::dgivensq(int64_t m, int64_t n, int64_t k, double* a, int64_t lda) {

  std::vector<double> work(m * k);
  for (int64_t i = 0; i < k; i++) {
    for (int64_t j = i + 1; j < m; j++) {
      work[i * m + j] = a[i * lda + j];
    }
  }

  for (int64_t i = 0; i < n; i++) {
    for (int64_t j = 0; j < m; j++) {
      a[i * lda + j] = (double)(i == j);
    }
  }

  dgivensm(0, 0, m, n, k, work.data(), m, a, lda);

}
