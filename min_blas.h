#pragma once

#include <cstdint>

namespace nbd {

  void drotg(double& a, double& b, double& c, double& s);

  void drot(int64_t n, double* x, int64_t incx, double* y, int64_t incy, double c, double s);

  void dztocs(double z, double& c, double& s);

  void dgivens(int64_t m, int64_t n, double* a, int64_t lda);

  void dgivensm(int side, int trans, int64_t m, int64_t n, int64_t k, const double* a, int64_t lda, double* c, int64_t ldc);

  void dgivensq(int64_t m, int64_t n, int64_t k, double* a, int64_t lda);

}