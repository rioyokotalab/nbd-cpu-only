
#pragma once

#include "linalg.hxx"

namespace nbd {

#define DIM_MAX 3

  struct Body {
    double X[DIM_MAX];
    double B;
  };

  typedef void (*eval_func_t) (double*);

  void laplace3d(double* r2);

  void yukawa3d(double* r2);

  void set_kernel_constants(double singularity, double alpha);

  void eval(eval_func_t ef, const Body* bi, const Body* bj, int64_t dim, double* out);

  void P2Pmat(eval_func_t ef, int64_t m, int64_t n, const Body bi[], const Body bj[], int64_t dim, Matrix& a, const int64_t sel_i[], const int64_t sel_j[]);

}
