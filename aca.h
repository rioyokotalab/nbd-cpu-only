
#pragma once
#include "nbd.h"

namespace nbd {

#define ACA_USE_NORM
  constexpr double ACA_EPI = 1.e-13;

  void daca_cells(eval_func_t r2f, const Cell* ci, const Cell* cj, int dim, int max_iters, double* u, int ldu, double* v, int ldv, int* info = nullptr);

  void daca(int m, int n, int max_iters, const double* a, int lda, double* u, int ldu, double* v, int ldv, int* info = nullptr);

}