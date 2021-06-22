
#pragma once
#include "nbd.h"

namespace nbd {

  void dsvd(eval_func_t r2f, const Cell* ci, const Cell* cj, int dim, double* s, double* u, int ldu, double* v, int ldv, int* info = nullptr);

  void dsvd(int m, int n, double* a, int lda, double* s, double* u, int ldu, double* v, int ldv, int* info = nullptr);

  void dlr_svd(int m, int n, int r, double* u, int ldu, double* v, int ldv, double* s, int* info = nullptr);

  void dtsvd_aca(int m, int n, int r, const double* a, int lda, double* s, double* u, int ldu, double* v, int ldv, int* info = nullptr);

}