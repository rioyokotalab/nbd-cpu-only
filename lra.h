
#pragma once
#include "nbd.h"

namespace nbd {

  void daca_cells(EvalFunc ef, const Cell* ci, const Cell* cj, int dim, int max_iters, double* u, int ldu, double* v, int ldv, int* info = nullptr);

  void daca(int m, int n, int max_iters, const double* a, int lda, double* u, int ldu, double* v, int ldv, int* info = nullptr);

  void did(int m, int n, int max_iters, const double* a, int lda, int* aj, double* x, int ldx, int* info = nullptr);

  /* Dense random sampling for left side */
  void ddspl(int m, int na, const double* a, int lda, int ns, double* s, int lds);

  /* LR random sampling for left side */
  void drspl(int m, int na, int r, const double* ua, int ldu, const double* va, int ldv, int ns, double* s, int lds);

  void dorth(int m, int n, double* a, int lda, double* r, int ldr);

}
