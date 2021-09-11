
#pragma once
#include "nbd.h"

namespace nbd {

  constexpr bool ACA_USE_NORM = true;
  constexpr double ACA_EPI = 1.e-7;

  void daca_cells(EvalFunc ef, const Cell* ci, const Cell* cj, int dim, int max_iters, double* u, int ldu, double* v, int ldv, int* info = nullptr);

  void daca(int m, int n, int max_iters, const double* a, int lda, double* u, int ldu, double* v, int ldv, int* info = nullptr);

  void did(int m, int n, int max_iters, const double* a, int lda, int* aj, double* x, int ldx, int* info = nullptr);

  /* Dense random sampling for left side */
  void ddspl(int m, int na, const double* a, int lda, int ns, double* s, int lds);

  /* LR random sampling for left side */
  void drspl(int m, int na, int r, const double* ua, int ldu, const double* va, int ldv, int ns, double* s, int lds);

  void dorth(int m, int n, double* a, int lda, double* r, int ldr);

  /* C(k x n) = UT(m x k) * A(m x n) */
  void dmul_ut(int m, int n, int k, const double* u, int ldu, const double* a, int lda, double* c, int ldc);

  /* S(m x n) = U(m x k) * VT(n x k) */
  void dmul_s(int m, int n, int k, const double* u, int ldu, const double* v, int ldv, double* s, int lds);

  void dmatcpy(int m, int n, const double* a, int lda, double* b, int ldb);

  /* B(k x l) = U^-1(m x k) * A(m x n) * VT^-1(n x l) */
  void dmul2_pinv(int m, int n, int k, int l, const double* a, int lda, const double* u, int ldu, const double* v, int ldv, double* b, int ldb);

}
