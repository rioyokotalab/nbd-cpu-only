
#pragma once
#include "nbd.h"

namespace nbd {

  /* A = U*S*VT, dsvd returns U(m, min(m, n)), S(min(m, n)), V(n, min(m, n));  */
  void dsvd(eval_func_t r2f, const Cell* ci, const Cell* cj, int dim, double* s, double* u, int ldu, double* v, int ldv, int* info = nullptr);

  /* A = U*S*VT, dsvd returns U(m, min(m, n)), S(min(m, n)), V(n, min(m, n));  */
  void dsvd(int m, int n, double* a, int lda, double* s, double* u, int ldu, double* v, int ldv, int* info = nullptr);

  /* Input low rank matrix A = U*VT, dlr_svd orthogonalize U and V and overwrites them, returns U(m, r), S(r), V(n, r);  */
  void dlr_svd(int m, int n, int r, double* u, int ldu, double* v, int ldv, double* s, int* info = nullptr);

  /* Truncate singular values that are smaller than Epi*max(m, n)*max(s); */
  void dtsvd(int m, int n, int r, double* s, int* out);

  /* Pseudo-invert a TSVD decomposed LR matrix(m x r, n x r) and multiply to the left of A'(n, na) <- LR^+ * A(m, na);  */
  void dpinvl_svd(int m, int n, int r, const double* s, const double* u, int ldu, const double* v, int ldv, double* a, int na, int lda);

  /* Pseudo-invert a TSVD decomposed LR matrix(m x r, n x r) and multiply to the right of A'(ma, m) <-  A(ma, n) * LR^+;  */
  void dpinvr_svd(int m, int n, int r, const double* s, const double* u, int ldu, const double* v, int ldv, double* a, int ma, int lda);

}