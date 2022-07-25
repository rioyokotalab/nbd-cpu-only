
#include "nbd.h"

#include "mkl.h"

void cpyMatToMat(int64_t m, int64_t n, const struct Matrix* m1, struct Matrix* m2, int64_t y1, int64_t x1, int64_t y2, int64_t x2) {
  MKL_Domatcopy('C', 'N', m, n, 1., &m1->A[y1 + x1 * m1->M], m1->M, &m2->A[y2 + x2 * m2->M], m2->M);
}

void qr_full(struct Matrix* Q, struct Matrix* R, double* tau) {
  LAPACKE_dgeqrf(LAPACK_COL_MAJOR, Q->M, R->N, Q->A, Q->M, tau);
  MKL_Domatcopy('C', 'N', R->M, R->N, 1., Q->A, Q->M, R->A, R->M);
  LAPACKE_dorgqr(LAPACK_COL_MAJOR, Q->M, Q->N, R->N, Q->A, Q->M, tau);
}

void svd_U(struct Matrix* A, struct Matrix* U, double* S) {
  int64_t rank_a = A->M < A->N ? A->M : A->N;
  int64_t lda = 1 < A->M ? A->M : 1;
  int64_t ldu = 1 < U->M ? U->M : 1;
  int64_t ldv = 1 < A->N ? A->N : 1;
  LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'N', A->M, A->N, A->A, lda, S, U->A, ldu, NULL, ldv, &S[rank_a]);
}

void mul_AS(struct Matrix* C, const struct Matrix* A, const double S[]) {
  char r = 'R';
  int32_t m = C->M, n = C->N, one = 1;
  int32_t lda = 1 < A->M ? A->M : 1;
  int32_t ldc = 1 < C->M ? C->M : 1;
  const double* a = A->A, *x = S;
  double* c = C->A;
  ddgmm_batch(&r, &m, &n, &a, &lda, &x, &one, &c, &ldc, &one, &one);
}

void id_row(struct Matrix* A, struct Matrix* U, int32_t arows[]) {
  int32_t lda = 1 < A->M ? A->M : 1;
  int32_t ldu = 1 < U->M ? U->M : 1;
  MKL_Domatcopy('C', 'N', A->M, A->N, 1., A->A, lda, U->A, ldu);
  LAPACKE_dgetrf(LAPACK_COL_MAJOR, A->M, A->N, A->A, lda, arows);
  cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, A->M, A->N, 1., A->A, lda, U->A, ldu);
  cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit, A->M, A->N, 1., A->A, lda, U->A, ldu);
}

void mmult(char ta, char tb, const struct Matrix* A, const struct Matrix* B, struct Matrix* C, double alpha, double beta) {
  int64_t k = (ta == 'N' || ta == 'n') ? A->N : A->M;
  CBLAS_TRANSPOSE tac = (ta == 'T' || ta == 't') ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE tbc = (tb == 'T' || tb == 't') ? CblasTrans : CblasNoTrans;
  int64_t lda = 1 < A->M ? A->M : 1;
  int64_t ldb = 1 < B->M ? B->M : 1;
  int64_t ldc = 1 < C->M ? C->M : 1;
  cblas_dgemm(CblasColMajor, tac, tbc, C->M, C->N, k, alpha, A->A, lda, B->A, ldb, beta, C->A, ldc);
}

void chol_decomp(struct Matrix* A) {
  int64_t lda = 1 < A->M ? A->M : 1;
  LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', A->M, A->A, lda);
}

void trsm_lowerA(struct Matrix* A, const struct Matrix* L) {
  int64_t lda = 1 < A->M ? A->M : 1;
  int64_t ldl = 1 < L->M ? L->M : 1;
  cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, A->M, A->N, 1., L->A, ldl, A->A, lda);
}

void upper_tri_reflec_mult(char side, const struct Matrix* R, struct Matrix* A) {
  int64_t ldr = 1 < R->M ? R->M : 1;
  int64_t lda = 1 < A->M ? A->M : 1;
  if (side == 'L' || side == 'l')
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, A->M, A->N, 1., R->A, ldr, A->A, lda);
  else if (side == 'R' || side == 'r')
    cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit, A->M, A->N, 1., R->A, ldr, A->A, lda);
}

void mat_solve(char type, struct Matrix* X, const struct Matrix* A) {
  int64_t lda = 1 < A->M ? A->M : 1;
  int64_t ldx = 1 < X->M ? X->M : 1;
  if (type == 'F' || type == 'f' || type == 'A' || type == 'a')
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, X->M, X->N, 1., A->A, lda, X->A, ldx);
  if (type == 'B' || type == 'b' || type == 'A' || type == 'a')
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit, X->M, X->N, 1., A->A, lda, X->A, ldx);
}

void nrm2_A(struct Matrix* A, double* nrm) {
  int64_t len_A = A->M * A->N;
  double nrm_A = cblas_dnrm2(len_A, A->A, 1);
  *nrm = nrm_A;
}

void scal_A(struct Matrix* A, double alpha) {
  int64_t len_A = A->M * A->N;
  cblas_dscal(len_A, alpha, A->A, 1);
}
