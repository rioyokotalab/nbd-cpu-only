
#include "nbd.h"

#include "mkl.h"

void mat_cpy(int64_t m, int64_t n, const struct Matrix* m1, struct Matrix* m2, int64_t y1, int64_t x1, int64_t y2, int64_t x2) {
  LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', m, n, &m1->A[y1 + x1 * m1->M], m1->LDA, &m2->A[y2 + x2 * m2->M], m2->LDA);
}

void mmult(char ta, char tb, const struct Matrix* A, const struct Matrix* B, struct Matrix* C, double alpha, double beta) {
  int64_t k = ta == 'N' ? A->N : A->M;
  CBLAS_TRANSPOSE tac = ta == 'N' ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE tbc = tb == 'N' ? CblasNoTrans : CblasTrans;
  int64_t lda = 1 < A->LDA ? A->LDA : 1;
  int64_t ldb = 1 < B->LDA ? B->LDA : 1;
  int64_t ldc = 1 < C->LDA ? C->LDA : 1;
  cblas_dgemm(CblasColMajor, tac, tbc, C->M, C->N, k, alpha, A->A, lda, B->A, ldb, beta, C->A, ldc);
}

void chol_decomp(struct Matrix* A) {
  int64_t lda = 1 < A->LDA ? A->LDA : 1;
  LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', A->M, A->A, lda);
}

void svd_U(struct Matrix* A, double* S) {
  int64_t rank_a = A->M < A->N ? A->M : A->N;
  int64_t lda = 1 < A->LDA ? A->LDA : 1;
  int64_t ldv = 1 < A->N ? A->N : 1;
  LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'O', 'N', A->M, A->N, A->A, lda, S, NULL, lda, NULL, ldv, &S[rank_a]);
}

void id_row(struct Matrix* A, int32_t arows[], double* work) {
  int64_t lda = 1 < A->LDA ? A->LDA : 1;
  LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', A->M, A->N, A->A, lda, work, lda);
  LAPACKE_dgetrf(LAPACK_COL_MAJOR, A->M, A->N, work, lda, arows);
  cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, A->M, A->N, 1., work, lda, A->A, lda);
  cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit, A->M, A->N, 1., work, lda, A->A, lda);
}

void upper_tri_reflec_mult(char side, int64_t lenR, const struct Matrix* R, struct Matrix* A) {
  int64_t lda = 1 < A->LDA ? A->LDA : 1;
  int64_t y = 0;
  if (side == 'L' || side == 'l')
    for (int64_t i = 0; i < lenR; i++) {
      int64_t ldr = 1 < R[i].LDA ? R[i].LDA : 1;
      cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, R[i].M, A->N, 1., R[i].A, ldr, &A->A[y], lda);
      y = y + R[i].M;
    }
  else if (side == 'R' || side == 'r')
    for (int64_t i = 0; i < lenR; i++) {
      int64_t ldr = 1 < R[i].LDA ? R[i].LDA : 1;
      cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit, A->M, R[i].M, 1., R[i].A, ldr, &A->A[y], lda);
      y = y + A->LDA * R[i].M;
    }
}

void qr_full(struct Matrix* Q, struct Matrix* R) {
  int64_t ldq = 1 < Q->LDA ? Q->LDA : 1;
  int64_t k = R->N;
  int64_t ldr = 1 < R->LDA ? R->LDA : 1;
  LAPACKE_dgeqrf(LAPACK_COL_MAJOR, Q->M, k, Q->A, ldq, R->A);
  double a0 = Q->N > 0 ? Q->A[0] : 0.;
  LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', R->M, R->N - 1, &Q->A[ldq], ldq, &R->A[ldr], ldr);
  LAPACKE_dorgqr(LAPACK_COL_MAJOR, Q->M, Q->N, k, Q->A, ldq, R->A);
  R->A[0] = a0;
}

void mat_solve(char type, struct Matrix* X, const struct Matrix* A) {
  int64_t lda = 1 < A->LDA ? A->LDA : 1;
  int64_t ldx = 1 < X->LDA ? X->LDA : 1;
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
