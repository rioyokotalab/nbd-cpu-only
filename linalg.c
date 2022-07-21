

#include "nbd.h"

#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "string.h"
#include "inttypes.h"

#ifdef USE_MKL
#include "mkl.h"
#else
#include "cblas.h"
#include "lapacke.h"
#endif

void cpyMatToMat(int64_t m, int64_t n, const struct Matrix* m1, struct Matrix* m2, int64_t y1, int64_t x1, int64_t y2, int64_t x2) {
#ifdef USE_MKL
  mkl_domatcopy('C', 'N', m, n, 1., &m1->A[y1 + x1 * m1->M], m1->M, &m2->A[y2 + x2 * m2->M], m2->M);
#else
  if (m == m1->M && m == m2->M)
    memcpy(&m2->A[y1 + x1 * m], &m1->A[y2 + x2 * m], sizeof(double) * m * n);
  else for (int64_t j = 0; j < n; j++) {
    int64_t j1 = y1 + (x1 + j) * m1->M;
    int64_t j2 = y2 + (x2 + j) * m2->M;
    memcpy(&m2->A[j2], &m1->A[j1], sizeof(double) * m);
  }
#endif
}

void qr_full(struct Matrix* Q, struct Matrix* R, double* tau) {
  LAPACKE_dgeqrf(LAPACK_COL_MAJOR, Q->M, R->N, Q->A, Q->M, tau);
  cpyMatToMat(R->M, R->N, Q, R, 0, 0, 0, 0);
  LAPACKE_dorgqr(LAPACK_COL_MAJOR, Q->M, Q->N, R->N, Q->A, Q->M, tau);
}

void updateSubU(struct Matrix* U, const struct Matrix* R1, const struct Matrix* R2) {
  cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, R1->N, U->N, 1., R1->A, R1->M, U->A, U->M);
  cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, R2->N, U->N, 1., R2->A, R2->M, &U->A[R1->N], U->M);
}

void svd_U(struct Matrix* A, struct Matrix* U, double* S) {
  int64_t rank_a = A->M < A->N ? A->M : A->N;
  LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'N', A->M, A->N, A->A, A->M, S, U->A, A->M, NULL, A->N, &S[rank_a]);
}

void mul_AS(struct Matrix* A, double* S) {
  for (int64_t i = 0; i < A->N; i++)
    cblas_dscal(A->M, S[i], &(A->A)[i * A->M], 1);
}

void id_row(struct Matrix* U, int32_t arows[], double* work) {
  struct Matrix A = (struct Matrix){ work, U->M, U->N };
  cblas_dcopy(A.M * A.N, U->A, 1, A.A, 1);
  LAPACKE_dgetrf(LAPACK_COL_MAJOR, A.M, A.N, A.A, A.M, arows);
  cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, A.M, A.N, 1., A.A, A.M, U->A, A.M);
  cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit, A.M, A.N, 1., A.A, A.M, U->A, A.M);
}

void mmult(char ta, char tb, const struct Matrix* A, const struct Matrix* B, struct Matrix* C, double alpha, double beta) {
  int64_t k = (ta == 'N' || ta == 'n') ? A->N : A->M;
  CBLAS_TRANSPOSE tac = (ta == 'T' || ta == 't') ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE tbc = (tb == 'T' || tb == 't') ? CblasTrans : CblasNoTrans;
  cblas_dgemm(CblasColMajor, tac, tbc, C->M, C->N, k, alpha, A->A, A->M, B->A, B->M, beta, C->A, C->M);
}

void chol_decomp(struct Matrix* A) {
  LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', A->M, A->A, A->M);
}

void trsm_lowerA(struct Matrix* A, const struct Matrix* L) {
  cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, A->M, A->N, 1., L->A, L->M, A->A, A->M);
}

void rsr(const struct Matrix* R1, const struct Matrix* R2, struct Matrix* S) {
  cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, R1->N, S->N, 1., R1->A, R1->M, S->A, S->M);
  cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit, S->M, R2->N, 1., R2->A, R2->M, S->A, S->M);
}

void mat_solve(char type, struct Matrix* X, const struct Matrix* A) {
  if (type == 'F' || type == 'f' || type == 'A' || type == 'a')
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, X->M, X->N, 1., A->A, A->M, X->A, X->M);
  if (type == 'B' || type == 'b' || type == 'A' || type == 'a')
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit, X->M, X->N, 1., A->A, A->M, X->A, X->M);
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
