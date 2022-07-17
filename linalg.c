

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
  mkl_domatcopy('C', 'N', m, n, 1., &(m1->A)[y1 + x1 * m1->M], m1->M, &(m2->A)[y2 + x2 * m2->M], m2->M);
#else
  for (int64_t j = 0; j < n; j++) {
    int64_t j1 = y1 + (x1 + j) * m1->M;
    int64_t j2 = y2 + (x2 + j) * m2->M;
    memcpy(&(m2->A)[j2], &(m1->A)[j1], sizeof(double) * m);
  }
#endif
}

void qr_full(struct Matrix* Q, struct Matrix* R) {
  int64_t m = Q->M;
  int64_t n = R->N;
  double* tau = (double*)calloc(n, sizeof(double));
  LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, Q->A, m, tau);
  cpyMatToMat(n, n, Q, R, 0, 0, 0, 0);
  LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, m, n, Q->A, m, tau);
  free(tau);
}

void updateSubU(struct Matrix* U, const struct Matrix* R1, const struct Matrix* R2) {
  int64_t m1 = R1->N;
  int64_t m2 = R2->N;
  int64_t n = U->N;
  cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, m1, n, 1., R1->A, R1->M, U->A, U->M);
  cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, m2, n, 1., R2->A, R2->M, &U->A[m1], U->M);
}

void lraID(double epi, struct Matrix* A, struct Matrix* U, int32_t arows[], int64_t* rnk_out) {
  int64_t rank_a = A->M < A->N ? A->M : A->N;
  int64_t rank = rank_a;
  if (*rnk_out > 0)
    rank = *rnk_out < rank ? *rnk_out : rank;

  double* work = (double*)malloc(sizeof(double) * (rank_a + rank_a + 1));
  LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'N', A->M, A->N, A->A, A->M, work, U->A, A->M, NULL, A->N, &work[rank_a]);
  if (epi > 0.) {
    int64_t r = 0;
    double sepi = work[0] * epi;
    while(r < rank && work[r] > sepi)
      r += 1;
    rank = r;
  }

  for (int64_t i = 0; i < rank; i++)
    cblas_dscal(A->M, work[i], &(U->A)[i * A->M], 1);
  memcpy(A->A, U->A, sizeof(double) * A->M * rank);

  LAPACKE_dgetrf(LAPACK_COL_MAJOR, A->M, rank, A->A, A->M, arows);
  cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, A->M, rank, 1., A->A, A->M, U->A, A->M);
  cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit, A->M, rank, 1., A->A, A->M, U->A, A->M);
  free(work);
  *rnk_out = rank;
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

void normalizeA(struct Matrix* A, const struct Matrix* B) {
  int64_t len_A = A->M * A->N;
  int64_t len_B = B->M * B->N;
  double nrm_A = cblas_dnrm2(len_A, A->A, 1);
  double nrm_B = cblas_dnrm2(len_B, B->A, 1);
  cblas_dscal(len_A, nrm_B / nrm_A, A->A, 1);
}

