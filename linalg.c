
#include "nbd.h"

#include "mkl.h"

#ifndef BATCH_LEN
#define BATCH_LEN 12000
#endif

struct MatrixCopyBatch {
  int32_t rows_array[BATCH_LEN], cols_array[BATCH_LEN], lda_array[BATCH_LEN], ldb_array[BATCH_LEN];
  const double* A_array[BATCH_LEN];
  double* B_array[BATCH_LEN];
} copy_batch;
int32_t copy_batch_count = 0;

void mat_cpy_batch(int64_t m, int64_t n, const struct Matrix* m1, struct Matrix* m2, int64_t y1, int64_t x1, int64_t y2, int64_t x2) {
  if (copy_batch_count == BATCH_LEN)
    mat_cpy_flush();
  int32_t i = copy_batch_count;
  copy_batch.rows_array[i] = m;
  copy_batch.cols_array[i] = n;
  copy_batch.A_array[i] = &m1->A[y1 + x1 * m1->M];
  copy_batch.lda_array[i] = m1->M;
  copy_batch.B_array[i] = &m2->A[y2 + x2 * m2->M];
  copy_batch.ldb_array[i] = m2->M;
  copy_batch_count = i + (int)(m > 0 && n > 0);
}

void mat_cpy_flush() {
#ifdef _MKL_BATCH
  char trans_array[BATCH_LEN];
  double alpha_array[BATCH_LEN];
  size_t rows_ui64[BATCH_LEN], cols_ui64[BATCH_LEN], lda_ui64[BATCH_LEN], ldb_ui64[BATCH_LEN], group_size[BATCH_LEN];
#pragma omp parallel for
  for (int32_t i = 0; i < copy_batch_count; i++) {
    trans_array[i] = 'N';
    alpha_array[i] = 1.;
    rows_ui64[i] = copy_batch.rows_array[i];
    cols_ui64[i] = copy_batch.cols_array[i];
    lda_ui64[i] = copy_batch.lda_array[i];
    ldb_ui64[i] = copy_batch.ldb_array[i];
    group_size[i] = 1;
  }
  MKL_Domatcopy_batch('C', trans_array, rows_ui64, cols_ui64, alpha_array,
    copy_batch.A_array, lda_ui64, copy_batch.B_array, ldb_ui64, copy_batch_count, group_size);
#else
#pragma omp parallel for
  for (int32_t i = 0; i < copy_batch_count; i++)
    LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', copy_batch.rows_array[i], copy_batch.cols_array[i],
      copy_batch.A_array[i], copy_batch.lda_array[i], copy_batch.B_array[i], copy_batch.ldb_array[i]);
#endif
  copy_batch_count = 0;
}

void mmult(char ta, char tb, const struct Matrix* A, const struct Matrix* B, struct Matrix* C, double alpha, double beta) {
  int64_t k = ta == 'N' ? A->N : A->M;
  CBLAS_TRANSPOSE tac = ta == 'N' ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE tbc = tb == 'N' ? CblasNoTrans : CblasTrans;
  int64_t lda = 1 < A->M ? A->M : 1;
  int64_t ldb = 1 < B->M ? B->M : 1;
  int64_t ldc = 1 < C->M ? C->M : 1;
  cblas_dgemm(CblasColMajor, tac, tbc, C->M, C->N, k, alpha, A->A, lda, B->A, ldb, beta, C->A, ldc);
}

void chol_decomp(struct Matrix* A) {
  int64_t lda = 1 < A->M ? A->M : 1;
  LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', A->M, A->A, lda);
}

void svd_U(struct Matrix* A, struct Matrix* U, double* S) {
  int64_t rank_a = A->M < A->N ? A->M : A->N;
  int64_t lda = 1 < A->M ? A->M : 1;
  int64_t ldu = 1 < U->M ? U->M : 1;
  int64_t ldv = 1 < A->N ? A->N : 1;
  LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'N', A->M, A->N, A->A, lda, S, U->A, ldu, NULL, ldv, &S[rank_a]);
}

struct IdRowBatch {
  int32_t rows_array[BATCH_LEN];
  int32_t cols_array[BATCH_LEN];
  double* A_array[BATCH_LEN];
  int32_t lda_array[BATCH_LEN];
  int32_t* piv_array[BATCH_LEN];
  double* work_array[BATCH_LEN];
} idrow_batch;
int32_t idrow_batch_count = 0;

void id_row_batch(struct Matrix* A, int32_t arows[], double* work) {
  if (idrow_batch_count == BATCH_LEN)
    id_row_flush();
  int32_t i = idrow_batch_count;
  idrow_batch.rows_array[i] = A->M;
  idrow_batch.cols_array[i] = A->N;
  idrow_batch.A_array[i] = A->A;
  idrow_batch.lda_array[i] = 1 < A->M ? A->M : 1;
  idrow_batch.piv_array[i] = arows;
  idrow_batch.work_array[i] = work;
  idrow_batch_count = i + (int)(A->M > 0 && A->N > 0);
}

void id_row_flush() {
#ifdef _MKL_BATCH
  char R_array[BATCH_LEN];
  char N_array[BATCH_LEN];
  char L_array[BATCH_LEN];
  char U_array[BATCH_LEN];
  double alpha_array[BATCH_LEN];
  int32_t one_array[BATCH_LEN];
  int32_t info_array[BATCH_LEN];

  size_t rows_ui64[BATCH_LEN];
  size_t cols_ui64[BATCH_LEN];
  size_t lda_ui64[BATCH_LEN];
  size_t group_size[BATCH_LEN];
#pragma omp parallel for
  for (int32_t i = 0; i < idrow_batch_count; i++) {
    R_array[i] = 'R';
    N_array[i] = 'N';
    L_array[i] = 'L';
    U_array[i] = 'U';
    alpha_array[i] = 1.;
    one_array[i] = 1;
    rows_ui64[i] = idrow_batch.rows_array[i];
    cols_ui64[i] = idrow_batch.cols_array[i];
    lda_ui64[i] = idrow_batch.lda_array[i];
    group_size[i] = 1;
  }
  
  MKL_Domatcopy_batch('C', N_array, rows_ui64, cols_ui64, alpha_array,
    (const double**)idrow_batch.A_array, lda_ui64, idrow_batch.work_array, lda_ui64, idrow_batch_count, group_size);
  
  dgetrf_batch(idrow_batch.rows_array, idrow_batch.cols_array, idrow_batch.work_array, idrow_batch.lda_array,
    idrow_batch.piv_array, &idrow_batch_count, one_array, info_array);

  dtrsm_batch(R_array, U_array, N_array, N_array, idrow_batch.rows_array, idrow_batch.cols_array, alpha_array,
    (const double**)idrow_batch.work_array, idrow_batch.lda_array, idrow_batch.A_array, idrow_batch.lda_array, &idrow_batch_count, one_array);

  dtrsm_batch(R_array, L_array, N_array, U_array, idrow_batch.rows_array, idrow_batch.cols_array, alpha_array,
    (const double**)idrow_batch.work_array, idrow_batch.lda_array, idrow_batch.A_array, idrow_batch.lda_array, &idrow_batch_count, one_array);
#else
#pragma omp parallel for
  for (int32_t i = 0; i < idrow_batch_count; i++) {
    LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', idrow_batch.rows_array[i], idrow_batch.cols_array[i], 
      idrow_batch.A_array[i], idrow_batch.lda_array[i], idrow_batch.work_array[i], idrow_batch.lda_array[i]);
    LAPACKE_dgetrf(LAPACK_COL_MAJOR, idrow_batch.rows_array[i], idrow_batch.cols_array[i],
      idrow_batch.work_array[i], idrow_batch.lda_array[i], idrow_batch.piv_array[i]);
    cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, idrow_batch.rows_array[i], idrow_batch.cols_array[i], 1.,
      idrow_batch.work_array[i], idrow_batch.lda_array[i], idrow_batch.A_array[i], idrow_batch.lda_array[i]);
    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit, idrow_batch.rows_array[i], idrow_batch.cols_array[i], 1.,
      idrow_batch.work_array[i], idrow_batch.lda_array[i], idrow_batch.A_array[i], idrow_batch.lda_array[i]);
  }
#endif
  idrow_batch_count = 0;
}

void upper_tri_reflec_mult(char side, int64_t lenR, const struct Matrix* R, struct Matrix* A) {
  int64_t lda = 1 < A->M ? A->M : 1;
  int64_t y = 0;
  if (side == 'L' || side == 'l')
    for (int64_t i = 0; i < lenR; i++) {
      int64_t ldr = 1 < R[i].M ? R[i].M : 1;
      cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, R[i].M, A->N, 1., R[i].A, ldr, &A->A[y], lda);
      y = y + R[i].M;
    }
  else if (side == 'R' || side == 'r')
    for (int64_t i = 0; i < lenR; i++) {
      int64_t ldr = 1 < R[i].M ? R[i].M : 1;
      cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit, A->M, R[i].M, 1., R[i].A, ldr, &A->A[y], lda);
      y = y + A->M * R[i].M;
    }
}

void qr_full(struct Matrix* Q, struct Matrix* R, double* tau) {
  int64_t ldq = 1 < Q->M ? Q->M : 1;
  int64_t ldr = 1 < R->M ? R->M : 1;
  LAPACKE_dgeqrf(LAPACK_COL_MAJOR, Q->M, R->N, Q->A, ldq, tau);
  LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', R->M, R->N, Q->A, ldq, R->A, ldr);
  LAPACKE_dorgqr(LAPACK_COL_MAJOR, Q->M, Q->N, R->N, Q->A, ldq, tau);
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
