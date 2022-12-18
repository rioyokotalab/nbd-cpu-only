
#include "nbd.h"
#include "mkl.h"
#include <string.h>

#define ALIGN 32

void init_batch_lib() { }

void finalize_batch_lib() { }

void alloc_matrices_aligned(double** A_ptr, double** A_buffer, int64_t M, int64_t N, int64_t count) {
  int64_t stride = M * N;
  *A_ptr = (double*)MKL_calloc(count * stride, sizeof(double), ALIGN);
  *A_buffer = *A_ptr;
}

void flush_buffer(char dir, double* A_ptr, double* A_buffer, int64_t len) {
  if (A_ptr != A_buffer) {
    if (dir == 'S')
      memcpy(A_ptr, A_buffer, sizeof(double) * len);
    else if (dir == 'G')
      memcpy(A_buffer, A_ptr, sizeof(double) * len);
  }
}

void free_matrices(double* A_ptr, double* A_buffer) {
  if (A_buffer != A_ptr)
    MKL_free(A_buffer);
  MKL_free(A_ptr);
}

void batch_cholesky_factor(int64_t R_dim, int64_t S_dim, const double* U_ptr, double* A_ptr, int64_t N_up, double** A_up, 
  int64_t N_rows, int64_t N_cols, int64_t col_offset, const int64_t row_A[], const int64_t col_A[], const int64_t dimr[], const int64_t dims[]) {
  
  *(&N_rows) = -1; // not used param to get rid of warning.
  int64_t N_dim = R_dim + S_dim;
  int64_t NNZ = col_A[N_cols] - col_A[0];
  int64_t stride = N_dim * N_dim;

  const double** A_lis_diag = (const double**)malloc(sizeof(double*) * N_cols);
  const double** U_lis_diag = (const double**)malloc(sizeof(double*) * N_cols);
  const double** U_lis = (const double**)malloc(sizeof(double*) * NNZ);
  const double** V_lis = (const double**)malloc(sizeof(double*) * NNZ);
  const double** ARS_lis = (const double**)malloc(sizeof(double*) * N_cols);

  double** D_lis = (double**)malloc(sizeof(double*) * N_cols);
  double** UD_lis = (double**)malloc(sizeof(double*) * N_cols);
  double** A_lis = (double**)malloc(sizeof(double*) * NNZ);
  double** B_lis = (double**)malloc(sizeof(double*) * NNZ);
  double** ASS_lis = (double**)malloc(sizeof(double*) * N_cols);

  double* UD_data = (double*)malloc(sizeof(double) * N_cols * stride);
  double* B_data = (double*)malloc(sizeof(double) * NNZ * stride);

  for (int64_t x = 0; x < N_cols; x++) {
    int64_t diag_id = 0;
    for (int64_t yx = col_A[x]; yx < col_A[x + 1]; yx++) {
      int64_t y = row_A[yx];
      if (x + col_offset == y)
        diag_id = yx;
      U_lis[yx] = U_ptr + stride * y;
      V_lis[yx] = UD_data + stride * x;
      A_lis[yx] = A_ptr + stride * yx;
      B_lis[yx] = B_data + stride * yx;
    }

    A_lis_diag[x] = A_ptr + stride * diag_id;
    U_lis_diag[x] = U_ptr + stride * (x + col_offset);
    ARS_lis[x] = A_ptr + stride * diag_id + R_dim;
    D_lis[x] = B_data + stride * x;
    UD_lis[x] = UD_data + stride * x;
    ASS_lis[x] = A_ptr + stride * diag_id + (N_dim + 1) * R_dim;
  }

  CBLAS_SIDE right = CblasRight;
  CBLAS_UPLO lower = CblasLower;
  CBLAS_TRANSPOSE trans = CblasTrans;
  CBLAS_TRANSPOSE no_trans = CblasNoTrans;
  CBLAS_DIAG non_unit = CblasNonUnit;
  double one = 1.;
  double zero = 0.;
  double minus_one = -1.;
  MKL_INT N = N_dim;
  MKL_INT R = R_dim;
  MKL_INT S = S_dim;
  MKL_INT D = N_cols;
  MKL_INT Z = NNZ;

  cblas_dgemm_batch(CblasColMajor, &no_trans, &no_trans, &N, &R, &N, &one, 
    A_lis_diag, &N, U_lis_diag, &N, &zero, UD_lis, &N, 1, &D);
  cblas_dgemm_batch(CblasColMajor, &trans, &no_trans, &R, &R, &N, &one, 
    U_lis_diag, &N, (const double**)UD_lis, &N, &zero, D_lis, &N, 1, &D);
  cblas_dcopy(stride * N_cols, U_ptr + stride * col_offset, 1, UD_data, 1);

  for (int64_t i = 0; i < N_cols; i++) {
    int64_t dimc = dimr[i + col_offset];
    if (dimc < R_dim)
      cblas_dcopy(R_dim - dimc, &one, 0, D_lis[i] + (N_dim + 1) * dimc, N_dim + 1);
    LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', R_dim, D_lis[i], N_dim);      
  }
  cblas_dtrsm_batch(CblasColMajor, &right, &lower, &trans, &non_unit, &N, &R, &one, 
    (const double**)D_lis, &N, UD_lis, &N, 1, &D);

  cblas_dgemm_batch(CblasColMajor, &trans, &no_trans, &N, &N, &N, &one, 
    U_lis, &N, (const double**)A_lis, &N, &zero, B_lis, &N, 1, &Z);
  cblas_dgemm_batch(CblasColMajor, &no_trans, &no_trans, &N, &N, &N, &one, 
    (const double**)B_lis, &N, V_lis, &N, &zero, A_lis, &N, 1, &Z);
  cblas_dgemm_batch(CblasColMajor, &no_trans, &trans, &S, &S, &R, &minus_one, 
    ARS_lis, &N, ARS_lis, &N, &one, ASS_lis, &N, 1, &D);

  for (int64_t x = 0; x < N_cols; x++)
    for (int64_t yx = col_A[x]; yx < col_A[x + 1]; yx++) {
      double* A = &A_ptr[yx * stride + (N_dim + 1) * R_dim];
      double* B = A_up[yx];
      int64_t y = row_A[yx];
      int64_t m = dims[y];
      int64_t n = dims[x + col_offset];
      LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', m, n, A, N_dim, B, N_up);
    }

  free(A_lis_diag);
  free(U_lis_diag);
  free(U_lis);
  free(V_lis);
  free(ARS_lis);

  free(D_lis);
  free(UD_lis);
  free(A_lis);
  free(B_lis);
  free(ASS_lis);

  free(UD_data);
  free(B_data);
}

void chol_decomp(double* A, int64_t Nblocks, int64_t block_dim, const int64_t dims[]) {
  int64_t lda = Nblocks * block_dim;
  double* B = (double*)calloc(lda * lda, sizeof(double));
  int64_t row = 0;
  for (int64_t i = 0; i < Nblocks; i++) {
    int64_t col = 0;
    for (int64_t j = 0; j < Nblocks; j++) {
      LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', dims[i], dims[j], &A[i * block_dim + (j * block_dim * lda)], lda,
        &B[row + col * lda], lda);
      col = col + dims[j];
    }
    row = row + dims[i];
  }
  LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', row, row, B, lda, A, lda);
  LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', row, A, lda);
  free(B);
}
