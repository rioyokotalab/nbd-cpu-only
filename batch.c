
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
  int64_t N_cols, int64_t col_offset, const int64_t row_A[], const int64_t col_A[], const int64_t dimr[]) {
  
  int64_t N_dim = R_dim + S_dim;
  int64_t NNZ = col_A[N_cols] - col_A[0];
  int64_t stride = N_dim * N_dim;

  const double** A_lis_diag = (const double**)malloc(sizeof(double*) * N_cols);
  const double** U_lis_diag = (const double**)malloc(sizeof(double*) * N_cols);
  const double** U_r = (const double**)malloc(sizeof(double*) * NNZ);
  const double** U_s = (const double**)malloc(sizeof(double*) * NNZ);
  const double** V_lis = (const double**)malloc(sizeof(double*) * NNZ);
  const double** ARS_lis = (const double**)malloc(sizeof(double*) * N_cols);
  const double** A_sx = (const double**)malloc(sizeof(double*) * N_cols);

  double** UD_lis = (double**)malloc(sizeof(double*) * N_cols);
  double** A_lis = (double**)malloc(sizeof(double*) * NNZ);
  double** B_lis = (double**)malloc(sizeof(double*) * N_cols);
  double** ASS_lis = (double**)malloc(sizeof(double*) * N_cols);

  double* UD_data = (double*)malloc(sizeof(double) * N_cols * stride);
  double* B_data = (double*)malloc(sizeof(double) * N_cols * stride);
  int64_t* diag_fill = (int64_t*)malloc(sizeof(int64_t) * N_cols * R_dim);
  int64_t fill_len = 0;

  for (int64_t x = 0; x < N_cols; x++) {
    int64_t diag_id = 0;
    for (int64_t yx = col_A[x]; yx < col_A[x + 1]; yx++) {
      int64_t y = row_A[yx];
      if (x + col_offset == y)
        diag_id = yx;
      U_r[yx] = U_ptr + stride * y;
      U_s[yx] = U_ptr + stride * y + R_dim * N_dim;
      V_lis[yx] = UD_data + stride * x;
      A_lis[yx] = A_ptr + stride * yx;
    }

    A_lis_diag[x] = A_ptr + stride * diag_id;
    B_lis[x] = B_data + stride * x;
    U_lis_diag[x] = U_ptr + stride * (x + col_offset);
    ARS_lis[x] = A_ptr + stride * diag_id + R_dim;
    UD_lis[x] = UD_data + stride * x;
    ASS_lis[x] = A_up[diag_id];
    A_sx[x] = B_data + stride * x + R_dim * N_dim;

    int64_t dimc = dimr[x + col_offset];
    int64_t fill_new = R_dim - dimc;
    for (int64_t i = 0; i < fill_new; i++)
      diag_fill[fill_len + i] = x * stride + (N_dim + 1) * (dimc + i);
    fill_len = fill_len + fill_new;
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
  MKL_INT U = N_up;
  MKL_INT R = R_dim;
  MKL_INT S = S_dim;
  MKL_INT D = N_cols;

  cblas_dgemm_batch(CblasColMajor, &no_trans, &no_trans, &N, &R, &N, &one, 
    A_lis_diag, &N, U_lis_diag, &N, &zero, UD_lis, &N, 1, &D);
  cblas_dgemm_batch(CblasColMajor, &trans, &no_trans, &R, &R, &N, &one, 
    U_lis_diag, &N, (const double**)UD_lis, &N, &zero, B_lis, &N, 1, &D);
  cblas_dcopy(stride * N_cols, U_ptr + stride * col_offset, 1, UD_data, 1);
  for (int64_t i = 0; i < fill_len; i++)
    B_data[diag_fill[i]] = 1.;

  for (int64_t i = 0; i < N_cols; i++)
    LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', R_dim, B_lis[i], N_dim);
  cblas_dtrsm_batch(CblasColMajor, &right, &lower, &trans, &non_unit, &N, &R, &one, 
    (const double**)B_lis, &N, UD_lis, &N, 1, &D);

  for (int64_t i = 0; i < NNZ; i += N_cols) {
    MKL_INT len = NNZ - i > N_cols ? N_cols : NNZ - i;
    cblas_dgemm_batch(CblasColMajor, &no_trans, &no_trans, &N, &N, &N, &one,
      (const double**)&A_lis[i], &N, &V_lis[i], &N, &zero, B_lis, &N, 1, &len);
    cblas_dgemm_batch(CblasColMajor, &trans, &no_trans, &N, &R, &N, &one, 
      &U_r[i], &N, (const double**)B_lis, &N, &zero, &A_lis[i], &N, 1, &len);
    cblas_dgemm_batch(CblasColMajor, &trans, &no_trans, &S, &S, &N, &one, 
      &U_s[i], &N, A_sx, &N, &zero, &A_up[i], &U, 1, &len);
  }
  cblas_dgemm_batch(CblasColMajor, &no_trans, &trans, &S, &S, &R, &minus_one,
    ARS_lis, &N, ARS_lis, &N, &one, ASS_lis, &U, 1, &D);

  free(A_lis_diag);
  free(U_lis_diag);
  free(U_r);
  free(U_s);
  free(V_lis);
  free(ARS_lis);

  free(UD_lis);
  free(A_lis);
  free(B_lis);
  free(ASS_lis);
  free(A_sx);

  free(UD_data);
  free(B_data);
  free(diag_fill);
}

void chol_decomp(double* A, int64_t Nblocks, int64_t block_dim, const int64_t dims[]) {
  int64_t lda = Nblocks * block_dim;
  int64_t row = 0;
  for (int64_t i = 0; i < Nblocks; i++) {
    int64_t Arow = i * block_dim;
    if (row < Arow)
      for (int64_t j = 0; j < dims[i]; j++) {
        int64_t rj = row + j;
        int64_t arj = Arow + j;
        cblas_dswap(lda - rj, &A[rj * (lda + 1)], 1, &A[arj * lda + rj], 1);
        cblas_dswap(rj + 1, &A[rj], lda, &A[arj], lda);
      }
    row = row + dims[i];
  }
  LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', row, A, lda);
}
