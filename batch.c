
#include "nbd.h"
#include "mkl.h"
#include <string.h>

#define ALIGN 32

void init_batch_lib() { }

void finalize_batch_lib() { }

void sync_batch_lib() { }

void alloc_matrices_aligned(double** A_ptr, int64_t M, int64_t N, int64_t count) {
  int64_t stride = M * N;
  *A_ptr = (double*)MKL_calloc(count * stride, sizeof(double), ALIGN);
}

void free_matrices(double* A_ptr) {
  MKL_free(A_ptr);
}

void copy_basis(char dir, const double* Ur_in, const double* Us_in, double* U_out, int64_t IR_dim, int64_t IS_dim, int64_t OR_dim, int64_t OS_dim, int64_t ldu_in, int64_t ldu_out) {
  if (dir == 'G' || dir == 'S') {
    IR_dim = IR_dim < OR_dim ? IR_dim : OR_dim;
    IS_dim = IS_dim < OS_dim ? IS_dim : OS_dim;
    int64_t n_in = IR_dim + IS_dim;
    MKL_Domatcopy('C', 'N', n_in, IR_dim, 1., Ur_in, ldu_in, U_out, ldu_out);
    MKL_Domatcopy('C', 'N', n_in, IS_dim, 1., Us_in, ldu_in, U_out + OR_dim * ldu_out, ldu_out);
  }
}

void copy_mat(char dir, const double* A_in, double* A_out, int64_t M_in, int64_t N_in, int64_t lda_in, int64_t M_out, int64_t N_out, int64_t lda_out) {
  if (dir == 'G' || dir == 'S') {
    M_in = M_in < M_out ? M_in : M_out;
    N_in = N_in < N_out ? N_in : N_out;
    MKL_Domatcopy('C', 'N', M_in, N_in, 1., A_in, lda_in, A_out, lda_out);
  }
}

void batch_cholesky_factor(int64_t R_dim, int64_t S_dim, const double* U_ptr, double* A_ptr, int64_t N_cols, int64_t col_offset, const int64_t row_A[], const int64_t col_A[]) {
  int64_t N_dim = R_dim + S_dim;
  int64_t NNZ = col_A[N_cols] - col_A[0];
  int64_t stride = N_dim * N_dim;

  const double** A_lis_diag = (const double**)MKL_malloc(sizeof(double*) * N_cols, ALIGN);
  const double** U_lis_diag = (const double**)MKL_malloc(sizeof(double*) * N_cols, ALIGN);
  const double** U_lis = (const double**)MKL_malloc(sizeof(double*) * NNZ, ALIGN);
  const double** V_lis = (const double**)MKL_malloc(sizeof(double*) * NNZ, ALIGN);
  const double** ARS_lis = (const double**)MKL_malloc(sizeof(double*) * N_cols, ALIGN);

  double** D_lis = (double**)MKL_malloc(sizeof(double*) * N_cols, ALIGN);
  double** UD_lis = (double**)MKL_malloc(sizeof(double*) * N_cols, ALIGN);
  double** A_lis = (double**)MKL_malloc(sizeof(double*) * NNZ, ALIGN);
  double** B_lis = (double**)MKL_malloc(sizeof(double*) * NNZ, ALIGN);
  double** ASS_lis = (double**)MKL_malloc(sizeof(double*) * N_cols, ALIGN);

  double* UD_data = (double*)MKL_malloc(sizeof(double) * N_cols * stride, ALIGN);
  double* B_data = (double*)MKL_malloc(sizeof(double) * NNZ * stride, ALIGN);

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

  cblas_dgemm_batch(CblasColMajor, &no_trans, &no_trans, &N_dim, &R_dim, &N_dim, &one, 
    A_lis_diag, &N_dim, U_lis_diag, &N_dim, &zero, UD_lis, &N_dim, 1, &N_cols);
  cblas_dgemm_batch(CblasColMajor, &trans, &no_trans, &R_dim, &R_dim, &N_dim, &one, 
    U_lis_diag, &N_dim, (const double**)UD_lis, &N_dim, &zero, D_lis, &N_dim, 1, &N_cols);
  cblas_dcopy(stride * N_cols, U_ptr + stride * col_offset, 1, UD_data, 1);

  for (int64_t i = 0; i < N_cols; i++) {
    int64_t info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', R_dim, D_lis[i], N_dim);
    if (info > 0)
      cblas_dcopy(R_dim - info + 1, &one, 0, D_lis[i] + (N_dim + 1) * (info - 1), N_dim + 1);
  }
  cblas_dtrsm_batch(CblasColMajor, &right, &lower, &trans, &non_unit, &N_dim, &R_dim, &one, 
    (const double**)D_lis, &N_dim, UD_lis, &N_dim, 1, &N_cols);

  cblas_dgemm_batch(CblasColMajor, &trans, &no_trans, &N_dim, &N_dim, &N_dim, &one, 
    U_lis, &N_dim, (const double**)A_lis, &N_dim, &zero, B_lis, &N_dim, 1, &NNZ);
  cblas_dgemm_batch(CblasColMajor, &no_trans, &no_trans, &N_dim, &N_dim, &N_dim, &one, 
    (const double**)B_lis, &N_dim, V_lis, &N_dim, &zero, A_lis, &N_dim, 1, &NNZ);
  cblas_dgemm_batch(CblasColMajor, &no_trans, &trans, &S_dim, &S_dim, &R_dim, &minus_one, 
    ARS_lis, &N_dim, ARS_lis, &N_dim, &one, ASS_lis, &N_dim, 1, &N_cols);

  MKL_free(A_lis_diag);
  MKL_free(U_lis_diag);
  MKL_free(U_lis);
  MKL_free(V_lis);
  MKL_free(ARS_lis);

  MKL_free(D_lis);
  MKL_free(UD_lis);
  MKL_free(A_lis);
  MKL_free(B_lis);
  MKL_free(ASS_lis);

  MKL_free(UD_data);
  MKL_free(B_data);
}

