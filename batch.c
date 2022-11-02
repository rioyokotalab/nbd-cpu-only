
#include "nbd.h"
#include "mkl.h"
#include <string.h>

#define ALIGN 32

void init_batch_lib() { }

void finalize_batch_lib() { }

void sync_batch_lib() { }

void alloc_matrices_aligned(double** A_ptr, int* M_align, int M, int N, int count) {
  int rem = M & (ALIGN - 1);
  *M_align = (rem ? ALIGN : 0) + M - rem;
  size_t A_stride_mat = (size_t)(*M_align) * N;
  *A_ptr = (double*)MKL_calloc(count * A_stride_mat, sizeof(double), ALIGN);
}

void free_matrices(double* A_ptr) {
  MKL_free(A_ptr);
}

void copy_basis(char dir, const double* Ur_in, const double* Us_in, double* U_out, int IR_dim, int IS_dim, int OR_dim, int OS_dim, int ldu_in, int ldu_out) {
  if (dir == 'G' || dir == 'S') {
    IR_dim = IR_dim < OR_dim ? IR_dim : OR_dim;
    IS_dim = IS_dim < OS_dim ? IS_dim : OS_dim;
    int n_in = IR_dim + IS_dim;
    MKL_Domatcopy('C', 'N', n_in, IR_dim, 1., Ur_in, ldu_in, U_out, ldu_out);
    MKL_Domatcopy('C', 'N', n_in, IS_dim, 1., Us_in, ldu_in, U_out + (size_t)OR_dim * ldu_out, ldu_out);

    if (dir == 'S') {
      int diff_r = OR_dim - IR_dim;
      int diff_s = OS_dim - IS_dim;
      double one = 1.;
      if (diff_r > 0)
        cblas_dcopy(diff_r, &one, 0, U_out + ((size_t)ldu_out * IR_dim + n_in), ldu_out + 1);
      if (diff_s > 0)
        cblas_dcopy(diff_s, &one, 0, U_out + (((size_t)ldu_out + 1) * ((size_t)OR_dim + IS_dim)), ldu_out + 1);
    }
  }
}

void copy_mat(char dir, const double* A_in, double* A_out, int M_in, int N_in, int lda_in, int M_out, int N_out, int lda_out) {
  if (dir == 'G' || dir == 'S') {
    M_in = M_in < M_out ? M_in : M_out;
    N_in = N_in < N_out ? N_in : N_out;
    MKL_Domatcopy('C', 'N', M_in, N_in, 1., A_in, lda_in, A_out, lda_out);

    if (dir == 'S') {
      int diff_m = M_out - M_in;
      int diff_n = N_out - N_in;
      int len_i = diff_m < diff_n ? diff_m : diff_n;
      double one = 1.;
      if (len_i > 0)
        cblas_dcopy(len_i, &one, 0, A_out + ((size_t)lda_out * N_in + M_in), lda_out + 1);
    }
  }
}

void compute_rs_splits_left(const double* U_ptr, const double* A_ptr, double* out_ptr, const int* row_A, int N, int N_align, int A_count) {
  const double** U_lis = (const double**)MKL_malloc(sizeof(double*) * A_count, ALIGN);
  const double** A_lis = (const double**)MKL_malloc(sizeof(double*) * A_count, ALIGN);
  double** O_lis = (double**)MKL_malloc(sizeof(double*) * A_count, ALIGN);

  size_t A_stride_mat = (size_t)N_align * N;
  for (int i = 0; i < A_count; i++) {
    int row = row_A[i];
    U_lis[i] = U_ptr + A_stride_mat * row;
    A_lis[i] = A_ptr + A_stride_mat * i;
    O_lis[i] = out_ptr + A_stride_mat * i;
  }

  CBLAS_TRANSPOSE trans = CblasConjTrans;
  CBLAS_TRANSPOSE no_trans = CblasNoTrans;
  double one = 1.;
  double zero = 0.;
  cblas_dgemm_batch(CblasColMajor, &trans, &no_trans, &N, &N, &N, &one, U_lis, &N_align, A_lis, &N_align, &zero, O_lis, &N_align, 1, &A_count);
  
  MKL_free(U_lis);
  MKL_free(A_lis);
  MKL_free(O_lis);
}

void compute_rs_splits_right(const double* V_ptr, const double* A_ptr, double* out_ptr, const int* col_A, int N, int N_align, int A_count) {
  const double** V_lis = (const double**)MKL_malloc(sizeof(double*) * A_count, ALIGN);
  const double** A_lis = (const double**)MKL_malloc(sizeof(double*) * A_count, ALIGN);
  double** O_lis = (double**)MKL_malloc(sizeof(double*) * A_count, ALIGN);

  size_t A_stride_mat = (size_t)N_align * N;
  int col = 0;
  for (int i = 0; i < A_count; i++) {
    while (col_A[col + 1] <= i)
      col = col + 1;
    V_lis[i] = V_ptr + A_stride_mat * col;
    A_lis[i] = A_ptr + A_stride_mat * i;
    O_lis[i] = out_ptr + A_stride_mat * i;
  }

  CBLAS_TRANSPOSE no_trans = CblasNoTrans;
  double one = 1.;
  double zero = 0.;
  cblas_dgemm_batch(CblasColMajor, &no_trans, &no_trans, &N, &N, &N, &one, A_lis, &N_align, V_lis, &N_align, &zero, O_lis, &N_align, 1, &A_count);
  
  MKL_free(V_lis);
  MKL_free(A_lis);
  MKL_free(O_lis);
}

void factor_diag(int N_diag, double* D_ptr, double* U_ptr, int R_dim, int S_dim, int N_align) {
  int N_dim = R_dim + S_dim;
  size_t A_stride_mat = (size_t)N_align * N_dim;
  size_t UD_stride = (size_t)N_align * R_dim;
  double** D_lis = (double**)MKL_malloc(sizeof(double*) * N_diag, ALIGN);
  double** U_lis = (double**)MKL_malloc(sizeof(double*) * N_diag, ALIGN);
  double** UD_lis = (double**)MKL_malloc(sizeof(double*) * N_diag, ALIGN);
  double* UD_data = (double*)MKL_malloc(sizeof(double) * N_diag * UD_stride, ALIGN);

  for (int i = 0; i < N_diag; i++) {
    D_lis[i] = D_ptr + A_stride_mat * i;
    U_lis[i] = U_ptr + A_stride_mat * i;
    UD_lis[i] = UD_data + UD_stride * i;
  }

  CBLAS_SIDE right = CblasRight;
  CBLAS_UPLO lower = CblasLower;
  CBLAS_TRANSPOSE trans = CblasTrans;
  CBLAS_TRANSPOSE no_trans = CblasNoTrans;
  CBLAS_DIAG non_unit = CblasNonUnit;
  double one = 1.;
  double zero = 0.;
  cblas_dgemm_batch(CblasColMajor, &no_trans, &no_trans, &N_dim, &R_dim, &N_dim, &one, 
    (const double**)D_lis, &N_align, (const double**)U_lis, &N_align, &zero, UD_lis, &N_align, 1, &N_diag);
  cblas_dgemm_batch(CblasColMajor, &trans, &no_trans, &R_dim, &R_dim, &N_dim, &one, 
    (const double**)U_lis, &N_align, (const double**)UD_lis, &N_align, &zero, D_lis, &N_align, 1, &N_diag);

  for (int i = 0; i < N_diag; i++)
    LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', R_dim, D_lis[i], N_align);
  cblas_dtrsm_batch(CblasColMajor, &right, &lower, &trans, &non_unit, &N_dim, &R_dim, &one, (const double**)D_lis, &N_align, U_lis, &N_align, 1, &N_diag);

  MKL_free(D_lis);
  MKL_free(U_lis);
  MKL_free(UD_lis);
  MKL_free(UD_data);
}

void schur_diag(int N_diag, double* A_ptr, const int* diag_idx, int R_dim, int S_dim, int N_align) {
  int N_dim = R_dim + S_dim;
  size_t A_stride_mat = (size_t)N_align * N_dim;
  const double** ARS_lis = (const double**)MKL_malloc(sizeof(double*) * N_diag, ALIGN);
  double** ASS_lis = (double**)MKL_malloc(sizeof(double*) * N_diag, ALIGN);

  for (int i = 0; i < N_diag; i++) {
    int diag = diag_idx[i];
    ARS_lis[i] = A_ptr + A_stride_mat * diag + R_dim;
    ASS_lis[i] = A_ptr + A_stride_mat * diag + ((size_t)N_align * R_dim + R_dim);
  }

  CBLAS_TRANSPOSE trans = CblasTrans;
  CBLAS_TRANSPOSE no_trans = CblasNoTrans;
  double one = 1.;
  double minus_one = -1.;
  cblas_dgemm_batch(CblasColMajor, &no_trans, &trans, &S_dim, &S_dim, &R_dim, &minus_one, ARS_lis, &N_align, ARS_lis, &N_align, &one, ASS_lis, &N_align, 1, &N_diag);

  MKL_free(ARS_lis);
  MKL_free(ASS_lis);
}

