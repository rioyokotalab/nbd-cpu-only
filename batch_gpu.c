
#include "nbd.h"
#include "mkl.h"
#include "magma_v2.h"

#include <stdlib.h>

#define ALIGN 32
magma_queue_t queue = NULL;

void init_batch_lib() {
  magma_init();
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  int num_device = magma_num_gpus();
  int device = mpi_rank % num_device;
  magma_queue_create(device, &queue);
}

void finalize_batch_lib() {
  magma_finalize();
}

void sync_batch_lib() {
  magma_queue_sync(queue);
}

void alloc_matrices_aligned(double** A_ptr, int M, int N, int count) {
  size_t stride = (size_t)M * N;
  magma_dmalloc(A_ptr, count * stride);
  magma_memset(*A_ptr, 0, count * stride * sizeof(double));
}

void free_matrices(double* A_ptr) {
  magma_free(A_ptr);
}

void copy_basis(char dir, const double* Ur_in, const double* Us_in, double* U_out, int IR_dim, int IS_dim, int OR_dim, int OS_dim, int ldu_in, int ldu_out) {
  IR_dim = IR_dim < OR_dim ? IR_dim : OR_dim;
  IS_dim = IS_dim < OS_dim ? IS_dim : OS_dim;
  int N_in = IR_dim + IS_dim;
  int N_out = OR_dim + OS_dim;
  if (dir == 'G') {
    magma_dgetmatrix_async(N_in, IR_dim, Ur_in, ldu_in, U_out, ldu_out, queue);
    magma_dgetmatrix_async(N_in, IS_dim, Us_in, ldu_in, U_out + (size_t)OR_dim * ldu_out, ldu_out, queue);
  }
  else if (dir == 'S') {
    int diff_r = OR_dim - IR_dim;
    int diff_s = OS_dim - IS_dim;
    double* U = (double*)calloc(N_out * N_out, sizeof(double));
    LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', N_in, IR_dim, Ur_in, ldu_in, U, N_out);
    LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', N_in, IS_dim, Us_in, ldu_in, U + (size_t)OR_dim * N_out, N_out);
    if (diff_r > 0)
      LAPACKE_dlaset(LAPACK_COL_MAJOR, 'A', diff_r, diff_r, 0., 1., U + ((size_t)N_out * IR_dim + N_in), N_out);
    if (diff_s > 0)
      LAPACKE_dlaset(LAPACK_COL_MAJOR, 'A', diff_s, diff_s, 0., 1., U + (((size_t)N_out + 1) * ((size_t)OR_dim + IS_dim)), N_out);
    magma_dsetmatrix(N_out, N_out, U, N_out, U_out, ldu_out, queue);
    free(U);
  }
}

void copy_mat(char dir, const double* A_in, double* A_out, int M_in, int N_in, int lda_in, int M_out, int N_out, int lda_out) {
  M_in = M_in < M_out ? M_in : M_out;
  N_in = N_in < N_out ? N_in : N_out;
  if (dir == 'G')
    magma_dgetmatrix_async(M_in, N_in, A_in, lda_in, A_out, lda_out, queue);
  else if (dir == 'S') {
    int diff_m = M_out - M_in;
    int diff_n = N_out - N_in;
    int len_i = diff_m < diff_n ? diff_m : diff_n;
    double* A = (double*)calloc(M_out * N_out, sizeof(double));
    LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', M_in, N_in, A_in, lda_in, A, M_out);
    if (len_i > 0)
      LAPACKE_dlaset(LAPACK_COL_MAJOR, 'A', diff_m, diff_n, 0., 1., A + ((size_t)M_out * N_in + M_in), M_out);
    magma_dsetmatrix(M_out, N_out, A, M_out, A_out, lda_out, queue);
    free(A);
  }
}

void batch_cholesky_factor(int R_dim, int S_dim, const double* U_ptr, double* A_ptr, int N_cols, int col_offset, const int row_A[], const int col_A[]) {
  int N_dim = R_dim + S_dim;
  int NNZ = col_A[N_cols] - col_A[0];
  size_t stride = (size_t)N_dim * N_dim;

  const double** A_lis_diag = (const double**)malloc(sizeof(double*) * N_cols);
  const double** U_lis_diag = (const double**)malloc(sizeof(double*) * N_cols);
  const double** U_lis = (const double**)malloc(sizeof(double*) * NNZ);
  const double** V_lis = (const double**)malloc(sizeof(double*) * NNZ);
  const double** ARS_lis = (const double**)malloc(sizeof(double*) * N_cols);
  const double** A_lis_diag_dev, **U_lis_diag_dev, **U_lis_dev, **V_lis_dev, **ARS_lis_dev;

  magma_malloc((void**)&A_lis_diag_dev, sizeof(double*) * N_cols);
  magma_malloc((void**)&U_lis_diag_dev, sizeof(double*) * N_cols);
  magma_malloc((void**)&U_lis_dev, sizeof(double*) * NNZ);
  magma_malloc((void**)&V_lis_dev, sizeof(double*) * NNZ);
  magma_malloc((void**)&ARS_lis_dev, sizeof(double*) * N_cols);

  double** D_lis = (double**)malloc(sizeof(double*) * N_cols);
  double** UD_lis = (double**)malloc(sizeof(double*) * N_cols);
  double** A_lis = (double**)malloc(sizeof(double*) * NNZ);
  double** B_lis = (double**)malloc(sizeof(double*) * NNZ);
  double** ASS_lis = (double**)malloc(sizeof(double*) * N_cols);
  double** D_lis_dev, **UD_lis_dev, **A_lis_dev, **B_lis_dev, **ASS_lis_dev;

  magma_malloc((void**)&D_lis_dev, sizeof(double*) * N_cols);
  magma_malloc((void**)&UD_lis_dev, sizeof(double*) * N_cols);
  magma_malloc((void**)&A_lis_dev, sizeof(double*) * NNZ);
  magma_malloc((void**)&B_lis_dev, sizeof(double*) * NNZ);
  magma_malloc((void**)&ASS_lis_dev, sizeof(double*) * N_cols);

  double* D_data, *UD_data, *B_data;
  magma_dmalloc(&D_data, N_cols * stride);
  magma_dmalloc(&UD_data, N_cols * stride);
  magma_dmalloc(&B_data, NNZ * stride);

  int* info_array;
  magma_imalloc(&info_array, N_cols);

  for (int x = 0; x < N_cols; x++) {
    int diag_id = 0;
    for (int yx = col_A[x]; yx < col_A[x + 1]; yx++) {
      int y = row_A[yx];
      if (x + col_offset == y)
        diag_id = yx;
      U_lis[yx] = U_ptr + stride * y;
      V_lis[yx] = UD_data + stride * x;
      A_lis[yx] = A_ptr + stride * yx;
      B_lis[yx] = B_data + stride * yx;
    }

    A_lis_diag[x] = A_ptr + stride * diag_id;
    U_lis_diag[x] = U_ptr + stride * row_A[diag_id];
    ARS_lis[x] = A_ptr + stride * diag_id + R_dim;
    D_lis[x] = D_data + stride * x;
    UD_lis[x] = UD_data + stride * x;
    ASS_lis[x] = A_ptr + stride * diag_id + (size_t)(N_dim + 1) * R_dim;
  }

  magma_setvector_async(N_cols, sizeof(double*), A_lis_diag, 1, A_lis_diag_dev, 1, queue);
  magma_setvector_async(N_cols, sizeof(double*), U_lis_diag, 1, U_lis_diag_dev, 1, queue);
  magma_setvector_async(NNZ, sizeof(double*), U_lis, 1, U_lis_dev, 1, queue);
  magma_setvector_async(NNZ, sizeof(double*), V_lis, 1, V_lis_dev, 1, queue);
  magma_setvector_async(N_cols, sizeof(double*), ARS_lis, 1, ARS_lis_dev, 1, queue);

  magma_setvector_async(N_cols, sizeof(double*), D_lis, 1, D_lis_dev, 1, queue);
  magma_setvector_async(N_cols, sizeof(double*), UD_lis, 1, UD_lis_dev, 1, queue);
  magma_setvector_async(NNZ, sizeof(double*), A_lis, 1, A_lis_dev, 1, queue);
  magma_setvector_async(NNZ, sizeof(double*), B_lis, 1, B_lis_dev, 1, queue);
  magma_setvector_async(N_cols, sizeof(double*), ASS_lis, 1, ASS_lis_dev, 1, queue);

  magmablas_dgemm_batched(MagmaNoTrans, MagmaNoTrans, N_dim, R_dim, N_dim, 1., 
    A_lis_diag_dev, N_dim, U_lis_diag_dev, N_dim, 0., UD_lis_dev, N_dim, N_cols, queue);
  magmablas_dgemm_batched(MagmaTrans, MagmaNoTrans, R_dim, R_dim, N_dim, 1., 
    U_lis_diag_dev, N_dim, (const double**)UD_lis_dev, N_dim, 0., D_lis_dev, N_dim, N_cols, queue);
  magmablas_dlacpy_batched(MagmaFull, N_dim, N_dim, U_lis_diag_dev, N_dim, UD_lis_dev, N_dim, N_cols, queue);

  magma_dpotrf_batched(MagmaLower, R_dim, D_lis_dev, N_dim, info_array, N_cols, queue);
  magmablas_dtrsm_batched(MagmaRight, MagmaLower, MagmaTrans, MagmaNonUnit, N_dim, R_dim, 1., 
    D_lis_dev, N_dim, UD_lis_dev, N_dim, N_cols, queue);

  magmablas_dgemm_batched(MagmaTrans, MagmaNoTrans, N_dim, N_dim, N_dim, 1., 
    U_lis_dev, N_dim, (const double**)A_lis_dev, N_dim, 0., B_lis_dev, N_dim, NNZ, queue);
  magmablas_dgemm_batched(MagmaNoTrans, MagmaNoTrans, N_dim, N_dim, N_dim, 1., 
    (const double**)B_lis_dev, N_dim, V_lis_dev, N_dim, 0., A_lis_dev, N_dim, NNZ, queue);
  magmablas_dgemm_batched(MagmaNoTrans, MagmaTrans, S_dim, S_dim, R_dim, -1., 
    ARS_lis_dev, N_dim, ARS_lis_dev, N_dim, 1., ASS_lis_dev, N_dim, N_cols, queue);

  magma_free(A_lis_diag_dev);
  magma_free(U_lis_diag_dev);
  magma_free(U_lis_dev);
  magma_free(V_lis_dev);
  magma_free(ARS_lis_dev);

  magma_free(D_lis_dev);
  magma_free(UD_lis_dev);
  magma_free(A_lis_dev);
  magma_free(B_lis_dev);
  magma_free(ASS_lis_dev);

  magma_free(D_data);
  magma_free(UD_data);
  magma_free(B_data);
  magma_free(info_array);

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
}


