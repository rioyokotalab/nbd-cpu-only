
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

void alloc_matrices_aligned(double** A_ptr, int* M_align, int M, int N, int count) {
  int rem = M & (ALIGN - 1);
  *M_align = (rem ? ALIGN : 0) + M - rem;
  size_t A_stride_mat = (size_t)(*M_align) * N;
  magma_dmalloc(A_ptr, count * A_stride_mat);
  magma_memset(*A_ptr, 0, count * A_stride_mat * sizeof(double));
}

void free_matrices(double* A_ptr) {
  magma_free(A_ptr);
}

void copy_basis(char dir, const double* Ur_in, const double* Us_in, double* U_out, int IR_dim, int IS_dim, int OR_dim, int OS_dim, int ldu_in, int ldu_out) {
  IR_dim = IR_dim < OR_dim ? IR_dim : OR_dim;
  IS_dim = IS_dim < OS_dim ? IS_dim : OS_dim;
  int n_in = IR_dim + IS_dim;
  if (dir == 'G') {
    magma_dgetmatrix(n_in, IR_dim, Ur_in, ldu_in, U_out, ldu_out, queue);
    magma_dgetmatrix(n_in, IS_dim, Us_in, ldu_in, U_out + (size_t)OR_dim * ldu_out, ldu_out, queue);
  }
  else if (dir == 'S') {
    magma_dsetmatrix(n_in, IR_dim, Ur_in, ldu_in, U_out, ldu_out, queue);
    magma_dsetmatrix(n_in, IS_dim, Us_in, ldu_in, U_out + (size_t)OR_dim * ldu_out, ldu_out, queue);

    int diff_r = OR_dim - IR_dim;
    int diff_s = OS_dim - IS_dim;
    int rs = diff_r < diff_s ? diff_s : diff_r;
    double* one_array = (double*)malloc(sizeof(double) * rs);
    for (int i = 0; i < rs; i++)
      one_array[i] = 1.;
    if (diff_r > 0)
      magma_dsetvector(diff_r, one_array, 1, U_out + ((size_t)ldu_out * IR_dim + n_in), ldu_out + 1, queue);
    if (diff_s > 0)
      magma_dsetvector(diff_s, one_array, 1, U_out + (((size_t)ldu_out + 1) * ((size_t)OR_dim + IS_dim)), ldu_out + 1, queue);
    free(one_array);
  }
}

void copy_mat(char dir, const double* A_in, double* A_out, int M_in, int N_in, int lda_in, int M_out, int N_out, int lda_out) {
  M_in = M_in < M_out ? M_in : M_out;
  N_in = N_in < N_out ? N_in : N_out;
  if (dir == 'G')
    magma_dgetmatrix(M_in, N_in, A_in, lda_in, A_out, lda_out, queue);
  else if (dir == 'S') {
    magma_dsetmatrix(M_in, N_in, A_in, lda_in, A_out, lda_out, queue);

    int diff_m = M_out - M_in;
    int diff_n = N_out - N_in;
    int len_i = diff_m < diff_n ? diff_m : diff_n;
    double* one_array = (double*)malloc(sizeof(double) * len_i);
    for (int i = 0; i < len_i; i++)
      one_array[i] = 1.;
    if (len_i > 0)
      magma_dsetvector(len_i, one_array, 1, A_out + ((size_t)lda_out * N_in + M_in), lda_out + 1, queue);
    free(one_array);
  }
}

void compute_rs_splits_left(const double* U_ptr, const double* A_ptr, double* out_ptr, const int* row_A, int N, int N_align, int A_count) {
  const double** U_lis = (const double**)malloc(sizeof(double*) * A_count), **U_lis_dev;
  const double** A_lis = (const double**)malloc(sizeof(double*) * A_count), **A_lis_dev;
  double** O_lis = (double**)malloc(sizeof(double*) * A_count), **O_lis_dev;
  magma_malloc((void**)&U_lis_dev, sizeof(double*) * A_count);
  magma_malloc((void**)&A_lis_dev, sizeof(double*) * A_count);
  magma_malloc((void**)&O_lis_dev, sizeof(double*) * A_count);

  size_t A_stride_mat = (size_t)N_align * N;
  for (int i = 0; i < A_count; i++) {
    int row = row_A[i];
    U_lis[i] = U_ptr + A_stride_mat * row;
    A_lis[i] = A_ptr + A_stride_mat * i;
    O_lis[i] = out_ptr + A_stride_mat * i;
  }
  magma_setvector_async(A_count, sizeof(double*), U_lis, 1, U_lis_dev, 1, queue);
  magma_setvector_async(A_count, sizeof(double*), A_lis, 1, A_lis_dev, 1, queue);
  magma_setvector_async(A_count, sizeof(double*), O_lis, 1, O_lis_dev, 1, queue);
  magmablas_dgemm_batched(MagmaTrans, MagmaNoTrans, N, N, N, 1., U_lis_dev, N_align, A_lis_dev, N_align, 0., O_lis_dev, N_align, A_count, queue);
  
  magma_free(U_lis_dev);
  magma_free(A_lis_dev);
  magma_free(O_lis_dev);
  free(U_lis);
  free(A_lis);
  free(O_lis);
}

void compute_rs_splits_right(const double* V_ptr, const double* A_ptr, double* out_ptr, const int* col_A, int N, int N_align, int A_count) {
  const double** V_lis = (const double**)malloc(sizeof(double*) * A_count), **V_lis_dev;
  const double** A_lis = (const double**)malloc(sizeof(double*) * A_count), **A_lis_dev;
  double** O_lis = (double**)malloc(sizeof(double*) * A_count), **O_lis_dev;
  magma_malloc((void**)&V_lis_dev, sizeof(double*) * A_count);
  magma_malloc((void**)&A_lis_dev, sizeof(double*) * A_count);
  magma_malloc((void**)&O_lis_dev, sizeof(double*) * A_count);

  size_t A_stride_mat = (size_t)N_align * N;
  int col = 0;
  for (int i = 0; i < A_count; i++) {
    while (col_A[col + 1] <= i)
      col = col + 1;
    V_lis[i] = V_ptr + A_stride_mat * col;
    A_lis[i] = A_ptr + A_stride_mat * i;
    O_lis[i] = out_ptr + A_stride_mat * i;
  }
  magma_setvector_async(A_count, sizeof(double*), V_lis, 1, V_lis_dev, 1, queue);
  magma_setvector_async(A_count, sizeof(double*), A_lis, 1, A_lis_dev, 1, queue);
  magma_setvector_async(A_count, sizeof(double*), O_lis, 1, O_lis_dev, 1, queue);
  magmablas_dgemm_batched(MagmaNoTrans, MagmaNoTrans, N, N, N, 1., A_lis_dev, N_align, V_lis_dev, N_align, 0., O_lis_dev, N_align, A_count, queue);

  magma_free(V_lis_dev);
  magma_free(A_lis_dev);
  magma_free(O_lis_dev);
  free(V_lis);
  free(A_lis);
  free(O_lis);
}

void factor_diag(int N_diag, double* D_ptr, double* U_ptr, int R_dim, int S_dim, int N_align) {
  int N_dim = R_dim + S_dim;
  size_t A_stride_mat = (size_t)N_align * N_dim;
  size_t UD_stride = (size_t)N_align * R_dim;
  double** D_lis = (double**)malloc(sizeof(double*) * N_diag), **D_lis_dev;
  double** U_lis = (double**)malloc(sizeof(double*) * N_diag), **U_lis_dev;
  double** UD_lis = (double**)malloc(sizeof(double*) * N_diag), **UD_lis_dev;
  double* UD_data_dev;
  int* info_array;
  magma_malloc((void**)&D_lis_dev, sizeof(double*) * N_diag);
  magma_malloc((void**)&U_lis_dev, sizeof(double*) * N_diag);
  magma_malloc((void**)&UD_lis_dev, sizeof(double*) * N_diag);
  magma_dmalloc(&UD_data_dev, N_diag * UD_stride);
  magma_imalloc(&info_array, N_diag);

  for (int i = 0; i < N_diag; i++) {
    D_lis[i] = D_ptr + A_stride_mat * i;
    U_lis[i] = U_ptr + A_stride_mat * i;
    UD_lis[i] = UD_data_dev + UD_stride * i;
  }

  magma_setvector_async(N_diag, sizeof(double*), D_lis, 1, D_lis_dev, 1, queue);
  magma_setvector_async(N_diag, sizeof(double*), U_lis, 1, U_lis_dev, 1, queue);
  magma_setvector_async(N_diag, sizeof(double*), UD_lis, 1, UD_lis_dev, 1, queue);

  magmablas_dgemm_batched(MagmaNoTrans, MagmaNoTrans, N_dim, R_dim, N_dim, 1., 
    (const double**)D_lis_dev, N_align, (const double**)U_lis_dev, N_align, 0., UD_lis_dev, N_align, N_diag, queue);
  magmablas_dgemm_batched(MagmaTrans, MagmaNoTrans, R_dim, R_dim, N_dim, 1., 
    (const double**)U_lis_dev, N_align, (const double**)UD_lis_dev, N_align, 0., D_lis_dev, N_align, N_diag, queue);
  magma_dpotrf_batched(MagmaLower, R_dim, D_lis_dev, N_align, info_array, N_diag, queue);
  magmablas_dtrsm_batched(MagmaRight, MagmaLower, MagmaTrans, MagmaNonUnit, N_dim, R_dim, 1., D_lis_dev, N_align, U_lis_dev, N_align, N_diag, queue);

  magma_free(D_lis_dev);
  magma_free(U_lis_dev);
  magma_free(UD_lis_dev);
  magma_free(UD_data_dev);
  magma_free(info_array);
  free(D_lis);
  free(U_lis);
  free(UD_lis);
}

void schur_diag(int N_diag, double* A_ptr, const int* diag_idx, int R_dim, int S_dim, int N_align) {
  int N_dim = R_dim + S_dim;
  size_t A_stride_mat = (size_t)N_align * N_dim;
  const double** ARS_lis = (const double**)malloc(sizeof(double*) * N_diag), **ARS_lis_dev;
  double** ASS_lis = (double**)malloc(sizeof(double*) * N_diag), **ASS_lis_dev;
  magma_malloc((void**)&ARS_lis_dev, sizeof(double*) * N_diag);
  magma_malloc((void**)&ASS_lis_dev, sizeof(double*) * N_diag);

  for (int i = 0; i < N_diag; i++) {
    int diag = diag_idx[i];
    ARS_lis[i] = A_ptr + A_stride_mat * diag + R_dim;
    ASS_lis[i] = A_ptr + A_stride_mat * diag + ((size_t)N_align * R_dim + R_dim);
  }
  magma_setvector_async(N_diag, sizeof(double*), ARS_lis, 1, ARS_lis_dev, 1, queue);
  magma_setvector_async(N_diag, sizeof(double*), ASS_lis, 1, ASS_lis_dev, 1, queue);
  magmablas_dgemm_batched(MagmaNoTrans, MagmaTrans, S_dim, S_dim, R_dim, -1., ARS_lis_dev, N_align, ARS_lis_dev, N_align, 1., ASS_lis_dev, N_align, N_diag, queue);

  magma_free(ARS_lis_dev);
  magma_free(ASS_lis_dev);
  free(ARS_lis);
  free(ASS_lis);
}

