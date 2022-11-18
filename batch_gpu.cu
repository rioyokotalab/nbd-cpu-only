
#include "nbd.h"
#include "cuda_runtime_api.h"
#include "cublas_v2.h"
#include "cusolverDn.h"

#include <stdlib.h>

cudaStream_t stream = NULL;
cublasHandle_t cublasH = NULL;
cusolverDnHandle_t cusolverH = NULL;

void init_batch_lib() {
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  int num_device;
  cudaGetDeviceCount(&num_device);
  int device = mpi_rank % num_device;
  cudaSetDevice(device);

  cudaStreamCreate(&stream);
  cublasCreate(&cublasH);
  cublasSetStream(cublasH, stream);

  cusolverDnCreate(&cusolverH);
  cusolverDnSetStream(cusolverH, stream);
}

void finalize_batch_lib() {
  if (stream)
    cudaStreamDestroy(stream);
  if (cublasH)
    cublasDestroy(cublasH);
  if (cusolverH)
    cusolverDnDestroy(cusolverH);
}

void sync_batch_lib() {
  cudaStreamSynchronize(stream);
}

void alloc_matrices_aligned(double** A_ptr, int M, int N, int count) {
  size_t stride = (size_t)M * N;
  size_t bytes = sizeof(double) * count * stride;
  cudaMalloc((void**)A_ptr, bytes);
  cudaMemset(*A_ptr, 0, bytes);
}

void free_matrices(double* A_ptr) {
  cudaFree(A_ptr);
}

void copy_basis(char dir, const double* Ur_in, const double* Us_in, double* U_out, int IR_dim, int IS_dim, int OR_dim, int OS_dim, int ldu_in, int ldu_out) {
  IR_dim = IR_dim < OR_dim ? IR_dim : OR_dim;
  IS_dim = IS_dim < OS_dim ? IS_dim : OS_dim;
  int N_in = IR_dim + IS_dim;

  size_t width = sizeof(double) * N_in;
  size_t dpitch = sizeof(double) * ldu_out;
  size_t spitch = sizeof(double) * ldu_in;

  if (dir == 'G') {
    cudaMemcpy2DAsync(U_out, dpitch, Ur_in, spitch, width, IR_dim, cudaMemcpyDeviceToHost, stream);
    cudaMemcpy2DAsync(U_out + (size_t)OR_dim * ldu_out, dpitch, Us_in, spitch, width, IS_dim, cudaMemcpyDeviceToHost, stream);
  }
  else if (dir == 'S') {
    cudaMemcpy2DAsync(U_out, dpitch, Ur_in, spitch, width, IR_dim, cudaMemcpyHostToDevice, stream);
    cudaMemcpy2DAsync(U_out + (size_t)OR_dim * ldu_out, dpitch, Us_in, spitch, width, IS_dim, cudaMemcpyHostToDevice, stream);
  }
}

void copy_mat(char dir, const double* A_in, double* A_out, int M_in, int N_in, int lda_in, int M_out, int N_out, int lda_out) {
  M_in = M_in < M_out ? M_in : M_out;
  N_in = N_in < N_out ? N_in : N_out;

  size_t width = sizeof(double) * M_in;
  size_t dpitch = sizeof(double) * lda_out;
  size_t spitch = sizeof(double) * lda_in;

  if (dir == 'G')
    cudaMemcpy2DAsync(A_out, dpitch, A_in, spitch, width, N_in, cudaMemcpyDeviceToHost, stream);
  else if (dir == 'S')
    cudaMemcpy2DAsync(A_out, dpitch, A_in, spitch, width, N_in, cudaMemcpyHostToDevice, stream);
}

__global__ void args_kernel(int R_dim, int S_dim, const double* U_ptr, double* A_ptr, int N_cols, int col_offset, const int row_A[], const int col_A[],
  double* UD_data, double* B_data, const double** A_lis_diag, const double** U_lis_diag, const double** U_lis, const double** V_lis, const double** ARS_lis,
  double** D_lis, double** UD_lis, double** A_lis, double** B_lis, double** ASS_lis) {
  
  int N_dim = R_dim + S_dim;
  size_t stride = (size_t)N_dim * N_dim;

  for (int yx = col_A[blockIdx.x] + threadIdx.x; yx < col_A[blockIdx.x + 1]; yx += blockDim.x) {
    int y = row_A[yx];
    if (blockIdx.x + col_offset == y) {
      A_lis_diag[blockIdx.x] = A_ptr + stride * yx;
      ARS_lis[blockIdx.x] = A_ptr + stride * yx + R_dim;
      ASS_lis[blockIdx.x] = A_ptr + stride * yx + (size_t)(N_dim + 1) * R_dim;
    }

    U_lis[yx] = U_ptr + stride * y;
    V_lis[yx] = UD_data + stride * blockIdx.x;
    A_lis[yx] = A_ptr + stride * yx;
    B_lis[yx] = B_data + stride * yx;
  }

  if (threadIdx.x == 0) {
    U_lis_diag[blockIdx.x] = U_ptr + stride * (blockIdx.x + col_offset);
    D_lis[blockIdx.x] = B_data + stride * blockIdx.x;
    UD_lis[blockIdx.x] = UD_data + stride * blockIdx.x;
  }
}

__global__ void diag_process_kernel(double* D_data) {
  int stride_m = blockDim.x * blockDim.x;
  int stride_row = blockDim.x + 1;
  double* data = D_data + stride_m * blockIdx.x + stride_row * threadIdx.x;
  if (*data == 0.)
    *data = 1.;
}

void batch_cholesky_factor(int R_dim, int S_dim, const double* U_ptr, double* A_ptr, int N_cols, int col_offset, const int row_A[], const int col_A[]) {
  int N_dim = R_dim + S_dim;
  int NNZ = col_A[N_cols] - col_A[0];
  size_t stride = (size_t)N_dim * N_dim;

  const double** A_lis_diag, **U_lis_diag, **U_lis, **V_lis, **ARS_lis;
  cudaMalloc((void**)&A_lis_diag, sizeof(double*) * N_cols);
  cudaMalloc((void**)&U_lis_diag, sizeof(double*) * N_cols);
  cudaMalloc((void**)&U_lis, sizeof(double*) * NNZ);
  cudaMalloc((void**)&V_lis, sizeof(double*) * NNZ);
  cudaMalloc((void**)&ARS_lis, sizeof(double*) * N_cols);

  double** D_lis, **UD_lis, **A_lis, **B_lis, **ASS_lis;
  cudaMalloc((void**)&D_lis, sizeof(double*) * N_cols);
  cudaMalloc((void**)&UD_lis, sizeof(double*) * N_cols);
  cudaMalloc((void**)&A_lis, sizeof(double*) * NNZ);
  cudaMalloc((void**)&B_lis, sizeof(double*) * NNZ);
  cudaMalloc((void**)&ASS_lis, sizeof(double*) * N_cols);

  double *UD_data, *B_data;
  cudaMalloc((void**)&UD_data, sizeof(double) * N_cols * stride);
  cudaMalloc((void**)&B_data, sizeof(double) * NNZ * stride);

  int* info_array, *row_arr;
  cudaMalloc((void**)&info_array, sizeof(int) * (N_cols + 1));
  cudaMalloc((void**)&row_arr, sizeof(int) * NNZ);

  cudaMemcpyAsync((void*)info_array, (void*)col_A, sizeof(int) * (N_cols + 1), cudaMemcpyHostToDevice);
  cudaMemcpyAsync((void*)row_arr, (void*)row_A, sizeof(int) * NNZ, cudaMemcpyHostToDevice);
  args_kernel<<<N_cols, 256, 0, stream>>>(R_dim, S_dim, U_ptr, A_ptr, N_cols, col_offset, row_arr, info_array,
    UD_data, B_data, A_lis_diag, U_lis_diag, U_lis, V_lis, ARS_lis, D_lis, UD_lis, A_lis, B_lis, ASS_lis);

  double one = 1., zero = 0., minus_one = -1.;
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N_dim, R_dim, N_dim, &one, 
    A_lis_diag, N_dim, U_lis_diag, N_dim, &zero, UD_lis, N_dim, N_cols);
  cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, R_dim, R_dim, N_dim, &one, 
    U_lis_diag, N_dim, (const double**)UD_lis, N_dim, &zero, D_lis, N_dim, N_cols);
  cublasDcopy(cublasH, stride * N_cols, U_ptr + stride * col_offset, 1, UD_data, 1);

  diag_process_kernel<<<N_cols, N_dim, 0, stream>>>(B_data);
  cusolverDnDpotrfBatched(cusolverH, CUBLAS_FILL_MODE_LOWER, R_dim, D_lis, N_dim, info_array, N_cols);
  cublasDtrsmBatched(cublasH, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, 
    N_dim, R_dim, &one, D_lis, N_dim, UD_lis, N_dim, N_cols);

  cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, N_dim, N_dim, N_dim, &one, 
    U_lis, N_dim, (const double**)A_lis, N_dim, &zero, B_lis, N_dim, NNZ);
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N_dim, N_dim, N_dim, &one, 
    (const double**)B_lis, N_dim, V_lis, N_dim, &zero, A_lis, N_dim, NNZ);
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, S_dim, S_dim, R_dim, &minus_one, 
    ARS_lis, N_dim, ARS_lis, N_dim, &one, ASS_lis, N_dim, N_cols);

  cudaFree(A_lis_diag);
  cudaFree(U_lis_diag);
  cudaFree(U_lis);
  cudaFree(V_lis);
  cudaFree(ARS_lis);

  cudaFree(D_lis);
  cudaFree(UD_lis);
  cudaFree(A_lis);
  cudaFree(B_lis);
  cudaFree(ASS_lis);

  cudaFree(UD_data);
  cudaFree(B_data);
  cudaFree(info_array);
  cudaFree(row_arr);
}


