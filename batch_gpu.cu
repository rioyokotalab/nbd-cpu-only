
#include "nbd.h"
#include "cuda_runtime_api.h"
#include "cublas_v2.h"
#include "cusolverDn.h"
#include "mkl.h"

#include <stdlib.h>
#include <string.h>

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

void alloc_matrices_aligned(double** A_ptr, double** A_buffer, int64_t M, int64_t N, int64_t count) {
  int64_t stride = M * N;
  int64_t bytes = sizeof(double) * count * stride;
  cudaMalloc((void**)A_ptr, bytes);
  *A_buffer = (double*)malloc(bytes);
  memset((void*)*A_buffer, 0, bytes);
}

void flush_buffer(char dir, double* A_ptr, double* A_buffer, int64_t len) {
  int64_t bytes = sizeof(double) * len;
  if (dir == 'G')
    cudaMemcpy(A_buffer, A_ptr, bytes, cudaMemcpyDeviceToHost);
  else if (dir == 'S')
    cudaMemcpyAsync(A_ptr, A_buffer, bytes, cudaMemcpyHostToDevice, stream);
}

void free_matrices(double* A_ptr, double* A_buffer) {
  cudaFree(A_ptr);
  free(A_buffer);
}

void copy_basis(const double* Ur_in, const double* Us_in, double* U_out, int64_t IR_dim, int64_t IS_dim, int64_t OR_dim, int64_t OS_dim, int64_t ldu_in, int64_t ldu_out) {
  IR_dim = IR_dim < OR_dim ? IR_dim : OR_dim;
  IS_dim = IS_dim < OS_dim ? IS_dim : OS_dim;
  int64_t n_in = IR_dim + IS_dim;
  MKL_Domatcopy('C', 'N', n_in, IR_dim, 1., Ur_in, ldu_in, U_out, ldu_out);
  MKL_Domatcopy('C', 'N', n_in, IS_dim, 1., Us_in, ldu_in, U_out + OR_dim * ldu_out, ldu_out);
}

__global__ void args_kernel(int64_t R_dim, int64_t S_dim, const double* U_ptr, double* A_ptr, int64_t N_cols, int64_t col_offset, const int64_t row_A[], const int64_t col_A[],
  double* UD_data, double* B_data, const double** A_lis_diag, const double** U_lis_diag, const double** U_lis, const double** V_lis, const double** ARS_lis,
  double** D_lis, double** UD_lis, double** A_lis, double** B_lis, double** ASS_lis) {
  
  int64_t N_dim = R_dim + S_dim;
  int64_t stride = N_dim * N_dim;

  for (int64_t x = blockIdx.x; x < N_cols; x += gridDim.x) {
    for (int64_t yx = col_A[x] + threadIdx.x; yx < col_A[x + 1]; yx += blockDim.x) {
      int64_t y = row_A[yx];
      if (x + col_offset == y) {
        A_lis_diag[x] = A_ptr + stride * yx;
        ARS_lis[x] = A_ptr + stride * yx + R_dim;
        ASS_lis[x] = A_ptr + stride * yx + (N_dim + 1) * R_dim;
      }
  
      U_lis[yx] = U_ptr + stride * y;
      V_lis[yx] = UD_data + stride * x;
      A_lis[yx] = A_ptr + stride * yx;
      B_lis[yx] = B_data + stride * yx;
    }
  
    if (threadIdx.x == 0) {
      U_lis_diag[x] = U_ptr + stride * (x + col_offset);
      D_lis[x] = B_data + stride * x;
      UD_lis[x] = UD_data + stride * x;
    }
  }
}

__global__ void diag_process_kernel(double* D_data, int64_t N_dim, int64_t N) {
  int64_t stride_m = N_dim * N_dim;
  int64_t stride_row = N_dim + 1;
  int64_t rem = N_dim & 3;
  int64_t N_dim_rem = N_dim - rem;

  for (int64_t b = blockIdx.x; b < N; b += gridDim.x) {
    double* data = D_data + stride_m * b;
    for (int64_t i = threadIdx.x * 4; i < N_dim_rem; i += blockDim.x * 4) {
      int64_t loc = i * stride_row;
      double d[4];
      d[0] = data[loc];
      d[1] = data[loc + stride_row];
      d[2] = data[loc + 2 * stride_row];
      d[3] = data[loc + 3 * stride_row];
      d[0] = (d[0] == 0.) ? 1. : d[0];
      d[1] = (d[1] == 0.) ? 1. : d[1];
      d[2] = (d[2] == 0.) ? 1. : d[2];
      d[3] = (d[3] == 0.) ? 1. : d[3];
      data[loc] = d[0];
      data[loc + stride_row] = d[1];
      data[loc + 2 * stride_row] = d[2];
      data[loc + 3 * stride_row] = d[3];
    }

    if (threadIdx.x < rem) {
      int64_t loc = (N_dim_rem + threadIdx.x) * stride_row; 
      double d = data[loc];
      d = (d == 0.) ? 1. : d;
      data[loc] = d;
    }
  }
}

__global__ void Aup_kernel(int64_t R_dim, int64_t S_dim, const double* A_ptr, int64_t N_up, double** A_up, int64_t N_cols, int64_t col_offset, 
  const int64_t row_A[], const int64_t col_A[], const int64_t dims[]) {
  
  int64_t N_dim = R_dim + S_dim;
  int64_t stride = N_dim * N_dim;

  for (int64_t x = blockIdx.y; x < N_cols; x += gridDim.y) {
    for (int64_t yx = col_A[x] + blockIdx.x; yx < col_A[x + 1]; yx += gridDim.x) {
      const double* A = &A_ptr[yx * stride + (N_dim + 1) * R_dim];
      double* B = A_up[yx];
      int64_t y = row_A[yx];
      int64_t m = dims[y];
      int64_t n = dims[x + col_offset];
      int64_t rem_m = m & 7;
      int64_t m8 = m - rem_m;
      double data[8];

      for (int64_t j = threadIdx.y; j < n; j += blockDim.y) {
        int64_t ja = j * N_dim;
        int64_t jb = j * N_up;

        for (int64_t i = threadIdx.x * 8; i < m8; i += blockDim.x * 8) {
          int64_t i0 = i;
          int64_t i1 = i + 1;
          int64_t i2 = i + 2;
          int64_t i3 = i + 3;
          int64_t i4 = i + 4;
          int64_t i5 = i + 5;
          int64_t i6 = i + 6;
          int64_t i7 = i + 7;

          data[0] = A[i0 + ja];
          data[1] = A[i1 + ja];
          data[2] = A[i2 + ja];
          data[3] = A[i3 + ja];
          data[4] = A[i4 + ja];
          data[5] = A[i5 + ja];
          data[6] = A[i6 + ja];
          data[7] = A[i7 + ja];

          B[i0 + jb] = data[0];
          B[i1 + jb] = data[1];
          B[i2 + jb] = data[2];
          B[i3 + jb] = data[3];
          B[i4 + jb] = data[4];
          B[i5 + jb] = data[5];
          B[i6 + jb] = data[6];
          B[i7 + jb] = data[7];
        }

        int64_t yy = threadIdx.x + m8;
        if (yy < m)
          B[yy + jb] = A[yy + ja];
      }
    }
  }
}

void batch_cholesky_factor(int64_t R_dim, int64_t S_dim, const double* U_ptr, double* A_ptr, int64_t N_up, double** A_up, 
  int64_t N_rows, int64_t N_cols, int64_t col_offset, const int64_t row_A[], const int64_t col_A[], const int64_t dims[]) {
  
  int64_t N_dim = R_dim + S_dim;
  int64_t NNZ = col_A[N_cols] - col_A[0];
  int64_t stride = N_dim * N_dim;

  const double** A_lis_diag, **U_lis_diag, **U_lis, **V_lis, **ARS_lis;
  cudaMalloc((void**)&A_lis_diag, sizeof(double*) * N_cols);
  cudaMalloc((void**)&U_lis_diag, sizeof(double*) * N_cols);
  cudaMalloc((void**)&U_lis, sizeof(double*) * NNZ);
  cudaMalloc((void**)&V_lis, sizeof(double*) * NNZ);
  cudaMalloc((void**)&ARS_lis, sizeof(double*) * N_cols);

  double** D_lis, **UD_lis, **A_lis, **B_lis, **ASS_lis, **A_up_dev;
  cudaMalloc((void**)&D_lis, sizeof(double*) * N_cols);
  cudaMalloc((void**)&UD_lis, sizeof(double*) * N_cols);
  cudaMalloc((void**)&A_lis, sizeof(double*) * NNZ);
  cudaMalloc((void**)&B_lis, sizeof(double*) * NNZ);
  cudaMalloc((void**)&ASS_lis, sizeof(double*) * N_cols);
  cudaMalloc((void**)&A_up_dev, sizeof(double*) * NNZ);

  double *UD_data, *B_data;
  cudaMalloc((void**)&UD_data, sizeof(double) * N_cols * stride);
  cudaMalloc((void**)&B_data, sizeof(double) * NNZ * stride);

  int* info_array;
  int64_t *col_arr, *row_arr, *dims_arr;
  cudaMalloc((void**)&info_array, sizeof(int) * N_cols);
  cudaMalloc((void**)&col_arr, sizeof(int64_t) * (N_cols + 1));
  cudaMalloc((void**)&row_arr, sizeof(int64_t) * NNZ);
  cudaMalloc((void**)&dims_arr, sizeof(int64_t) * N_rows);
  
  int grid1D = N_cols >= 8 ? 8 : 1;
  int block1D = N_dim >= 128 ? 256 : 64;
  dim3 grid2D(4, N_cols >= 8 ? 8 : 1, 1);
  dim3 block2D(8, N_dim >= 128 ? 32 : 8, 1);

  cudaMemcpyAsync((void*)col_arr, (void*)col_A, sizeof(int64_t) * (N_cols + 1), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync((void*)row_arr, (void*)row_A, sizeof(int64_t) * NNZ, cudaMemcpyHostToDevice, stream);
  args_kernel<<<grid1D, block1D, 0, stream>>>(R_dim, S_dim, U_ptr, A_ptr, N_cols, col_offset, row_arr, col_arr,
    UD_data, B_data, A_lis_diag, U_lis_diag, U_lis, V_lis, ARS_lis, D_lis, UD_lis, A_lis, B_lis, ASS_lis);

  double one = 1., zero = 0., minus_one = -1.;
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N_dim, R_dim, N_dim, &one, 
    A_lis_diag, N_dim, U_lis_diag, N_dim, &zero, UD_lis, N_dim, N_cols);
  cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, R_dim, R_dim, N_dim, &one, 
    U_lis_diag, N_dim, (const double**)UD_lis, N_dim, &zero, D_lis, N_dim, N_cols);
  cublasDcopy(cublasH, stride * N_cols, U_ptr + stride * col_offset, 1, UD_data, 1);

  diag_process_kernel<<<grid1D, block1D, 0, stream>>>(B_data, N_dim, N_cols);
  cusolverDnDpotrfBatched(cusolverH, CUBLAS_FILL_MODE_LOWER, R_dim, D_lis, N_dim, info_array, N_cols);
  cublasDtrsmBatched(cublasH, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, 
    N_dim, R_dim, &one, D_lis, N_dim, UD_lis, N_dim, N_cols);

  cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, N_dim, N_dim, N_dim, &one, 
    U_lis, N_dim, (const double**)A_lis, N_dim, &zero, B_lis, N_dim, NNZ);
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N_dim, N_dim, N_dim, &one, 
    (const double**)B_lis, N_dim, V_lis, N_dim, &zero, A_lis, N_dim, NNZ);
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, S_dim, S_dim, R_dim, &minus_one, 
    ARS_lis, N_dim, ARS_lis, N_dim, &one, ASS_lis, N_dim, N_cols);

  cudaMemcpyAsync((void*)A_up_dev, (void*)A_up, sizeof(double*) * NNZ, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync((void*)dims_arr, (void*)dims, sizeof(int64_t) * N_rows, cudaMemcpyHostToDevice, stream);
  Aup_kernel<<<grid2D, block2D, 0, stream>>>(R_dim, S_dim, A_ptr, N_up, A_up_dev, N_cols, col_offset, row_arr, col_arr, dims_arr);

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
  cudaFree(A_up_dev);

  cudaFree(UD_data);
  cudaFree(B_data);
  cudaFree(info_array);
  cudaFree(col_arr);
  cudaFree(row_arr);
  cudaFree(dims_arr);
}

void chol_decomp(double* A, int64_t N) {
  int* info, Lwork;
  cusolverDnDpotrf_bufferSize(cusolverH, CUBLAS_FILL_MODE_LOWER, N, A, N, &Lwork);
  double* Workspace;
  cudaMalloc((void**)&info, sizeof(int));
  cudaMalloc((void**)&Workspace, sizeof(double) * Lwork);

  diag_process_kernel<<<1, 256, 0, stream>>>(A, N, 1);
  cusolverDnDpotrf(cusolverH, CUBLAS_FILL_MODE_LOWER, N, A, N, Workspace, Lwork, info);

  cudaFree(info);
  cudaFree(Workspace);
}
