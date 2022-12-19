
#include "nbd.h"
#include "cuda_runtime_api.h"
#include "cublas_v2.h"
#include "cusolverDn.h"
#include "mkl.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

#include <thrust/copy.h>
#include <thrust/tabulate.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/distance.h>
#include <thrust/iterator/permutation_iterator.h>

#include <stdio.h>
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
  const char* env = getenv("PROCS_PER_DEVICE");
  int procs_per_device = env == NULL ? 1 : atoi(env);
  int device = (mpi_rank / procs_per_device) % num_device;
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

struct set_double_ptr {
  double* _A; int64_t _stride;
  set_double_ptr(double* A, int64_t stride) { _A = A; _stride = stride; };
  __host__ __device__ double* operator()(int64_t i) const { return &_A[_stride * i]; }
};

struct set_const_double_ptr {
  const double* _A; int64_t _stride;
  set_const_double_ptr(const double* A, int64_t stride) { _A = A; _stride = stride; };
  __host__ __device__ const double* operator()(int64_t i) const { return &_A[_stride * i]; }
};

struct cmp_int {
  int64_t _offset;
  cmp_int(int64_t offset) { _offset = offset; };
  __host__ __device__ bool operator()(int64_t i) const { return i == _offset; }
};

void batch_cholesky_factor(int64_t R_dim, int64_t S_dim, const double* U_ptr, double* A_ptr, int64_t N_up, double** A_up, 
  int64_t N_rows, int64_t N_cols, int64_t col_offset, const int64_t row_A[], const int64_t col_A[], const int64_t dimr[], const int64_t dims[]) {
  
  int64_t N_dim = R_dim + S_dim;
  int64_t NNZ = col_A[N_cols];
  int64_t stride = N_dim * N_dim;

  double *UD_data, *B_data;
  cudaMalloc((void**)&UD_data, sizeof(double) * N_cols * stride);
  cudaMalloc((void**)&B_data, sizeof(double) * NNZ * stride);

  int* info_array;
  cudaMalloc((void**)&info_array, sizeof(int) * N_cols);

  thrust::host_vector<int64_t> col_coo(NNZ, 0);
  thrust::host_vector<int64_t> diag_idx(N_cols);
  thrust::host_vector<int64_t> diag_fills(0);

  for (int64_t i = 0; i < N_cols; i++) {
    thrust::fill(col_coo.begin() + col_A[i], col_coo.begin() + col_A[i + 1], i);
    diag_idx[i] = thrust::distance(row_A, thrust::find_if(row_A + col_A[i], row_A + col_A[i + 1], cmp_int(i + col_offset)));
    int64_t dimc = dimr[i + col_offset];
    int64_t size_old = diag_fills.size();
    int64_t size_new = size_old + R_dim - dimc;
    diag_fills.resize(size_new);
    thrust::sequence(diag_fills.begin() + size_old, diag_fills.begin() + size_new, i * stride + (N_dim + 1) * dimc, N_dim + 1);
  }

  thrust::device_vector<int64_t> col_arr(NNZ);
  thrust::device_vector<int64_t> row_arr(NNZ);
  thrust::device_vector<int64_t> diag_arr(N_cols);
  thrust::device_vector<int64_t> diag_fill_dev(diag_fills.size());

  thrust::device_vector<double*> A_lis(NNZ);
  thrust::device_vector<double*> B_lis(NNZ);
  thrust::device_vector<double*> UD_lis(N_cols);
  thrust::device_vector<double*> A_up_dev(NNZ);
  thrust::device_vector<const double*> U_lis(NNZ);
  thrust::device_vector<const double*> V_lis(NNZ);

  thrust::copy(col_coo.begin(), col_coo.end(), col_arr.begin());
  thrust::copy(row_A, &row_A[NNZ], row_arr.begin());
  thrust::copy(A_up, &A_up[NNZ], A_up_dev.begin());
  thrust::copy(diag_idx.begin(), diag_idx.end(), diag_arr.begin());
  thrust::copy(diag_fills.begin(), diag_fills.end(), diag_fill_dev.begin());

  thrust::tabulate(thrust::cuda::par.on(stream), B_lis.begin(), B_lis.end(), set_double_ptr(B_data, stride));
  thrust::tabulate(thrust::cuda::par.on(stream), UD_lis.begin(), UD_lis.end(), set_double_ptr(UD_data, stride));
  thrust::tabulate(thrust::cuda::par.on(stream), U_lis.begin(), U_lis.begin() + N_cols, set_const_double_ptr(&U_ptr[stride * col_offset], stride));
  thrust::transform(thrust::cuda::par.on(stream), diag_arr.begin(), diag_arr.end(), A_lis.begin(), set_double_ptr(A_ptr, stride));

  double one = 1., zero = 0., minus_one = -1.;
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N_dim, R_dim, N_dim, &one, 
    (const double**)thrust::raw_pointer_cast(A_lis.data()), N_dim, thrust::raw_pointer_cast(U_lis.data()), N_dim, &zero, thrust::raw_pointer_cast(UD_lis.data()), N_dim, N_cols);
  cublasDgemmStridedBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, R_dim, R_dim, N_dim, &one, 
    &U_ptr[stride * col_offset], N_dim, stride, UD_data, N_dim, stride, &zero, B_data, N_dim, stride, N_cols);
  cublasDcopy(cublasH, stride * N_cols, U_ptr + stride * col_offset, 1, UD_data, 1);
  
  thrust::fill(thrust::cuda::par.on(stream), thrust::make_permutation_iterator(B_data, diag_fill_dev.begin()), 
    thrust::make_permutation_iterator(B_data, diag_fill_dev.end()), 1.);

  cusolverDnDpotrfBatched(cusolverH, CUBLAS_FILL_MODE_LOWER, R_dim, thrust::raw_pointer_cast(B_lis.data()), N_dim, info_array, N_cols);
  cublasDtrsmBatched(cublasH, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, 
    N_dim, R_dim, &one, thrust::raw_pointer_cast(B_lis.data()), N_dim, thrust::raw_pointer_cast(UD_lis.data()), N_dim, N_cols);

  thrust::transform(thrust::cuda::par.on(stream), row_arr.begin(), row_arr.end(), U_lis.begin(), set_const_double_ptr(U_ptr, stride));
  thrust::transform(thrust::cuda::par.on(stream), col_arr.begin(), col_arr.end(), V_lis.begin(), set_const_double_ptr(UD_data, stride));
  thrust::tabulate(thrust::cuda::par.on(stream), A_lis.begin(), A_lis.end(), set_double_ptr(A_ptr, stride));

  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N_dim, N_dim, N_dim, &one, 
    (const double**)thrust::raw_pointer_cast(A_lis.data()), N_dim, thrust::raw_pointer_cast(V_lis.data()), N_dim, &zero, thrust::raw_pointer_cast(B_lis.data()), N_dim, NNZ);
  cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, N_dim, R_dim, N_dim, &one, 
    thrust::raw_pointer_cast(U_lis.data()), N_dim, (const double**)thrust::raw_pointer_cast(B_lis.data()), N_dim, &zero, thrust::raw_pointer_cast(A_lis.data()), N_dim, NNZ);

  thrust::transform(thrust::cuda::par.on(stream), row_arr.begin(), row_arr.end(), U_lis.begin(), set_const_double_ptr(&U_ptr[R_dim * N_dim], stride));
  thrust::tabulate(thrust::cuda::par.on(stream), B_lis.begin(), B_lis.end(), set_double_ptr(&B_data[R_dim * N_dim], stride));

  cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, S_dim, S_dim, N_dim, &one, 
    thrust::raw_pointer_cast(U_lis.data()), N_dim, (const double**)thrust::raw_pointer_cast(B_lis.data()), N_dim, &zero, thrust::raw_pointer_cast(A_up_dev.data()), N_up, NNZ);
  
  thrust::transform(thrust::cuda::par.on(stream), diag_arr.begin(), diag_arr.end(), B_lis.begin(), set_double_ptr(&A_ptr[R_dim], stride));
  thrust::copy(thrust::cuda::par.on(stream), 
    thrust::make_permutation_iterator(A_up_dev.begin(), diag_arr.begin()),
    thrust::make_permutation_iterator(A_up_dev.begin(), diag_arr.end()),
    A_lis.begin());
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, S_dim, S_dim, R_dim, &minus_one, 
    (const double**)thrust::raw_pointer_cast(B_lis.data()), N_dim, (const double**)thrust::raw_pointer_cast(B_lis.data()), N_dim, &one, thrust::raw_pointer_cast(A_lis.data()), N_up, N_cols);

  cudaFree(UD_data);
  cudaFree(B_data);
  cudaFree(info_array);
}

void chol_decomp(double* A, int64_t Nblocks, int64_t block_dim, const int64_t dims[]) {
  int64_t lda = Nblocks * block_dim;
  double* B;
  cudaMalloc((void**)&B, sizeof(double) * lda * lda);
  cudaMemset(B, 0, sizeof(double) * lda * lda);
  int64_t row = 0;
  for (int64_t i = 0; i < Nblocks; i++) {
    int64_t col = 0;
    for (int64_t j = 0; j < Nblocks; j++) {
      cudaMemcpy2D(&B[row + col * lda], sizeof(double) * lda,
        &A[i * block_dim + (j * block_dim * lda)], sizeof(double) * lda, sizeof(double) * dims[i], dims[j], cudaMemcpyDeviceToDevice);
      col = col + dims[j];
    }
    row = row + dims[i];
  }
  cudaMemcpy(A, B, sizeof(double) * lda * lda, cudaMemcpyDeviceToDevice);

  int* info, Lwork;
  cusolverDnDpotrf_bufferSize(cusolverH, CUBLAS_FILL_MODE_LOWER, row, A, lda, &Lwork);
  double* Workspace;
  cudaMalloc((void**)&info, sizeof(int));
  cudaMalloc((void**)&Workspace, sizeof(double) * Lwork);
  cusolverDnDpotrf(cusolverH, CUBLAS_FILL_MODE_LOWER, row, A, lda, Workspace, Lwork, info);
  cudaFree(info);
  cudaFree(Workspace);
  cudaFree(B);
}
