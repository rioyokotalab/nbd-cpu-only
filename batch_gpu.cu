
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
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

cudaStream_t stream = NULL;
cublasHandle_t cublasH = NULL;
cusolverDnHandle_t cusolverH = NULL;
double *D_DATA = NULL;
int64_t D_DATA_SIZE = 0;

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

  size_t free, total;
  cudaMemGetInfo(&free, &total);
  D_DATA_SIZE = (free >> 4) / sizeof(double);
  cudaMalloc((void**)&D_DATA, sizeof(double) * D_DATA_SIZE);
}

void finalize_batch_lib() {
  if (stream)
    cudaStreamDestroy(stream);
  if (cublasH)
    cublasDestroy(cublasH);
  if (cusolverH)
    cusolverDnDestroy(cusolverH);
  if (D_DATA)
    cudaFree(D_DATA);
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

struct set_const_double2_ptr {
  const double* _A, *_B; int64_t _stride;
  set_const_double2_ptr(const double* A, const double* B, int64_t stride) { _A = A; _B = B; _stride = stride; };
  __host__ __device__ thrust::tuple<const double*, const double*> operator()(int64_t i) const 
  { return thrust::make_tuple(&_A[_stride * i], &_B[_stride * i]); }
};

struct set_double4_ptr {
  double* _A, *_B; const double* _C, *_D; int64_t _stride;
  set_double4_ptr(double* A, double* B, const double* C, const double* D, int64_t stride) 
  { _A = A; _B = B; _C = C; _D = D; _stride = stride; };
  __host__ __device__ thrust::tuple<double*, double*, const double*, const double*> operator()(int64_t i) const
  { return thrust::make_tuple(&_A[_stride * i], &_B[_stride * i], &_C[_stride * i], &_D[_stride * i]); }
};

void batch_cholesky_factor(int64_t R_dim, int64_t S_dim, const double* U_ptr, double* A_ptr, int64_t N_up, double** A_up, 
  int64_t N_cols, int64_t col_offset, const int64_t row_A[], const int64_t col_A[], const int64_t dimr[]) {
  
  int64_t N_dim = R_dim + S_dim;
  int64_t NNZ = col_A[N_cols];
  int64_t stride = N_dim * N_dim;

  double *UD_data, *B_data;
  if (N_cols * stride * 2 > D_DATA_SIZE) {
    cudaMalloc((void**)&UD_data, sizeof(double) * N_cols * stride);
    cudaMalloc((void**)&B_data, sizeof(double) * N_cols * stride);
  }
  else {
    UD_data = D_DATA;
    B_data = &D_DATA[N_cols * stride];
  }

  int* info_array;
  cudaMalloc((void**)&info_array, sizeof(int) * N_cols);

  thrust::host_vector<const double*> V_host(NNZ);
  thrust::host_vector<int64_t> diag_host(N_cols);
  thrust::host_vector<int64_t> diag_fill_host(0);
  thrust::host_vector<double*> A_ss_host(N_cols);
  diag_fill_host.reserve(N_cols * R_dim);

  for (int64_t i = 0; i < N_cols; i++) {
    thrust::fill(V_host.begin() + col_A[i], V_host.begin() + col_A[i + 1], &UD_data[i * stride]);
    int64_t diag_idx = thrust::distance(row_A, thrust::find(row_A + col_A[i], row_A + col_A[i + 1], i + col_offset));
    diag_host[i] = diag_idx;
    A_ss_host[i] = A_up[diag_idx];
    int64_t dimc = dimr[i + col_offset];
    int64_t size_old = diag_fill_host.size();
    int64_t size_new = size_old + R_dim - dimc;
    diag_fill_host.resize(size_new);
    thrust::sequence(diag_fill_host.begin() + size_old, diag_fill_host.begin() + size_new, i * stride + (N_dim + 1) * dimc, N_dim + 1);
  }

  thrust::device_vector<int64_t> row_arr(NNZ);
  thrust::device_vector<int64_t> diag_arr(N_cols);

  thrust::device_vector<double*> A_lis(NNZ);
  thrust::device_vector<double*> A_up_dev(NNZ);
  thrust::device_vector<const double*> V_lis(NNZ);
  thrust::device_vector<const double*> U_r(NNZ);
  thrust::device_vector<const double*> U_s(NNZ);

  thrust::device_vector<double*> B_lis(N_cols);
  thrust::device_vector<const double*> D_lis(N_cols);
  thrust::device_vector<double*> UD_lis(N_cols);
  thrust::device_vector<const double*> U_lis(N_cols);
  thrust::device_vector<const double*> A_sx(N_cols);
  thrust::device_vector<const double*> A_sr(N_cols);
  thrust::device_vector<double*> A_ss(N_cols);
  thrust::device_vector<int64_t> diag_fill_dev(diag_fill_host.size());

  thrust::copy(row_A, &row_A[NNZ], row_arr.begin());
  thrust::copy(A_up, &A_up[NNZ], A_up_dev.begin());
  thrust::copy(A_ss_host.begin(), A_ss_host.end(), A_ss.begin());
  thrust::copy(V_host.begin(), V_host.end(), V_lis.begin());
  thrust::copy(diag_host.begin(), diag_host.end(), diag_arr.begin());
  thrust::copy(diag_fill_host.begin(), diag_fill_host.end(), diag_fill_dev.begin());

  thrust::tabulate(A_lis.begin(), A_lis.end(), set_double_ptr(A_ptr, stride));
  auto zip_iter = thrust::make_zip_iterator(thrust::make_tuple(B_lis.begin(), UD_lis.begin(), U_lis.begin(), A_sx.begin()));
  thrust::tabulate(zip_iter, zip_iter + N_cols, set_double4_ptr(B_data, UD_data, &U_ptr[stride * col_offset], &B_data[R_dim * N_dim], stride));

  thrust::transform(row_arr.begin(), row_arr.end(), thrust::make_zip_iterator(thrust::make_tuple(U_r.begin(), U_s.begin())),
    set_const_double2_ptr(U_ptr, &U_ptr[R_dim * N_dim], stride));
  thrust::transform(diag_arr.begin(), diag_arr.end(), thrust::make_zip_iterator(thrust::make_tuple(D_lis.begin(), A_sr.begin())),
    set_const_double2_ptr(A_ptr, &A_ptr[R_dim], stride));

  double one = 1., zero = 0., minus_one = -1.;
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N_dim, R_dim, N_dim, &one, 
    thrust::raw_pointer_cast(D_lis.data()), N_dim, thrust::raw_pointer_cast(U_lis.data()), N_dim, &zero, thrust::raw_pointer_cast(UD_lis.data()), N_dim, N_cols);
  cublasDgemmStridedBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, R_dim, R_dim, N_dim, &one, 
    &U_ptr[stride * col_offset], N_dim, stride, UD_data, N_dim, stride, &zero, B_data, N_dim, stride, N_cols);
  
  thrust::fill(thrust::cuda::par.on(stream), thrust::make_permutation_iterator(B_data, diag_fill_dev.begin()), 
    thrust::make_permutation_iterator(B_data, diag_fill_dev.end()), 1.);
  cublasDcopy(cublasH, stride * N_cols, U_ptr + stride * col_offset, 1, UD_data, 1);
  cusolverDnDpotrfBatched(cusolverH, CUBLAS_FILL_MODE_LOWER, R_dim, thrust::raw_pointer_cast(B_lis.data()), N_dim, info_array, N_cols);
  cublasDtrsmBatched(cublasH, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, 
    N_dim, R_dim, &one, thrust::raw_pointer_cast(B_lis.data()), N_dim, thrust::raw_pointer_cast(UD_lis.data()), N_dim, N_cols);

  for (int64_t i = 0; i < NNZ; i += N_cols) {
    int64_t len = NNZ - i > N_cols ? N_cols : NNZ - i;
    cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N_dim, N_dim, N_dim, &one, 
      (const double**)thrust::raw_pointer_cast(A_lis.data()) + i, N_dim, thrust::raw_pointer_cast(V_lis.data()) + i, N_dim, 
      &zero, thrust::raw_pointer_cast(B_lis.data()), N_dim, len);
    cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, N_dim, R_dim, N_dim, &one, 
      thrust::raw_pointer_cast(U_r.data()) + i, N_dim, (const double**)thrust::raw_pointer_cast(B_lis.data()), N_dim, 
      &zero, thrust::raw_pointer_cast(A_lis.data()) + i, N_dim, len);
    cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, S_dim, S_dim, N_dim, &one, 
      thrust::raw_pointer_cast(U_s.data()) + i, N_dim, thrust::raw_pointer_cast(A_sx.data()), N_dim, 
      &zero, thrust::raw_pointer_cast(A_up_dev.data()) + i, N_up, len);
  }
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, S_dim, S_dim, R_dim, &minus_one, 
    thrust::raw_pointer_cast(A_sr.data()), N_dim, thrust::raw_pointer_cast(A_sr.data()), N_dim, 
    &one, thrust::raw_pointer_cast(A_ss.data()), N_up, N_cols);

  cudaStreamSynchronize(stream);
  if (N_cols * stride * 2 > D_DATA_SIZE) {
    cudaFree(UD_data);
    cudaFree(B_data);
  }
  cudaFree(info_array);
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
        cublasDswap(cublasH, lda - rj, &A[rj * (lda + 1)], 1, &A[arj * lda + rj], 1);
        cublasDswap(cublasH, rj + 1, &A[rj], lda, &A[arj], lda);
      }
    row = row + dims[i];
  }

  int* info, Lwork;
  cusolverDnDpotrf_bufferSize(cusolverH, CUBLAS_FILL_MODE_LOWER, row, A, lda, &Lwork);
  double* Workspace;
  cudaMalloc((void**)&info, sizeof(int));
  cudaMalloc((void**)&Workspace, sizeof(double) * Lwork);
  cusolverDnDpotrf(cusolverH, CUBLAS_FILL_MODE_LOWER, row, A, lda, Workspace, Lwork, info);
  cudaFree(info);
  cudaFree(Workspace);
}
