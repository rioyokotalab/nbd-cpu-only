
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
#include <thrust/scan.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
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

void copy_basis(const double* Ur_in, const double* Us_in, double* U_out, int64_t IR_dim, int64_t IS_dim, int64_t OR_dim, int64_t OS_dim, int64_t ldu_in, int64_t ldu_out) {
  IR_dim = IR_dim < OR_dim ? IR_dim : OR_dim;
  IS_dim = IS_dim < OS_dim ? IS_dim : OS_dim;
  int64_t n_in = IR_dim + IS_dim;
  MKL_Domatcopy('C', 'N', n_in, IR_dim, 1., Ur_in, ldu_in, U_out, ldu_out);
  MKL_Domatcopy('C', 'N', n_in, IS_dim, 1., Us_in, ldu_in, U_out + OR_dim * ldu_out, ldu_out);
}

__global__ void diag_process_kernel(double* D_data, int64_t N_dim, int64_t N) {
  int64_t stride_m = N_dim * N_dim;
  int64_t stride_row = N_dim + 1;
  int64_t rem = N_dim & 3;
  int64_t N_dim_rem = N_dim - rem;

  for (int64_t b = blockIdx.x; b < N; b += gridDim.x) {
    double* data = D_data + stride_m * b;
    double d[4];
    int id[4];

    for (int64_t i = threadIdx.x * 4; i < N_dim_rem; i += blockDim.x * 4) {
      int64_t l0 = i * stride_row;
      int64_t l1 = (i + 1) * stride_row;
      int64_t l2 = (i + 2) * stride_row;
      int64_t l3 = (i + 3) * stride_row;

      d[0] = data[l0];
      d[1] = data[l1];
      d[2] = data[l2];
      d[3] = data[l3];

      id[0] = d[0] == 0.;
      id[1] = d[1] == 0.;
      id[2] = d[2] == 0.;
      id[3] = d[3] == 0.;

      data[l0] = d[0] + id[0];
      data[l1] = d[1] + id[1];
      data[l2] = d[2] + id[2];
      data[l3] = d[3] + id[3];
    }

    if (threadIdx.x < rem) {
      int64_t loc = (N_dim_rem + threadIdx.x) * stride_row; 
      d[0] = data[loc];
      id[0] = d[0] == 0.;
      data[loc] = d[0] + id[0];
    }
  }
}

__global__ void Aup_kernel(const int64_t row_dims[], const int64_t col_dims[], const double* A, int64_t lda, int64_t stride_a, double** B, int64_t ldb, int64_t N) {
  
  for (int64_t i = blockIdx.x; i < N; i += gridDim.x) {
    const double* A_ptr = &A[i * stride_a];
    double* B_ptr = B[i];
    int64_t m = row_dims[i];
    int64_t n = col_dims[i];
    int64_t m8 = m >> 3;
    int64_t iters = m8 * n;
    double data[8];

    for (int64_t j = threadIdx.x; j < iters; j += blockDim.x) {
      int64_t col = j / m8;
      int64_t row = (j - col * m8) << 3;
      int64_t ja = col * lda;
      int64_t jb = col * ldb;

      int64_t i0 = row;
      int64_t i1 = row + 1;
      int64_t i2 = row + 2;
      int64_t i3 = row + 3;
      int64_t i4 = row + 4;
      int64_t i5 = row + 5;
      int64_t i6 = row + 6;
      int64_t i7 = row + 7;

      data[0] = A_ptr[i0 + ja];
      data[1] = A_ptr[i1 + ja];
      data[2] = A_ptr[i2 + ja];
      data[3] = A_ptr[i3 + ja];
      data[4] = A_ptr[i4 + ja];
      data[5] = A_ptr[i5 + ja];
      data[6] = A_ptr[i6 + ja];
      data[7] = A_ptr[i7 + ja];

      B_ptr[i0 + jb] = data[0];
      B_ptr[i1 + jb] = data[1];
      B_ptr[i2 + jb] = data[2];
      B_ptr[i3 + jb] = data[3];
      B_ptr[i4 + jb] = data[4];
      B_ptr[i5 + jb] = data[5];
      B_ptr[i6 + jb] = data[6];
      B_ptr[i7 + jb] = data[7];
    }

    for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
      int64_t ja = j * lda;
      int64_t jb = j * ldb;
      int64_t row = m8 << 3;

      int64_t i0 = row;
      int64_t i1 = row + 1;
      int64_t i2 = row + 2;
      int64_t i3 = row + 3;
      int64_t i4 = row + 4;
      int64_t i5 = row + 5;
      int64_t i6 = row + 6;
      int64_t i7 = row + 7;

      i0 = i0 < m ? i0 : m - 1;
      i1 = i1 < m ? i1 : m - 1;
      i2 = i2 < m ? i2 : m - 1;
      i3 = i3 < m ? i3 : m - 1;
      i4 = i4 < m ? i4 : m - 1;
      i5 = i5 < m ? i5 : m - 1;
      i6 = i6 < m ? i6 : m - 1;
      i7 = i7 < m ? i7 : m - 1;

      data[0] = A_ptr[i0 + ja];
      data[1] = A_ptr[i1 + ja];
      data[2] = A_ptr[i2 + ja];
      data[3] = A_ptr[i3 + ja];
      data[4] = A_ptr[i4 + ja];
      data[5] = A_ptr[i5 + ja];
      data[6] = A_ptr[i6 + ja];
      data[7] = A_ptr[i7 + ja];

      B_ptr[i0 + jb] = data[0];
      B_ptr[i1 + jb] = data[1];
      B_ptr[i2 + jb] = data[2];
      B_ptr[i3 + jb] = data[3];
      B_ptr[i4 + jb] = data[4];
      B_ptr[i5 + jb] = data[5];
      B_ptr[i6 + jb] = data[6];
      B_ptr[i7 + jb] = data[7];
    }
  }
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
  __host__ __device__ int64_t operator()(int64_t i, int64_t j) const { return (int64_t)((i - j) == _offset); }
};

void batch_cholesky_factor(int64_t R_dim, int64_t S_dim, const double* U_ptr, double* A_ptr, int64_t N_up, double** A_up, 
  int64_t N_rows, int64_t N_cols, int64_t col_offset, const int64_t row_A[], const int64_t col_A[], const int64_t dims[]) {
  
  int64_t N_dim = R_dim + S_dim;
  int64_t NNZ = col_A[N_cols];
  int64_t stride = N_dim * N_dim;

  double *UD_data, *B_data;
  cudaMalloc((void**)&UD_data, sizeof(double) * N_cols * stride);
  cudaMalloc((void**)&B_data, sizeof(double) * NNZ * stride);

  int* info_array;
  cudaMalloc((void**)&info_array, sizeof(int) * N_cols);

  thrust::host_vector<int64_t> col_coo(NNZ, 0);
  thrust::host_vector<int64_t> stencil(NNZ);
  thrust::host_vector<int64_t> diag_idx(N_cols);

  thrust::fill(thrust::make_permutation_iterator(col_coo.begin(), col_A + 1),
    thrust::make_permutation_iterator(col_coo.begin(), col_A + N_cols), 
    (int64_t)1);
  thrust::inclusive_scan(col_coo.begin(), col_coo.end(), col_coo.begin());
  thrust::transform(row_A, row_A + NNZ, col_coo.begin(), stencil.begin(), cmp_int(col_offset));
  thrust::counting_iterator<int64_t> it(0);
  thrust::copy_if(it, it + NNZ, stencil.begin(), diag_idx.begin(), thrust::identity<int64_t>());

  thrust::device_vector<int64_t> dims_arr(N_rows);
  thrust::device_vector<int64_t> col_arr(NNZ);
  thrust::device_vector<int64_t> row_arr(NNZ);
  thrust::device_vector<int64_t> col_dims(NNZ);
  thrust::device_vector<int64_t> row_dims(NNZ);
  thrust::device_vector<int64_t> diag_arr(N_cols);

  thrust::device_vector<double*> A_lis(NNZ);
  thrust::device_vector<double*> B_lis(NNZ);
  thrust::device_vector<double*> UD_lis(N_cols);
  thrust::device_vector<double*> A_up_dev(NNZ);
  thrust::device_vector<const double*> U_lis(NNZ);
  thrust::device_vector<const double*> V_lis(NNZ);

  thrust::copy(dims, &dims[N_rows], dims_arr.begin());
  thrust::copy(col_coo.begin(), col_coo.end(), col_arr.begin());
  thrust::copy(row_A, &row_A[NNZ], row_arr.begin());
  thrust::copy(A_up, &A_up[NNZ], A_up_dev.begin());
  thrust::copy(diag_idx.begin(), diag_idx.end(), diag_arr.begin());

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
  
  int grid = N_cols >= 16 ? 16 : N_cols;
  int block = N_dim >= 128 ? 512 : 64;
  diag_process_kernel<<<grid, block, 0, stream>>>(B_data, N_dim, N_cols);
  cusolverDnDpotrfBatched(cusolverH, CUBLAS_FILL_MODE_LOWER, R_dim, thrust::raw_pointer_cast(B_lis.data()), N_dim, info_array, N_cols);
  cublasDtrsmBatched(cublasH, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, 
    N_dim, R_dim, &one, thrust::raw_pointer_cast(B_lis.data()), N_dim, thrust::raw_pointer_cast(UD_lis.data()), N_dim, N_cols);

  thrust::transform(thrust::cuda::par.on(stream), row_arr.begin(), row_arr.end(), U_lis.begin(), set_const_double_ptr(U_ptr, stride));
  thrust::transform(thrust::cuda::par.on(stream), col_arr.begin(), col_arr.end(), V_lis.begin(), set_const_double_ptr(UD_data, stride));
  thrust::tabulate(thrust::cuda::par.on(stream), A_lis.begin(), A_lis.end(), set_double_ptr(A_ptr, stride));

  cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, N_dim, N_dim, N_dim, &one, 
    thrust::raw_pointer_cast(U_lis.data()), N_dim, (const double**)thrust::raw_pointer_cast(A_lis.data()), N_dim, &zero, thrust::raw_pointer_cast(B_lis.data()), N_dim, NNZ);
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N_dim, N_dim, N_dim, &one, 
    (const double**)thrust::raw_pointer_cast(B_lis.data()), N_dim, thrust::raw_pointer_cast(V_lis.data()), N_dim, &zero, thrust::raw_pointer_cast(A_lis.data()), N_dim, NNZ);

  thrust::transform(thrust::cuda::par.on(stream), diag_arr.begin(), diag_arr.end(), B_lis.begin(), set_double_ptr(&A_ptr[R_dim], stride));
  thrust::transform(thrust::cuda::par.on(stream), diag_arr.begin(), diag_arr.end(), A_lis.begin(), set_double_ptr(&A_ptr[R_dim * (N_dim + 1)], stride));

  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, S_dim, S_dim, R_dim, &minus_one, 
    (const double**)thrust::raw_pointer_cast(B_lis.data()), N_dim, (const double**)thrust::raw_pointer_cast(B_lis.data()), N_dim, &one, thrust::raw_pointer_cast(A_lis.data()), N_dim, N_cols);

  thrust::copy(thrust::cuda::par.on(stream), 
    thrust::make_permutation_iterator(dims_arr.begin(), row_arr.begin()),
    thrust::make_permutation_iterator(dims_arr.begin(), row_arr.end()),
    row_dims.begin());
  thrust::copy(thrust::cuda::par.on(stream), 
    thrust::make_permutation_iterator(dims_arr.begin() + col_offset, col_arr.begin()),
    thrust::make_permutation_iterator(dims_arr.begin() + col_offset, col_arr.end()),
    col_dims.begin());

  grid = NNZ >= 32 ? 32 : NNZ;
  block = R_dim >= 64 ? 512 : 128;
  Aup_kernel<<<grid, block, 0, stream>>>(thrust::raw_pointer_cast(row_dims.data()), thrust::raw_pointer_cast(col_dims.data()),
    A_ptr + (N_dim + 1) * R_dim, N_dim, stride, thrust::raw_pointer_cast(A_up_dev.data()), N_up, NNZ);

  cudaFree(UD_data);
  cudaFree(B_data);
  cudaFree(info_array);
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
