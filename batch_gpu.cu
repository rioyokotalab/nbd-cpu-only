
#include "nbd.h"
#include "cuda_runtime_api.h"
#include "cublas_v2.h"
#include "cusolverDn.h"
#include "mkl.h"

#include <thrust/functional.h>
#include <thrust/copy.h>
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
  stream = NULL;
  cublasH = NULL;
  cusolverH = NULL;
}

void set_work_size(int64_t Lwork, double** D_DATA, int64_t* D_DATA_SIZE) {
  if (Lwork > *D_DATA_SIZE) {
    *D_DATA_SIZE = Lwork;
    if (*D_DATA)
      cudaFree(*D_DATA);
    cudaMalloc((void**)D_DATA, sizeof(double) * Lwork);
  }
  else if (Lwork <= 0) {
    *D_DATA_SIZE = 0;
    if (*D_DATA)
      cudaFree(*D_DATA);
  }
}

void allocBufferedList(void** A_ptr, void** A_buffer, int64_t element_size, int64_t count) {
  int64_t bytes = element_size * count;
  cudaMalloc((void**)A_ptr, bytes);
  *A_buffer = malloc(bytes);
  memset((void*)*A_buffer, 0, bytes);
}

void flushBuffer(char dir, void* A_ptr, void* A_buffer, int64_t element_size, int64_t count) {
  int64_t bytes = element_size * count;
  if (dir == 'G' || dir == 'g')
    cudaMemcpy(A_buffer, A_ptr, bytes, cudaMemcpyDeviceToHost);
  else if (dir == 'S' || dir == 's')
    cudaMemcpy(A_ptr, A_buffer, bytes, cudaMemcpyHostToDevice);
}

void freeBufferedList(void* A_ptr, void* A_buffer) {
  cudaFree(A_ptr);
  free(A_buffer);
}

struct BatchedFactorParams { 
  int64_t N_r, N_s, N_upper, L_diag, L_nnz, L_fill, *F_d;
  const double** A_d, **U_d, **U_r, **U_s, **V_x, **A_rs, **A_sx, *U_d0;
  double** U_dx, **A_x, **B_x, **A_ss, **A_upper, *UD_data, *B_data;
  int* info;
};

void batchParamsCreate(void** params, int64_t R_dim, int64_t S_dim, const double* U_ptr, double* A_ptr, int64_t N_up, double** A_up, double* Workspace,
  int64_t N_cols, int64_t col_offset, const int64_t row_A[], const int64_t col_A[], const int64_t dimr[]) {
  
  int64_t N_dim = R_dim + S_dim;
  int64_t NNZ = col_A[N_cols] - col_A[0];
  int64_t stride = N_dim * N_dim;

  const double** _A_d, **_U_d, **_U_r, **_U_s, **_V_x, **_A_rs, **_A_sx;
  double** _U_dx, **_A_x, **_B_x, **_A_ss, **_A_upper;
  _A_d = (const double**)malloc(sizeof(double*) * N_cols);
  _U_d = (const double**)malloc(sizeof(double*) * N_cols);
  _U_r = (const double**)malloc(sizeof(double*) * NNZ);
  _U_s = (const double**)malloc(sizeof(double*) * NNZ);
  _V_x = (const double**)malloc(sizeof(double*) * NNZ);
  _A_rs = (const double**)malloc(sizeof(double*) * N_cols);
  _A_sx = (const double**)malloc(sizeof(double*) * N_cols);

  _U_dx = (double**)malloc(sizeof(double*) * N_cols);
  _A_x = (double**)malloc(sizeof(double*) * NNZ);
  _B_x = (double**)malloc(sizeof(double*) * N_cols);
  _A_ss = (double**)malloc(sizeof(double*) * N_cols);
  _A_upper = (double**)malloc(sizeof(double*) * NNZ);

  double* _UD_data = Workspace;
  double* _B_data = &Workspace[N_cols * stride];
  const double* _U_d0 = U_ptr + stride * col_offset;
  int64_t* _F_d = (int64_t*)malloc(sizeof(int64_t) * N_cols * R_dim);
  int64_t _F_len = 0;

  for (int64_t x = 0; x < N_cols; x++) {
    int64_t diag_id = 0;
    for (int64_t yx = col_A[x]; yx < col_A[x + 1]; yx++) {
      int64_t y = row_A[yx];
      if (x + col_offset == y)
        diag_id = yx;
      _U_r[yx] = U_ptr + stride * y;
      _U_s[yx] = U_ptr + stride * y + R_dim * N_dim;
      _V_x[yx] = _UD_data + stride * x;
      _A_x[yx] = A_ptr + stride * yx;
      _A_upper[yx] = A_up[yx];
    }

    _A_d[x] = A_ptr + stride * diag_id;
    _B_x[x] = _B_data + stride * x;
    _U_d[x] = U_ptr + stride * (x + col_offset);
    _A_rs[x] = A_ptr + stride * diag_id + R_dim;
    _U_dx[x] = _UD_data + stride * x;
    _A_ss[x] = A_up[diag_id];
    _A_sx[x] = _B_data + stride * x + R_dim * N_dim;

    int64_t dimc = dimr[x + col_offset];
    int64_t fill_new = R_dim - dimc;
    for (int64_t i = 0; i < fill_new; i++)
      _F_d[_F_len + i] = x * stride + (N_dim + 1) * (dimc + i);
    _F_len = _F_len + fill_new;
  }
  
  struct BatchedFactorParams* params_ptr = (struct BatchedFactorParams*)malloc(sizeof(struct BatchedFactorParams));
  params_ptr->N_r = R_dim;
  params_ptr->N_s = S_dim;
  params_ptr->N_upper = N_up;
  params_ptr->L_diag = N_cols;
  params_ptr->L_nnz = NNZ;
  params_ptr->L_fill = _F_len;
  cudaMalloc((void**)&(params_ptr->F_d), sizeof(int64_t) * N_cols * R_dim);

  cudaMalloc((void**)&(params_ptr->A_d), sizeof(double*) * N_cols);
  cudaMalloc((void**)&(params_ptr->U_d), sizeof(double*) * N_cols);
  cudaMalloc((void**)&(params_ptr->U_r), sizeof(double*) * NNZ);
  cudaMalloc((void**)&(params_ptr->U_s), sizeof(double*) * NNZ);
  cudaMalloc((void**)&(params_ptr->V_x), sizeof(double*) * NNZ);
  cudaMalloc((void**)&(params_ptr->A_rs), sizeof(double*) * N_cols);
  cudaMalloc((void**)&(params_ptr->A_sx), sizeof(double*) * N_cols);

  cudaMalloc((void**)&(params_ptr->U_dx), sizeof(double*) * N_cols);
  cudaMalloc((void**)&(params_ptr->A_x), sizeof(double*) * NNZ);
  cudaMalloc((void**)&(params_ptr->B_x), sizeof(double*) * N_cols);
  cudaMalloc((void**)&(params_ptr->A_ss), sizeof(double*) * N_cols);
  cudaMalloc((void**)&(params_ptr->A_upper), sizeof(double*) * NNZ);

  params_ptr->U_d0 = _U_d0;
  params_ptr->UD_data = _UD_data;
  params_ptr->B_data = _B_data;

  cudaMalloc((void**)&(params_ptr->info), sizeof(int) * N_cols);
  *params = params_ptr;

  cudaMemcpy(params_ptr->F_d, _F_d, sizeof(int64_t) * N_cols * R_dim, cudaMemcpyHostToDevice);
  cudaMemcpy(params_ptr->A_d, _A_d, sizeof(double*) * N_cols, cudaMemcpyHostToDevice);
  cudaMemcpy(params_ptr->U_d, _U_d, sizeof(double*) * N_cols, cudaMemcpyHostToDevice);
  cudaMemcpy(params_ptr->U_r, _U_r, sizeof(double*) * NNZ, cudaMemcpyHostToDevice);
  cudaMemcpy(params_ptr->U_s, _U_s, sizeof(double*) * NNZ, cudaMemcpyHostToDevice);
  cudaMemcpy(params_ptr->V_x, _V_x, sizeof(double*) * NNZ, cudaMemcpyHostToDevice);
  cudaMemcpy(params_ptr->A_rs, _A_rs, sizeof(double*) * N_cols, cudaMemcpyHostToDevice);
  cudaMemcpy(params_ptr->A_sx, _A_sx, sizeof(double*) * N_cols, cudaMemcpyHostToDevice);

  cudaMemcpy(params_ptr->U_dx, _U_dx, sizeof(double*) * N_cols, cudaMemcpyHostToDevice);
  cudaMemcpy(params_ptr->A_x, _A_x, sizeof(double*) * NNZ, cudaMemcpyHostToDevice);
  cudaMemcpy(params_ptr->B_x, _B_x, sizeof(double*) * N_cols, cudaMemcpyHostToDevice);
  cudaMemcpy(params_ptr->A_ss, _A_ss, sizeof(double*) * N_cols, cudaMemcpyHostToDevice);
  cudaMemcpy(params_ptr->A_upper, _A_upper, sizeof(double*) * NNZ, cudaMemcpyHostToDevice);

  free(_F_d);
  free(_A_d);
  free(_U_d);
  free(_U_r);
  free(_U_s);
  free(_V_x);
  free(_A_rs);
  free(_A_sx);
  free(_U_dx);
  free(_A_x);
  free(_B_x);
  free(_A_ss);
  free(_A_upper);
}

void batchParamsDestory(void* params) {
  struct BatchedFactorParams* params_ptr = (struct BatchedFactorParams*)params;
  if (params_ptr->F_d)
    cudaFree(params_ptr->F_d);
  if (params_ptr->A_d)
    cudaFree(params_ptr->A_d);
  if (params_ptr->U_d)
    cudaFree(params_ptr->U_d);
  if (params_ptr->U_r)
    cudaFree(params_ptr->U_r);
  if (params_ptr->U_s)
    cudaFree(params_ptr->U_s);
  if (params_ptr->V_x)
    cudaFree(params_ptr->V_x);
  if (params_ptr->A_rs)
    cudaFree(params_ptr->A_rs);
  if (params_ptr->A_sx)
    cudaFree(params_ptr->A_sx);
  if (params_ptr->U_dx)
    cudaFree(params_ptr->U_dx);
  if (params_ptr->A_x)
    cudaFree(params_ptr->A_x);
  if (params_ptr->B_x)
    cudaFree(params_ptr->B_x);
  if (params_ptr->A_ss)
    cudaFree(params_ptr->A_ss);
  if (params_ptr->A_upper)
    cudaFree(params_ptr->A_upper);
  if (params_ptr->info)
    cudaFree(params_ptr->info);

  free(params);
}

void batchCholeskyFactor(void* params_ptr) {
  struct BatchedFactorParams* params = (struct BatchedFactorParams*)params_ptr;
  int64_t U = params->N_upper, R = params->N_r, S = params->N_s, N = R + S, D = params->L_diag;
  double one = 1., zero = 0., minus_one = -1.;

  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, R, N, &one, 
    params->A_d, N, params->U_d, N, &zero, params->U_dx, N, D);
  cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, R, R, N, &one, 
    params->U_d, N, (const double**)(params->U_dx), N, &zero, params->B_x, N, D);
  cublasDcopy(cublasH, N * N * D, params->U_d0, 1, params->UD_data, 1);
  thrust::fill(thrust::cuda::par.on(stream), thrust::make_permutation_iterator(params->B_data, params->F_d), 
    thrust::make_permutation_iterator(params->B_data, params->F_d + params->L_fill), 1.);

  cusolverDnDpotrfBatched(cusolverH, CUBLAS_FILL_MODE_LOWER, R, params->B_x, N, params->info, D);
  cublasDtrsmBatched(cublasH, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, 
    N, R, &one, (const double**)(params->B_x), N, params->U_dx, N, D);

  for (int64_t i = 0; i < params->L_nnz; i += D) {
    int64_t len = params->L_nnz - i > D ? D : params->L_nnz - i;
    cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, 
      (const double**)(&params->A_x[i]), N, &params->V_x[i], N, &zero, params->B_x, N, len);
    cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, N, R, N, &one, 
      &params->U_r[i], N, (const double**)(params->B_x), N, &zero, &params->A_x[i], N, len);
    cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, S, S, N, &one, 
      &params->U_s[i], N, params->A_sx, N, &zero, &params->A_upper[i], U, len);
  }
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, S, S, R, &minus_one, 
    params->A_rs, N, params->A_rs, N, &one, params->A_ss, U, D);
  
  cudaStreamSynchronize(stream);
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
  double* Workspace;
  cusolverDnDpotrf_bufferSize(cusolverH, CUBLAS_FILL_MODE_LOWER, row, A, lda, &Lwork);
  cudaMalloc((void**)&Workspace, sizeof(double) * Lwork);
  cudaMalloc((void**)&info, sizeof(int));
  cusolverDnDpotrf(cusolverH, CUBLAS_FILL_MODE_LOWER, row, A, lda, Workspace, Lwork, info);
  cudaStreamSynchronize(stream);
  cudaFree(Workspace);
  cudaFree(info);
}
