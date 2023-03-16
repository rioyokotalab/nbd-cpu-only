
#include "nbd.hxx"
#include "profile.hxx"

#include "cuda_runtime_api.h"
#include "cublas_v2.h"
#include "cusolverDn.h"
#include "mkl.h"

#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

cudaStream_t stream = NULL;
cublasHandle_t cublasH = NULL;
cusolverDnHandle_t cusolverH = NULL;

void init_libs(int* argc, char*** argv) {
  assert(MPI_Init(argc, argv) == MPI_SUCCESS);
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  int num_device;
  int gpu_avail = (cudaGetDeviceCount(&num_device) == cudaSuccess);

  if (gpu_avail) {
    int device = mpi_rank % num_device;
    cudaSetDevice(device);
    
    cudaStreamCreate(&stream);
    cublasCreate(&cublasH);
    cublasSetStream(cublasH, stream);

    cusolverDnCreate(&cusolverH);
    cusolverDnSetStream(cusolverH, stream);
  }
}

void fin_libs() {
  if (stream)
    cudaStreamDestroy(stream);
  if (cublasH)
    cublasDestroy(cublasH);
  if (cusolverH)
    cusolverDnDestroy(cusolverH);
  MPI_Finalize();
}

void mat_cpy(int64_t m, int64_t n, const struct Matrix* m1, struct Matrix* m2, int64_t y1, int64_t x1, int64_t y2, int64_t x2) {
  LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', m, n, &m1->A[y1 + x1 * m1->M], m1->LDA, &m2->A[y2 + x2 * m2->M], m2->LDA);
}

void mmult(char ta, char tb, const struct Matrix* A, const struct Matrix* B, struct Matrix* C, double alpha, double beta) {
  int64_t k = ta == 'N' ? A->N : A->M;
  CBLAS_TRANSPOSE tac = ta == 'N' ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE tbc = tb == 'N' ? CblasNoTrans : CblasTrans;
  int64_t lda = 1 < A->LDA ? A->LDA : 1;
  int64_t ldb = 1 < B->LDA ? B->LDA : 1;
  int64_t ldc = 1 < C->LDA ? C->LDA : 1;
  cblas_dgemm(CblasColMajor, tac, tbc, C->M, C->N, k, alpha, A->A, lda, B->A, ldb, beta, C->A, ldc);
}


void svd_U(struct Matrix* A, double* S) {
  int64_t rank_a = A->M < A->N ? A->M : A->N;
  int64_t lda = 1 < A->LDA ? A->LDA : 1;
  int64_t ldv = 1 < A->N ? A->N : 1;
  LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'O', 'N', A->M, A->N, A->A, lda, S, NULL, lda, NULL, ldv, &S[rank_a]);
}

void id_row(struct Matrix* U, struct Matrix* A, int32_t arows[]) {
  int64_t lda = 1 < A->LDA ? A->LDA : 1;
  int64_t ldu = 1 < U->LDA ? U->LDA : 1;
  LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', A->M, A->N, A->A, lda, U->A, ldu);
  LAPACKE_dgetrf(LAPACK_COL_MAJOR, A->M, A->N, A->A, lda, arows);
  cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, A->M, A->N, 1., A->A, lda, U->A, ldu);
  cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit, A->M, A->N, 1., A->A, lda, U->A, ldu);
}

void upper_tri_reflec_mult(char side, int64_t lenR, const struct Matrix* R, struct Matrix* A) {
  int64_t lda = 1 < A->LDA ? A->LDA : 1;
  double work[A->M * A->N];
  LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'F', A->M, A->N, A->A, lda, work, A->M);
  int64_t y = 0;
  if (side == 'L' || side == 'l')
    for (int64_t i = 0; i < lenR; i++) {
      int64_t ldr = 1 < R[i].LDA ? R[i].LDA : 1;
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, R[i].M, A->N, R[i].M, 1., R[i].A, ldr, &work[y], A->M, 0., &A->A[y], lda);
      y = y + R[i].M;
    }
  else if (side == 'R' || side == 'r')
    for (int64_t i = 0; i < lenR; i++) {
      int64_t ldr = 1 < R[i].LDA ? R[i].LDA : 1;
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, A->M, R[i].M, R[i].M, 1., &work[y * A->M], A->M, R[i].A, ldr, 0., &A->A[y * lda], lda);
      y = y + R[i].M;
    }
}

void qr_full(struct Matrix* Q, struct Matrix* R) {
  int64_t ldq = 1 < Q->LDA ? Q->LDA : 1;
  int64_t k = R->N;
  int64_t ldr = 1 < R->LDA ? R->LDA : 1;
  double tau[Q->N];
  LAPACKE_dgeqrf(LAPACK_COL_MAJOR, Q->M, k, Q->A, ldq, tau);
  LAPACKE_dlaset(LAPACK_COL_MAJOR, 'L', R->M, R->N, 0., 0., R->A, ldr);
  LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', R->M, R->N, Q->A, ldq, R->A, ldr);
  LAPACKE_dorgqr(LAPACK_COL_MAJOR, Q->M, Q->N, k, Q->A, ldq, tau);
}

void mat_solve(char type, struct Matrix* X, const struct Matrix* A) {
  int64_t lda = 1 < A->LDA ? A->LDA : 1;
  int64_t ldx = 1 < X->LDA ? X->LDA : 1;
  if (type == 'F' || type == 'f' || type == 'A' || type == 'a')
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, X->M, X->N, 1., A->A, lda, X->A, ldx);
  if (type == 'B' || type == 'b' || type == 'A' || type == 'a')
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit, X->M, X->N, 1., A->A, lda, X->A, ldx);
}

void nrm2_A(struct Matrix* A, double* nrm) {
  int64_t len_A = A->M * A->N;
  double nrm_A = cblas_dnrm2(len_A, A->A, 1);
  *nrm = nrm_A;
}

void scal_A(struct Matrix* A, double alpha) {
  int64_t len_A = A->M * A->N;
  cblas_dscal(len_A, alpha, A->A, 1);
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

int64_t shuffle_batch_dgemm(const double** A, const double** B, double** C, int64_t* sizes, int64_t batch_size) {
  int64_t iters = 0, total_batched = 0;
  while (total_batched < batch_size) {
    int64_t batched = 0;
    for (int64_t i = total_batched; i < batch_size; i++) {
      double* ci = C[i];
      int64_t loc = std::distance(C, std::find(C + total_batched, C + total_batched + batched, ci));
      if (loc == (total_batched + batched)) {
        if (loc < i) {
          std::iter_swap(&A[i], &A[loc]);
          std::iter_swap(&B[i], &B[loc]);
          std::iter_swap(&C[i], &C[loc]);
        }
        batched = batched + 1;
      }
    }

    total_batched = total_batched + batched;
    sizes[iters] = batched;
    iters = iters + 1;
  }
  return iters;
}

void batchParamsCreate(struct BatchedFactorParams* params, int64_t R_dim, int64_t S_dim, const double* U_ptr, double* A_ptr, double* X_ptr, int64_t N_up, double** A_up, double** X_up,
  double* Workspace, int64_t Lwork, int64_t N_rows, int64_t N_cols, int64_t col_offset, const int64_t row_A[], const int64_t col_A[]) {
  
  int64_t N_dim = R_dim + S_dim;
  int64_t NNZ = col_A[N_cols] - col_A[0];
  int64_t stride = N_dim * N_dim;
  int64_t lenB = (Lwork / stride) - N_cols;
  lenB = lenB > NNZ ? NNZ : lenB;
  int64_t N_cols_aligned = 16 * ((N_cols >> 4) + ((N_cols & 15) > 0));
  int64_t NNZ_aligned = 16 * ((NNZ >> 4) + ((NNZ & 15) > 0));

  const int64_t NZ = 19, ND = 10, NH = 6;
  std::vector<double*> ptrs_nnz_cpu(NZ * NNZ_aligned);
  std::vector<double*> ptrs_diag_cpu(ND * N_cols_aligned);
  std::vector<double*> ptrs_host_cpu(NH * NNZ_aligned); 

  const double** _U_r = (const double**)&ptrs_nnz_cpu[0 * NNZ_aligned];
  const double** _U_s = (const double**)&ptrs_nnz_cpu[1 * NNZ_aligned];
  const double** _V_x = (const double**)&ptrs_nnz_cpu[2 * NNZ_aligned];
  const double** _A_sx = (const double**)&ptrs_nnz_cpu[3 * NNZ_aligned];
  double** _A_x = (double**)&ptrs_nnz_cpu[4 * NNZ_aligned];
  double** _B_x = (double**)&ptrs_nnz_cpu[5 * NNZ_aligned];
  double** _A_upper = (double**)&ptrs_nnz_cpu[6 * NNZ_aligned];

  const double** _FwdRR_A = (const double**)&ptrs_nnz_cpu[7 * NNZ_aligned];
  const double** _FwdRR_B = (const double**)&ptrs_nnz_cpu[8 * NNZ_aligned];
  double** _FwdRR_Xc = (double**)&ptrs_nnz_cpu[9 * NNZ_aligned];
  const double** _FwdRS_A = (const double**)&ptrs_nnz_cpu[10 * NNZ_aligned];
  const double** _FwdRS_Xc = (const double**)&ptrs_nnz_cpu[11 * NNZ_aligned];
  double** _FwdRS_Xo = (double**)&ptrs_nnz_cpu[12 * NNZ_aligned];
  const double** _BackRR_A = (const double**)&ptrs_nnz_cpu[13 * NNZ_aligned];
  const double** _BackRR_Xc = (const double**)&ptrs_nnz_cpu[14 * NNZ_aligned];
  double** _BackRR_B = (double**)&ptrs_nnz_cpu[15 * NNZ_aligned];
  const double** _BackRS_A = (const double**)&ptrs_nnz_cpu[16 * NNZ_aligned];
  const double** _BackRS_Xo = (const double**)&ptrs_nnz_cpu[17 * NNZ_aligned];
  double** _BackRS_Xc = (double**)&ptrs_nnz_cpu[18 * NNZ_aligned];

  const double** _A_d = (const double**)&ptrs_diag_cpu[0 * N_cols_aligned];
  const double** _U_d = (const double**)&ptrs_diag_cpu[1 * N_cols_aligned];
  const double** _U_ds = (const double**)&ptrs_diag_cpu[2 * N_cols_aligned];
  const double** _A_rs = (const double**)&ptrs_diag_cpu[3 * N_cols_aligned];
  double** _U_dx = (double**)&ptrs_diag_cpu[4 * N_cols_aligned];
  double** _A_ss = (double**)&ptrs_diag_cpu[5 * N_cols_aligned];
  double** _X_d = (double**)&ptrs_diag_cpu[6 * N_cols_aligned];
  double** _Xc_d = (double**)&ptrs_diag_cpu[7 * N_cols_aligned];
  double** _Xo_d = (double**)&ptrs_diag_cpu[8 * N_cols_aligned];
  double** _B_d = (double**)&ptrs_diag_cpu[9 * N_cols_aligned];

  const double** _A_xlo = (const double**)&ptrs_host_cpu[0 * NNZ_aligned];
  const double** _A_sr = (const double**)&ptrs_host_cpu[1 * NNZ_aligned];
  double** _Xc_y = (double**)&ptrs_host_cpu[2 * NNZ_aligned];
  double** _Xc_x = (double**)&ptrs_host_cpu[3 * NNZ_aligned];
  double** _Xo_y = (double**)&ptrs_host_cpu[4 * NNZ_aligned];
  double** _B_xlo = (double**)&ptrs_host_cpu[5 * NNZ_aligned];

  double* _UD_data = Workspace;
  double* _B_data = &Workspace[N_cols * stride];
  double* _Xc_data;
  cudaMalloc(&_Xc_data, sizeof(double) * N_rows * R_dim);
  cudaMemset(_Xc_data, 0, sizeof(double) * N_rows * R_dim);

  int64_t countL = 0;
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

      if (x + col_offset < y) {
        _A_xlo[countL] = A_ptr + stride * yx;
        _Xc_y[countL] = _Xc_data + R_dim * y;
        _B_xlo[countL] = _UD_data + R_dim * (x + col_offset);
        countL = countL + 1;
      }
      _Xo_y[yx] = X_up[y];
      _Xc_x[yx] = _Xc_data + R_dim * (x + col_offset);
      _A_sr[yx] = A_ptr + stride * yx + R_dim;
    }

    _A_d[x] = A_ptr + stride * diag_id;
    _U_d[x] = U_ptr + stride * (x + col_offset);
    _U_ds[x] = U_ptr + stride * (x + col_offset) + R_dim * N_dim;
    _A_rs[x] = A_ptr + stride * diag_id + R_dim;
    _U_dx[x] = _UD_data + stride * x;
    _A_ss[x] = A_up[diag_id];

    _X_d[x] = X_ptr + N_dim * (x + col_offset);
    _Xc_d[x] = _Xc_data + R_dim * (x + col_offset);
    _Xo_d[x] = X_up[x + col_offset];
    _B_d[x] = _UD_data + R_dim * (x + col_offset);
  }

  for (int64_t x = 0; x < lenB; x++) {
    _B_x[x] = _B_data + stride * x;
    _A_sx[x] = _B_data + stride * x + R_dim * N_dim;
  }
    
  memset((void*)params, 0, sizeof(struct BatchedFactorParams));

  std::vector<int64_t> batch_sizes(N_rows);
  int64_t len;
  len = shuffle_batch_dgemm(_A_xlo, (const double**)_B_xlo, _Xc_y, batch_sizes.data(), countL);
  params->FwdRR_batch.resize(len);
  std::copy(_A_xlo, &_A_xlo[countL], _FwdRR_A);
  std::copy(_B_xlo, &_B_xlo[countL], _FwdRR_B);
  std::copy(_Xc_y, &_Xc_y[countL], _FwdRR_Xc);
  std::copy(batch_sizes.begin(), batch_sizes.begin() + len, params->FwdRR_batch.begin());

  len = shuffle_batch_dgemm(_A_sr, (const double**)_Xc_x, _Xo_y, batch_sizes.data(), NNZ);
  params->FwdRS_batch.resize(len);
  std::copy(_A_sr, &_A_sr[NNZ], _FwdRS_A);
  std::copy(_Xc_x, &_Xc_x[NNZ], _FwdRS_Xc);
  std::copy(_Xo_y, &_Xo_y[NNZ], _FwdRS_Xo);
  std::copy(batch_sizes.begin(), batch_sizes.begin() + len, params->FwdRS_batch.begin());

  len = shuffle_batch_dgemm(_A_xlo, (const double**)_Xc_y, _B_xlo, batch_sizes.data(), countL);
  params->BackRR_batch.resize(len);
  std::copy(_A_xlo, &_A_xlo[countL], _BackRR_A);
  std::copy(_Xc_y, &_Xc_y[countL], _BackRR_Xc);
  std::copy(_B_xlo, &_B_xlo[countL], _BackRR_B);
  std::copy(batch_sizes.begin(), batch_sizes.begin() + len, params->BackRR_batch.begin());

  len = shuffle_batch_dgemm(_A_sr, (const double**)_Xo_y, _Xc_x, batch_sizes.data(), NNZ);
  params->BackRS_batch.resize(len);
  std::copy(_A_sr, &_A_sr[NNZ], _BackRS_A);
  std::copy(_Xo_y, &_Xo_y[NNZ], _BackRS_Xo);
  std::copy(_Xc_x, &_Xc_x[NNZ], _BackRS_Xc);
  std::copy(batch_sizes.begin(), batch_sizes.begin() + len, params->BackRS_batch.begin());

  params->N_r = R_dim;
  params->N_s = S_dim;
  params->N_upper = N_up;
  params->L_diag = N_cols;
  params->L_nnz = NNZ;
  params->L_lower = countL;
  params->L_rows = N_rows;
  params->L_tmp = lenB;

  void** ptrs_nnz, **ptrs_diag;
  cudaMalloc((void**)&ptrs_nnz, sizeof(double*) * NNZ_aligned * NZ);
  cudaMalloc((void**)&ptrs_diag, sizeof(double*) * N_cols_aligned * ND);

  params->U_r = (const double**)&ptrs_nnz[0 * NNZ_aligned];
  params->U_s = (const double**)&ptrs_nnz[1 * NNZ_aligned];
  params->V_x = (const double**)&ptrs_nnz[2 * NNZ_aligned];
  params->A_sx = (const double**)&ptrs_nnz[3 * NNZ_aligned];
  params->A_x = (double**)&ptrs_nnz[4 * NNZ_aligned];
  params->B_x = (double**)&ptrs_nnz[5 * NNZ_aligned];
  params->A_upper = (double**)&ptrs_nnz[6 * NNZ_aligned];

  params->FwdRR_A = (const double**)&ptrs_nnz[7 * NNZ_aligned];
  params->FwdRR_B = (const double**)&ptrs_nnz[8 * NNZ_aligned];
  params->FwdRR_Xc = (double**)&ptrs_nnz[9 * NNZ_aligned];
  params->FwdRS_A = (const double**)&ptrs_nnz[10 * NNZ_aligned];
  params->FwdRS_Xc = (const double**)&ptrs_nnz[11 * NNZ_aligned];
  params->FwdRS_Xo = (double**)&ptrs_nnz[12 * NNZ_aligned];
  params->BackRR_A = (const double**)&ptrs_nnz[13 * NNZ_aligned];
  params->BackRR_Xc = (const double**)&ptrs_nnz[14 * NNZ_aligned];
  params->BackRR_B = (double**)&ptrs_nnz[15 * NNZ_aligned];
  params->BackRS_A = (const double**)&ptrs_nnz[16 * NNZ_aligned];
  params->BackRS_Xo = (const double**)&ptrs_nnz[17 * NNZ_aligned];
  params->BackRS_Xc = (double**)&ptrs_nnz[18 * NNZ_aligned];

  params->A_d = (const double**)&ptrs_diag[0 * N_cols_aligned];
  params->U_d = (const double**)&ptrs_diag[1 * N_cols_aligned];
  params->U_ds = (const double**)&ptrs_diag[2 * N_cols_aligned];
  params->A_rs = (const double**)&ptrs_diag[3 * N_cols_aligned];
  params->U_dx = (double**)&ptrs_diag[4 * N_cols_aligned];
  params->A_ss = (double**)&ptrs_diag[5 * N_cols_aligned];
  params->X_d = (double**)&ptrs_diag[6 * N_cols_aligned];
  params->Xc_d = (double**)&ptrs_diag[7 * N_cols_aligned];
  params->Xo_d = (double**)&ptrs_diag[8 * N_cols_aligned];
  params->B_d = (double**)&ptrs_diag[9 * N_cols_aligned];

  params->U_d0 = U_ptr + stride * col_offset;
  params->Xc_d0 = _Xc_data + R_dim * col_offset;
  params->B_d0 = _UD_data + R_dim * col_offset;
  params->UD_data = _UD_data;
  params->A_data = A_ptr;
  params->B_data = _B_data;
  params->X_data = X_ptr;
  params->Xc_data = _Xc_data;

  cudaMalloc((void**)&(params->info), sizeof(int) * N_cols);
  cudaMemcpy(ptrs_nnz, ptrs_nnz_cpu.data(), sizeof(double*) * NNZ_aligned * NZ, cudaMemcpyHostToDevice);
  cudaMemcpy(ptrs_diag, ptrs_diag_cpu.data(), sizeof(double*) * N_cols_aligned * ND, cudaMemcpyHostToDevice);
}

void batchParamsDestory(struct BatchedFactorParams* params) {
  if (params->A_d)
    cudaFree(params->A_d);
  if (params->U_r)
    cudaFree(params->U_r);
  if (params->info)
    cudaFree(params->info);
  if (params->Xc_data)
    cudaFree(params->Xc_data);
}

void batchCholeskyFactor(struct BatchedFactorParams* params, const struct CellComm* comm) {
  int64_t U = params->N_upper, R = params->N_r, S = params->N_s, N = R + S, D = params->L_diag;
  int64_t alen = N * N * params->L_nnz;
  double one = 1., zero = 0., minus_one = -1.;

#ifdef _PROF
  cudaEvent_t e1, e2;
  cudaEventCreate(&e1);
  cudaEventCreate(&e2);
  cudaEventRecord(e1, stream);
#endif
  level_merge_gpu(params->A_data, alen, stream, comm);
  dup_bcast_gpu(params->A_data, alen, stream, comm);
#ifdef _PROF
  cudaEventRecord(e2, stream);
#endif

  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, R, N, &one, 
    params->A_d, N, params->U_d, N, &zero, params->U_dx, N, D);
  cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, R, R, N, &one, 
    params->U_d, N, (const double**)(params->U_dx), N, &zero, params->B_x, N, D);
  cublasDcopy(cublasH, N * N * D, params->U_d0, 1, params->UD_data, 1);

  cusolverDnDpotrfBatched(cusolverH, CUBLAS_FILL_MODE_LOWER, R, params->B_x, N, params->info, D);
  cublasDtrsmBatched(cublasH, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, 
    N, R, &one, (const double**)(params->B_x), N, params->U_dx, N, D);

  for (int64_t i = 0; i < params->L_nnz; i += params->L_tmp) {
    int64_t len = params->L_nnz - i > params->L_tmp ? params->L_tmp : params->L_nnz - i;
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
#ifdef _PROF
  float time = 0.;
  cudaEventElapsedTime(&time, e1, e2);
  recordCommTime((double)time * 1.e-3);
  cudaEventDestroy(e1);
  cudaEventDestroy(e2);
#endif
}

void batchForwardULV(struct BatchedFactorParams* params, const struct CellComm* comm) {
  int64_t R = params->N_r, S = params->N_s, N = R + S, D = params->L_diag, ONE = 1;
  double one = 1., zero = 0., minus_one = -1.;

#ifdef _PROF
  cudaEvent_t e1, e2;
  cudaEventCreate(&e1);
  cudaEventCreate(&e2);
  cudaEventRecord(e1, stream);
#endif
  level_merge_gpu(params->X_data, params->L_rows * N, stream, comm);
  neighbor_reduce_gpu(params->X_data, N, stream, comm);
  dup_bcast_gpu(params->X_data, params->L_rows * N, stream, comm);
#ifdef _PROF
  cudaEventRecord(e2, stream);
#endif

  cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, R, ONE, N, &one,
    params->U_d, N, (const double**)params->X_d, N, &zero, params->Xc_d, R, D);
  cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, S, ONE, N, &one,
    params->U_ds, N, (const double**)params->X_d, N, &zero, params->Xo_d, S, D);
  cublasDcopy(cublasH, R * D, params->Xc_d0, 1, params->B_d0, 1);
  cublasDtrsmBatched(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 
    R, ONE, &one, params->A_d, N, params->B_d, N, D);

  int64_t row = 0;
  for (int64_t i = 0; i < (int64_t)params->FwdRR_batch.size(); i++) {
    int64_t len = params->FwdRR_batch[i];
    cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, R, ONE, R, &minus_one, 
      &params->FwdRR_A[row], N, &params->FwdRR_B[row], R, &one, &params->FwdRR_Xc[row], R, len);
    row = row + len;
  }

#ifdef _PROF
  cudaEvent_t e3, e4;
  cudaEventCreate(&e3);
  cudaEventCreate(&e4);
  cudaEventRecord(e3, stream);
#endif
  neighbor_reduce_gpu(params->Xc_data, R, stream, comm);
  dup_bcast_gpu(params->Xc_data, params->L_rows * R, stream, comm);
#ifdef _PROF
  cudaEventRecord(e4, stream);
#endif

  cublasDtrsmBatched(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 
    R, ONE, &one, params->A_d, N, params->Xc_d, R, D);

  row = 0;
  for (int64_t i = 0; i < (int64_t)params->FwdRS_batch.size(); i++) {
    int64_t len = params->FwdRS_batch[i];
    cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, S, ONE, R, &minus_one, 
      &params->FwdRS_A[row], N, &params->FwdRS_Xc[row], R, &one, &params->FwdRS_Xo[row], S, len);
    row = row + len;
  }

  cudaStreamSynchronize(stream);
#ifdef _PROF
  float time = 0.;
  cudaEventElapsedTime(&time, e1, e2);
  recordCommTime((double)time * 1.e-3);
  cudaEventElapsedTime(&time, e3, e4);
  recordCommTime((double)time * 1.e-3);
  cudaEventDestroy(e1);
  cudaEventDestroy(e2);
  cudaEventDestroy(e3);
  cudaEventDestroy(e4);
#endif
}

void batchBackwardULV(struct BatchedFactorParams* params, const struct CellComm* comm) {
  int64_t R = params->N_r, S = params->N_s, N = R + S, D = params->L_diag, ONE = 1;
  double one = 1., zero = 0., minus_one = -1.;

  int64_t row = 0;
  for (int64_t i = 0; i < (int64_t)params->BackRS_batch.size(); i++) {
    int64_t len = params->BackRS_batch[i];
    cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, R, ONE, S, &minus_one, 
      &params->BackRS_A[row], N, &params->BackRS_Xo[row], S, &one, &params->BackRS_Xc[row], R, len);
    row = row + len;
  }
  cublasDcopy(cublasH, R * D, params->Xc_d0, 1, params->B_d0, 1);
  cublasDtrsmBatched(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, 
    R, ONE, &one, params->A_d, N, params->Xc_d, R, D);

#ifdef _PROF
  cudaEvent_t e1, e2;
  cudaEventCreate(&e1);
  cudaEventCreate(&e2);
  cudaEventRecord(e1, stream);
#endif
  neighbor_bcast_gpu(params->Xc_data, R, stream, comm);
  dup_bcast_gpu(params->Xc_data, params->L_rows * R, stream, comm);
#ifdef _PROF
  cudaEventRecord(e2, stream);
#endif
  
  row = 0;
  for (int64_t i = 0; i < (int64_t)params->BackRR_batch.size(); i++) {
    int64_t len = params->BackRR_batch[i];
    cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, R, ONE, R, &minus_one, 
      &params->BackRR_A[row], N, &params->BackRR_Xc[row], R, &one, &params->BackRR_B[row], R, len);
    row = row + len;
  }
  cublasDtrsmBatched(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, 
    R, ONE, &one, params->A_d, N, params->B_d, R, D);
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, ONE, R, &one,
    params->U_d, N, (const double**)params->B_d, R, &zero, params->X_d, N, D);
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, ONE, S, &one,
    params->U_ds, N, (const double**)params->Xo_d, S, &one, params->X_d, N, D);
    
#ifdef _PROF
  cudaEvent_t e3, e4;
  cudaEventCreate(&e3);
  cudaEventCreate(&e4);
  cudaEventRecord(e3, stream);
#endif
  neighbor_bcast_gpu(params->X_data, N, stream, comm);
  dup_bcast_gpu(params->X_data, params->L_rows * N, stream, comm);
#ifdef _PROF
  cudaEventRecord(e4, stream);
#endif

  cudaStreamSynchronize(stream);
#ifdef _PROF
  float time = 0.;
  cudaEventElapsedTime(&time, e1, e2);
  recordCommTime((double)time * 1.e-3);
  cudaEventElapsedTime(&time, e3, e4);
  recordCommTime((double)time * 1.e-3);
  cudaEventDestroy(e1);
  cudaEventDestroy(e2);
  cudaEventDestroy(e3);
  cudaEventDestroy(e4);
#endif
}

void lastParamsCreate(struct BatchedFactorParams* params, double* A, double* X, int64_t N) {
  memset((void*)params, 0, sizeof(struct BatchedFactorParams));

  params->A_data = A;
  params->X_data = X;
  params->N_r = N;

  int Lwork;
  cusolverDnDpotrf_bufferSize(cusolverH, CUBLAS_FILL_MODE_LOWER, N, A, N, &Lwork);
  cudaMalloc((void**)&params->Xc_data, sizeof(double) * Lwork);
  cudaMalloc((void**)&params->info, sizeof(int));
  params->L_tmp = Lwork;
}

void chol_decomp(struct BatchedFactorParams* params, const struct CellComm* comm) {
  double* A = params->A_data;
  int64_t N = params->N_r;
  int64_t alen = N * N;

#ifdef _PROF
  cudaEvent_t e1, e2;
  cudaEventCreate(&e1);
  cudaEventCreate(&e2);
  cudaEventRecord(e1, stream);
#endif
  level_merge_gpu(params->A_data, alen, stream, comm);
  dup_bcast_gpu(params->A_data, alen, stream, comm);
#ifdef _PROF
  cudaEventRecord(e2, stream);
#endif

  cusolverDnDpotrf(cusolverH, CUBLAS_FILL_MODE_LOWER, N, A, N, params->Xc_data, params->L_tmp, params->info);
  cudaStreamSynchronize(stream);
#ifdef _PROF
  float time = 0.;
  cudaEventElapsedTime(&time, e1, e2);
  recordCommTime((double)time * 1.e-3);
  cudaEventDestroy(e1);
  cudaEventDestroy(e2);
#endif
}


void chol_solve(struct BatchedFactorParams* params, const struct CellComm* comm) {
  const double* A = params->A_data;
  double* X = params->X_data;
  int64_t N = params->N_r;

#ifdef _PROF
  cudaEvent_t e1, e2;
  cudaEventCreate(&e1);
  cudaEventCreate(&e2);
  cudaEventRecord(e1, stream);
#endif
  level_merge_gpu(X, N, stream, comm);
  dup_bcast_gpu(X, N, stream, comm);
#ifdef _PROF
  cudaEventRecord(e2, stream);
#endif

  cusolverDnDpotrs(cusolverH, CUBLAS_FILL_MODE_LOWER, N, 1, A, N, X, N, params->info);
  cudaStreamSynchronize(stream);
#ifdef _PROF
  float time = 0.;
  cudaEventElapsedTime(&time, e1, e2);
  recordCommTime((double)time * 1.e-3);
  cudaEventDestroy(e1);
  cudaEventDestroy(e2);
#endif
}


