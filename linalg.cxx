
#include "nbd.hxx"

#include "cuda_runtime_api.h"
#include "cublas_v2.h"
#include "cusolverDn.h"
#include "mkl.h"

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

cudaStream_t stream = NULL;
cublasHandle_t cublasH = NULL;
cusolverDnHandle_t cusolverH = NULL;

cudaStream_t init_libs(int* argc, char*** argv) {
  if (MPI_Init(argc, argv) != MPI_SUCCESS)
    fprintf(stderr, "MPI Init Error\n");
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
  return stream;
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

void mmult(char ta, char tb, const struct Matrix* A, const struct Matrix* B, struct Matrix* C, double alpha, double beta) {
  int64_t k = ta == 'N' ? A->N : A->M;
  CBLAS_TRANSPOSE tac = ta == 'N' ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE tbc = tb == 'N' ? CblasNoTrans : CblasTrans;
  int64_t lda = 1 < A->LDA ? A->LDA : 1;
  int64_t ldb = 1 < B->LDA ? B->LDA : 1;
  int64_t ldc = 1 < C->LDA ? C->LDA : 1;
  cblas_dgemm(CblasColMajor, tac, tbc, C->M, C->N, k, alpha, A->A, lda, B->A, ldb, beta, C->A, ldc);
}

void mul_AS(const struct Matrix* RU, const struct Matrix* RV, struct Matrix* A) {
  if (A->M > 0 && A->N > 0) {
    std::vector<double> tmp(A->M * A->N);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A->M, A->N, A->M, 1., RU->A, RU->LDA, A->A, A->LDA, 0., &tmp[0], A->M);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, A->M, A->N, A->N, 1., &tmp[0], A->M, RV->A, RV->LDA, 0., A->A, A->LDA);
  }
}

int64_t compute_basis(const EvalDouble& eval, double epi, int64_t rank_min, int64_t rank_max, 
  int64_t M, double* A, int64_t LDA, double Xbodies[], int64_t Nclose, const double Cbodies[], int64_t Nfar, const double Fbodies[]) {

  if (M > 0 && (Nclose > 0 || Nfar > 0)) {
    int64_t ldm = std::max(M, Nclose + Nfar);
    std::vector<double> Aall(M * ldm, 0.), U(M * M), S(M * 2);
    std::vector<MKL_INT> ipiv(M);
    gen_matrix(eval, Nclose, M, Cbodies, Xbodies, &Aall[0], ldm);
    gen_matrix(eval, Nfar, M, Fbodies, Xbodies, &Aall[Nclose], ldm);

    for (int64_t i = 0; i < Nclose; i += M) {
      int64_t len = std::min(M, Nclose - i);
      gen_matrix(eval, len, len, &Cbodies[i * 3], &Cbodies[i * 3], &U[0], M);
      LAPACKE_dgesv(LAPACK_COL_MAJOR, len, M, &U[0], M, &ipiv[0], &Aall[i], ldm);
    }

    LAPACKE_dgetrf(LAPACK_COL_MAJOR, Nclose + Nfar, M, &Aall[0], ldm, &ipiv[0]);
    LAPACKE_dlaset(LAPACK_COL_MAJOR, 'L', M - 1, M - 1, 0., 0., &Aall[1], ldm);

    mkl_domatcopy('C', 'T', M, M, 1., &Aall[0], ldm, &U[0], M);
    LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'O', 'N', M, M, &U[0], M, &S[0], NULL, M, NULL, M, &S[M]);

    double s0 = S[0] * epi;
    rank_max = rank_max <= 0 ? M : std::min(rank_max, M);
    rank_min = rank_min <= 0 ? 0 : std::min(rank_min, M);
    int64_t rank = epi > 0. ?
      std::distance(S.begin(), std::find_if(S.begin() + rank_min, S.begin() + rank_max, [s0](double& s) { return s < s0; })) : rank_max;

    if (rank > 0) {
      LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'F', M, rank, &U[0], M, &Aall[0], M);
      LAPACKE_dgetrf(LAPACK_COL_MAJOR, M, rank, &U[0], M, &ipiv[0]);
      cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, M, rank, 1., &U[0], M, &Aall[0], M);
      cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit, M, rank, 1., &U[0], M, &Aall[0], M);

      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, rank, M, 1., A, LDA, &Aall[0], M, 0., &U[0], M);
      LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'O', M, rank, &U[0], M, &S[0], A, LDA, &U[0], M, &S[M]);

      for (MKL_INT i = 0; i < rank; i++) {
        MKL_INT piv = ipiv[i] - 1;
        if (piv != i)
          cblas_dswap(3, &Xbodies[i * 3], 1, &Xbodies[piv * 3], 1);
        vdMul(rank, &S[0], &U[i * M], &A[(M + i) * LDA]);
      }
    }
    return rank;
  }
  return 0;
}


void mat_vec_reference(const EvalDouble& eval, int64_t begin, int64_t end, double B[], int64_t nbodies, const double* bodies, const double Xbodies[]) {
  int64_t M = end - begin;
  int64_t N = nbodies;
  int64_t size = 1024;
  std::vector<double> A(size * size);
  
  for (int64_t i = 0; i < M; i += size) {
    int64_t y = begin + i;
    int64_t m = std::min(M - i, size);
    const double* bi = &bodies[y * 3];
    for (int64_t j = 0; j < N; j += size) {
      const double* bj = &bodies[j * 3];
      int64_t n = std::min(N - j, size);
      gen_matrix(eval, m, n, bi, bj, &A[0], size);
      cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, 1., &A[0], size, &Xbodies[j], 1, 1., &B[i], 1);
    }
  }
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

int64_t partition_DLU(int64_t row_coords[], int64_t col_coords[], int64_t orders[], int64_t N_cols, int64_t col_offset, const int64_t row_A[], const int64_t col_A[]) {
  int64_t NNZ = col_A[N_cols] - col_A[0];
  std::vector<std::tuple<int64_t, int64_t, int64_t>> coo_list(NNZ);
  std::iota(orders, &orders[NNZ], 0);
  for (int64_t x = 0; x < N_cols; x++) {
    int64_t begin = col_A[x] - col_A[0];
    int64_t end = col_A[x + 1] - col_A[0];
    std::transform(row_A + begin, row_A + end, orders + begin, coo_list.begin() + begin, 
      [x, col_offset](int64_t y, int64_t yx) -> std::tuple<int64_t, int64_t, int64_t> { return std::make_tuple(y, x + col_offset, yx); });
  }

  auto iter = std::stable_partition(coo_list.begin(), coo_list.end(), 
    [](std::tuple<int64_t, int64_t, int64_t> i) -> bool { return std::get<0>(i) == std::get<1>(i); });
  auto iterL = std::stable_partition(iter, coo_list.end(),
    [](std::tuple<int64_t, int64_t, int64_t> i) -> bool { return std::get<0>(i) > std::get<1>(i); });

  std::transform(coo_list.begin(), coo_list.end(), row_coords,
    [](std::tuple<int64_t, int64_t, int64_t> i) -> int64_t { return std::get<0>(i); });
  std::transform(coo_list.begin(), coo_list.end(), col_coords, 
    [](std::tuple<int64_t, int64_t, int64_t> i) -> int64_t { return std::get<1>(i); });
  std::transform(coo_list.begin(), coo_list.end(), orders, 
    [](std::tuple<int64_t, int64_t, int64_t> i) -> int64_t { return std::get<2>(i); });
  return std::distance(iter, iterL);
}

int64_t count_apperance_x(const int64_t X[], int64_t AX[], int64_t lenX) {
  std::pair<const int64_t*, const int64_t*> minmax_e = std::minmax_element(X, &X[lenX]);
  int64_t min_e = *std::get<0>(minmax_e);
  int64_t max_e = *std::get<1>(minmax_e);
  std::vector<int64_t> count(max_e - min_e + 1, 0);
  for (int64_t i = 0; i < lenX; i++) {
    int64_t x = X[i] - min_e;
    int64_t c = count[x];
    AX[i] = c;
    count[x] = c + 1;
  }
  return *std::max_element(count.begin(), count.end());
}

void batchParamsCreate(struct BatchedFactorParams* params, int64_t R_dim, int64_t S_dim, const double* U_ptr, double* A_ptr, double* X_ptr, int64_t N_up, double** A_up, double** X_up,
  double* Workspace, int64_t Lwork, int64_t N_rows, int64_t N_cols, int64_t col_offset, const int64_t row_A[], const int64_t col_A[]) {
  
  int64_t N_dim = R_dim + S_dim;
  int64_t NNZ = col_A[N_cols] - col_A[0];
  int64_t stride = N_dim * N_dim;
  int64_t lenB = (Lwork / stride) - N_cols;
  lenB = lenB > NNZ ? NNZ : lenB;
  int64_t N_rows_aligned = ((N_rows >> 4) + ((N_rows & 15) > 0)) * 16;
  int64_t NNZ_aligned = ((NNZ >> 4) + ((NNZ & 15) > 0)) * 16;

  std::vector<int64_t> rows(NNZ), cols(NNZ), orders(NNZ);
  int64_t lenL = partition_DLU(&rows[0], &cols[0], &orders[0], N_cols, col_offset, row_A, col_A);
  std::vector<int64_t> urows(NNZ), ucols(NNZ);
  int64_t K = count_apperance_x(&rows[0], &urows[0], NNZ);
  K = std::max(K, count_apperance_x(&cols[0], &ucols[0], NNZ));

  std::vector<double> one_data(K, 1.);
  double* one_data_dev;
  cudaMalloc(&one_data_dev, sizeof(double) * K);
  cudaMemcpy(one_data_dev, &one_data[0], sizeof(double) * K, cudaMemcpyHostToDevice);

  const int64_t NZ = 13, ND = 6;
  std::vector<double*> ptrs_nnz_cpu(NZ * NNZ_aligned);
  std::vector<double*> ptrs_diag_cpu(ND * N_rows_aligned);

  const double** _U_r = (const double**)&ptrs_nnz_cpu[0 * NNZ_aligned];
  const double** _U_s = (const double**)&ptrs_nnz_cpu[1 * NNZ_aligned];
  const double** _V_x = (const double**)&ptrs_nnz_cpu[2 * NNZ_aligned];
  const double** _A_sx = (const double**)&ptrs_nnz_cpu[3 * NNZ_aligned];
  double** _A_x = (double**)&ptrs_nnz_cpu[4 * NNZ_aligned];
  double** _B_x = (double**)&ptrs_nnz_cpu[5 * NNZ_aligned];
  double** _A_upper = (double**)&ptrs_nnz_cpu[6 * NNZ_aligned];
  double** _A_s = (double**)&ptrs_nnz_cpu[7 * NNZ_aligned];
  double** _Xo_Y = (double**)&ptrs_nnz_cpu[8 * NNZ_aligned];
  double** _Xc_Y = (double**)&ptrs_nnz_cpu[9 * NNZ_aligned];
  double** _Xc_X = (double**)&ptrs_nnz_cpu[10 * NNZ_aligned];
  double** _ACC_Y = (double**)&ptrs_nnz_cpu[11 * NNZ_aligned];
  double** _ACC_X = (double**)&ptrs_nnz_cpu[12 * NNZ_aligned];
  
  double** _X_d = (double**)&ptrs_diag_cpu[0 * N_rows_aligned];
  const double** _U_i = (const double**)&ptrs_diag_cpu[1 * N_rows_aligned];
  double** _ACC_I = (double**)&ptrs_diag_cpu[2 * N_rows_aligned];
  double** _B_X = (double**)&ptrs_diag_cpu[3 * N_rows_aligned];
  double** _Xo_I = (double**)&ptrs_diag_cpu[4 * N_rows_aligned];
  double** _ONE_LIST = (double**)&ptrs_diag_cpu[5 * N_rows_aligned];

  double* _V_data = Workspace;
  double* _B_data = &Workspace[N_cols * stride];

  std::vector<int64_t> ind(std::max(N_rows, NNZ) + 1);
  std::iota(ind.begin(), ind.end(), 0);

  std::transform(rows.begin(), rows.end(), _U_r, [U_ptr, stride](int64_t y) -> const double* { return &U_ptr[stride * y]; });
  std::transform(rows.begin(), rows.end(), _U_s, [U_ptr, stride, R_dim, N_dim](int64_t y) -> const double* { return &U_ptr[stride * y + R_dim * N_dim]; });
  std::transform(cols.begin(), cols.end(), _V_x, [_V_data, stride, col_offset](int64_t x) -> double* { return &_V_data[stride * (x - col_offset)]; });
  std::transform(orders.begin(), orders.end(), _A_x, [A_ptr, stride](int64_t yx) -> double* { return &A_ptr[stride * yx]; });
  std::transform(orders.begin(), orders.end(), _A_s, [A_ptr, stride, R_dim](int64_t yx) -> double* { return &A_ptr[stride * yx + R_dim]; });
  std::transform(orders.begin(), orders.end(), _A_upper, [A_up](int64_t yx) -> double* { return A_up[yx]; });

  std::transform(rows.begin(), rows.end(), _Xo_Y, [X_up](int64_t y) -> double* { return X_up[y]; });
  std::transform(rows.begin(), rows.end(), _Xc_Y, [X_ptr, R_dim](int64_t y) -> double* { return &X_ptr[y * R_dim]; });
  std::transform(cols.begin(), cols.end(), _Xc_X, [X_ptr, R_dim](int64_t x) -> double* { return &X_ptr[x * R_dim]; });
  std::transform(ind.begin(), ind.begin() + N_cols, _B_X, [_V_data, R_dim](int64_t x) -> double* { return &_V_data[x * R_dim]; });
  std::transform(ind.begin(), ind.begin() + N_rows, _Xo_I, [X_up](int64_t i) -> double* { return X_up[i]; });

  std::transform(rows.begin(), rows.end(), urows.begin(), _ACC_Y, 
    [_V_data, K, N_dim](int64_t y, int64_t uy) -> double* { return &_V_data[(y * K + uy) * N_dim]; });
  std::transform(cols.begin(), cols.end(), ucols.begin(), _ACC_X, 
    [_V_data, K, N_dim, col_offset](int64_t x, int64_t ux) -> double* { return &_V_data[((x - col_offset) * K + ux) * N_dim]; });
  std::transform(ind.begin(), ind.begin() + N_rows, _ACC_I, [_V_data, K, N_dim](int64_t i) -> double* { return &_V_data[i * N_dim * K]; });
  std::fill(_ONE_LIST, _ONE_LIST + N_rows, one_data_dev);

  std::transform(ind.begin(), ind.begin() + lenB, _B_x, [_B_data, stride](int64_t i) -> double* { return &_B_data[i * stride]; });
  std::transform(ind.begin(), ind.begin() + lenB, _A_sx, [_B_data, stride, R_dim, N_dim](int64_t i) -> double* { return &_B_data[i * stride + R_dim * N_dim]; });
  std::transform(ind.begin(), ind.begin() + N_cols, _X_d, [X_ptr, N_dim, col_offset](int64_t i) -> double* { return &X_ptr[N_dim * (i + col_offset)]; });
  std::transform(ind.begin(), ind.begin() + N_cols, _U_i, [U_ptr, stride, R_dim, N_rows](int64_t i) -> const double* { return &U_ptr[stride * N_rows + R_dim * i]; });
  
  memset((void*)params, 0, sizeof(struct BatchedFactorParams));

  params->N_r = R_dim;
  params->N_s = S_dim;
  params->N_upper = N_up;
  params->L_diag = N_cols;
  params->L_nnz = NNZ;
  params->L_lower = lenL;
  params->L_rows = N_rows;
  params->L_tmp = lenB;
  params->K = K;

  void** ptrs_nnz, **ptrs_diag;
  cudaMalloc((void**)&ptrs_nnz, sizeof(double*) * NNZ_aligned * NZ);
  cudaMalloc((void**)&ptrs_diag, sizeof(double*) * N_rows_aligned * ND);

  params->U_r = (const double**)&ptrs_nnz[0 * NNZ_aligned];
  params->U_s = (const double**)&ptrs_nnz[1 * NNZ_aligned];
  params->V_x = (double**)&ptrs_nnz[2 * NNZ_aligned];
  params->A_sx = (const double**)&ptrs_nnz[3 * NNZ_aligned];
  params->A_x = (double**)&ptrs_nnz[4 * NNZ_aligned];
  params->B_x = (double**)&ptrs_nnz[5 * NNZ_aligned];
  params->A_upper = (double**)&ptrs_nnz[6 * NNZ_aligned];
  params->A_s = (double**)&ptrs_nnz[7 * NNZ_aligned];
  params->Xo_Y = (double**)&ptrs_nnz[8 * NNZ_aligned];
  params->Xc_Y = (double**)&ptrs_nnz[9 * NNZ_aligned];
  params->Xc_X = (double**)&ptrs_nnz[10 * NNZ_aligned];
  params->ACC_Y = (double**)&ptrs_nnz[11 * NNZ_aligned];
  params->ACC_X = (double**)&ptrs_nnz[12 * NNZ_aligned];

  params->X_d = (double**)&ptrs_diag[0 * N_rows_aligned];
  params->U_i = (const double**)&ptrs_diag[1 * N_rows_aligned];
  params->ACC_I = (double**)&ptrs_diag[2 * N_rows_aligned];
  params->B_X = (double**)&ptrs_diag[3 * N_rows_aligned];
  params->Xo_I = (double**)&ptrs_diag[4 * N_rows_aligned];
  params->ONE_LIST = (double**)&ptrs_diag[5 * N_rows_aligned];

  params->U_d0 = U_ptr + stride * col_offset;
  params->Xc_d0 = X_ptr + R_dim * col_offset;
  params->V_data = _V_data;
  params->A_data = A_ptr;
  params->X_data = X_ptr;
  params->ONE_DATA = one_data_dev;

  cudaMemcpy(ptrs_nnz, ptrs_nnz_cpu.data(), sizeof(double*) * NNZ_aligned * NZ, cudaMemcpyHostToDevice);
  cudaMemcpy(ptrs_diag, ptrs_diag_cpu.data(), sizeof(double*) * N_rows_aligned * ND, cudaMemcpyHostToDevice);
}

void batchParamsDestory(struct BatchedFactorParams* params) {
  if (params->X_d)
    cudaFree(params->X_d);
  if (params->U_r)
    cudaFree(params->U_r);
  if (params->ONE_DATA)
    cudaFree(params->ONE_DATA);
}

void batchCholeskyFactor(struct BatchedFactorParams* params, const struct CellComm* comm) {
  int64_t U = params->N_upper, R = params->N_r, S = params->N_s, N = R + S, D = params->L_diag;
  int64_t alen = N * N * params->L_nnz;
  double one = 1., zero = 0., minus_one = -1.;

  level_merge_gpu(params->A_data, alen, comm);
  dup_bcast_gpu(params->A_data, alen, comm);

  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, 
    params->A_x, N, params->U_r, N, &zero, params->B_x, N, D);
  cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, N, R, N, &one, 
    params->U_r, N, params->B_x, N, &zero, params->A_x, N, D);
  cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, S, S, N, &one, 
    params->U_s, N, params->A_sx, N, &zero, params->A_upper, U, D);
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, 1, R, 1, &one, 
    params->ONE_LIST, 1, params->U_i, 1, &one, params->A_x, N + 1, D);

  cusolverDnDpotrfBatched(cusolverH, CUBLAS_FILL_MODE_LOWER, R, params->A_x, N, NULL, D);
  cublasDtrsmBatched(cublasH, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, 
    S, R, &one, params->A_x, N, params->A_s, N, D);
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, S, S, R, &minus_one, 
    params->A_s, N, params->A_s, N, &one, params->A_upper, U, D);
  cublasDtrsmBatched(cublasH, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 
    S, R, &one, params->A_x, N, params->A_s, N, D);

  if (D < params->L_nnz) {
    cublasDcopy(cublasH, N * N * D, params->U_d0, 1, params->V_data, 1);
    cublasDtrsmBatched(cublasH, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, 
      N, R, &one, params->A_x, N, params->V_x, N, D);
    cublasDtrsmBatched(cublasH, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 
      N, R, &one, params->A_x, N, params->V_x, N, D);
  }

  for (int64_t i = D; i < params->L_nnz; i += params->L_tmp) {
    int64_t len = std::min(params->L_nnz - i, params->L_tmp);
    cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, 
      &params->A_x[i], N, &params->V_x[i], N, &zero, params->B_x, N, len);
    cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, N, R, N, &one, 
      &params->U_r[i], N, params->B_x, N, &zero, &params->A_x[i], N, len);
    cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, S, S, N, &one,
      &params->U_s[i], N, params->A_sx, N, &zero, &params->A_upper[i], U, len);
  }
}

void batchForwardULV(struct BatchedFactorParams* params, const struct CellComm* comm) {
  int64_t R = params->N_r, S = params->N_s, N = R + S, D = params->L_diag, ONE = 1;
  int64_t K = params->K;
  double one = 1., zero = 0., minus_one = -1.;

  level_merge_gpu(params->X_data, params->L_rows * N, comm);
  neighbor_reduce_gpu(params->X_data, N, comm);
  dup_bcast_gpu(params->X_data, params->L_rows * N, comm);

  cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, R, ONE, N, &one,
    params->U_r, N, params->X_d, N, &zero, params->B_X, R, D);
  cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, S, ONE, N, &one,
    params->U_s, N, params->X_d, N, &zero, params->Xo_Y, S, D);
  cudaMemsetAsync(params->X_data, 0, sizeof(double) * params->L_rows * R, stream);
  cublasDcopy(cublasH, R * D, params->V_data, 1, params->Xc_d0, 1);

  cudaMemsetAsync(params->V_data, 0, sizeof(double) * params->L_rows * N * K, stream);
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, S, ONE, R, &one, 
    params->A_s, N, params->Xc_X, R, &zero, params->ACC_Y, N, params->L_nnz);
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, S, ONE, K, &minus_one,
    params->ACC_I, N, params->ONE_LIST, K, &one, params->Xo_I, S, params->L_rows);

  cudaMemsetAsync(params->V_data, 0, sizeof(double) * params->L_rows * N * K, stream);
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, R, ONE, R, &one, 
    &params->A_x[D], N, &params->Xc_X[D], R, &zero, &params->ACC_Y[D], N, params->L_lower);
  cublasDgemmStridedBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, R, ONE, K, &minus_one,
    params->V_data, N, N * K, params->ONE_DATA, K, 0, &one, params->X_data, R, R, params->L_rows);

  neighbor_reduce_gpu(params->X_data, R, comm);
  dup_bcast_gpu(params->X_data, params->L_rows * R, comm);
}

void batchBackwardULV(struct BatchedFactorParams* params, const struct CellComm* comm) {
  int64_t R = params->N_r, S = params->N_s, N = R + S, D = params->L_diag, ONE = 1;
  int64_t K = params->K;
  double one = 1., zero = 0., minus_one = -1.;

  cublasDtrsmBatched(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 
    R, ONE, &one, params->A_x, N, params->Xc_X, R, D);
  cublasDtrsmBatched(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, 
    R, ONE, &one, params->A_x, N, params->Xc_X, R, D);

  neighbor_bcast_gpu(params->X_data, R, comm);
  dup_bcast_gpu(params->X_data, params->L_rows * R, comm);

  cudaMemsetAsync(params->V_data, 0, sizeof(double) * D * N * K, stream);
  cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, R, ONE, S, &one, 
    params->A_s, N, params->Xo_Y, S, &zero, params->ACC_X, N, params->L_nnz);
  
  cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, R, ONE, R, &one, 
    &params->A_x[D], N, &params->Xc_Y[D], R, &one, &params->ACC_X[D], N, params->L_lower);
  cublasDgemmStridedBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, R, ONE, K, &minus_one,
    params->V_data, N, N * K, params->ONE_DATA, K, 0, &one, params->Xc_d0, R, R, D);
  
  cublasDcopy(cublasH, R * D, params->Xc_d0, 1, params->V_data, 1);
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, ONE, R, &one,
    params->U_r, N, params->B_X, R, &zero, params->X_d, N, D);
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, ONE, S, &one,
    params->U_s, N, params->Xo_Y, S, &one, params->X_d, N, D);
    
  neighbor_bcast_gpu(params->X_data, N, comm);
  dup_bcast_gpu(params->X_data, params->L_rows * N, comm);
}

void lastParamsCreate(struct BatchedFactorParams* params, double* A, double* X, int64_t N, int64_t S, int64_t clen, const int64_t cdims[]) {
  memset((void*)params, 0, sizeof(struct BatchedFactorParams));

  params->A_data = A;
  params->X_data = X;
  params->N_r = N;

  int Lwork;
  cusolverDnDpotrf_bufferSize(cusolverH, CUBLAS_FILL_MODE_LOWER, N, A, N, &Lwork);
  Lwork = std::max((int64_t)Lwork, N);
  cudaMalloc((void**)&params->ONE_DATA, sizeof(double) * Lwork);
  params->L_tmp = Lwork;

  std::vector<double> I(N, 1.);
  for (int64_t i = 0; i < clen; i++)
    std::fill(I.begin() + i * S, I.begin() + i * S + cdims[i], 0.);
  cudaMemcpy(params->ONE_DATA, &I[0], sizeof(double) * N, cudaMemcpyHostToDevice);
}

void chol_decomp(struct BatchedFactorParams* params, const struct CellComm* comm) {
  double* A = params->A_data;
  int64_t N = params->N_r;
  int64_t alen = N * N;
  double one = 1.;

  level_merge_gpu(params->A_data, alen, comm);
  dup_bcast_gpu(params->A_data, alen, comm);

  cublasDaxpy(cublasH, N, &one, params->ONE_DATA, 1, A, N + 1);
  cusolverDnDpotrf(cusolverH, CUBLAS_FILL_MODE_LOWER, N, A, N, params->ONE_DATA, params->L_tmp, NULL);
}


void chol_solve(struct BatchedFactorParams* params, const struct CellComm* comm) {
  const double* A = params->A_data;
  double* X = params->X_data;
  int64_t N = params->N_r;

  level_merge_gpu(X, N, comm);
  dup_bcast_gpu(X, N, comm);

  cusolverDnDpotrs(cusolverH, CUBLAS_FILL_MODE_LOWER, N, 1, A, N, X, N, NULL);
}


