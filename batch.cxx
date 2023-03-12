
#include "nbd.hxx"
#include "profile.hxx"

#include "mkl.h"
#include "omp.h"
#include <string.h>
#include <cstdio>

#define ALIGN 16

void set_work_size(int64_t Lwork, double** D_DATA, int64_t* D_DATA_SIZE) {
  if (Lwork > *D_DATA_SIZE) {
    *D_DATA_SIZE = Lwork;
    if (*D_DATA)
      MKL_free(*D_DATA);
    *D_DATA = (double*)MKL_malloc(sizeof(double) * Lwork, ALIGN);
  }
  else if (Lwork <= 0) {
    *D_DATA_SIZE = 0;
    if (*D_DATA)
      MKL_free(*D_DATA);
  }
}

void allocBufferedList(void** A_ptr, void** A_buffer, int64_t element_size, int64_t count) {
  *A_ptr = MKL_calloc(count, element_size, ALIGN);
  *A_buffer = *A_ptr;
}

void flushBuffer(char dir, void* A_ptr, void* A_buffer, int64_t element_size, int64_t count) {
  if (A_ptr != A_buffer) {
    if (dir == 'S')
      memcpy(A_ptr, A_buffer, element_size * count);
    else if (dir == 'G')
      memcpy(A_buffer, A_ptr, element_size * count);
  }
}

void freeBufferedList(void* A_ptr, void* A_buffer) {
  if (A_buffer != A_ptr)
    MKL_free(A_buffer);
  MKL_free(A_ptr);
}

struct BatchedFactorParams { 
  int64_t N_r, N_s, N_upper, L_diag, L_nnz, L_tmp;
  const double** A_d, **U_d, **U_r, **U_s, **V_x, **A_rs, **A_sx;
  double** U_dx, **A_x, **B_x, **A_ss, **A_upper, *UD_data, *A_data, *B_data;
};

void batchParamsCreate(void** params, int64_t R_dim, int64_t S_dim, const double* U_ptr, double* A_ptr, int64_t N_up, double** A_up, double* Workspace, int64_t Lwork,
  int64_t N_cols, int64_t col_offset, const int64_t row_A[], const int64_t col_A[]) {
  
  int64_t N_dim = R_dim + S_dim;
  int64_t NNZ = col_A[N_cols] - col_A[0];
  int64_t stride = N_dim * N_dim;
  int64_t lenB = (Lwork / stride) - N_cols;
  lenB = lenB > NNZ ? NNZ : lenB;

  const double** _A_d = (const double**)malloc(sizeof(double*) * N_cols);
  const double** _U_d = (const double**)malloc(sizeof(double*) * N_cols);
  const double** _U_r = (const double**)malloc(sizeof(double*) * NNZ);
  const double** _U_s = (const double**)malloc(sizeof(double*) * NNZ);
  const double** _V_x = (const double**)malloc(sizeof(double*) * NNZ);
  const double** _A_rs = (const double**)malloc(sizeof(double*) * N_cols);
  const double** _A_sx = (const double**)malloc(sizeof(double*) * lenB);

  double** _U_dx = (double**)malloc(sizeof(double*) * N_cols);
  double** _A_x = (double**)malloc(sizeof(double*) * NNZ);
  double** _B_x = (double**)malloc(sizeof(double*) * lenB);
  double** _A_ss = (double**)malloc(sizeof(double*) * N_cols);
  double** _A_upper = (double**)malloc(sizeof(double*) * NNZ);

  double* _UD_data = Workspace;
  double* _B_data = &Workspace[N_cols * stride];

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
    _U_d[x] = U_ptr + stride * (x + col_offset);
    _A_rs[x] = A_ptr + stride * diag_id + R_dim;
    _U_dx[x] = _UD_data + stride * x;
    _A_ss[x] = A_up[diag_id];
  }

  for (int64_t x = 0; x < lenB; x++) {
    _B_x[x] = _B_data + stride * x;
    _A_sx[x] = _B_data + stride * x + R_dim * N_dim;
  }
  
  struct BatchedFactorParams* params_ptr = (struct BatchedFactorParams*)malloc(sizeof(struct BatchedFactorParams));
  params_ptr->N_r = R_dim;
  params_ptr->N_s = S_dim;
  params_ptr->N_upper = N_up;
  params_ptr->L_diag = N_cols;
  params_ptr->L_nnz = NNZ;
  params_ptr->L_tmp = lenB;

  params_ptr->A_d = _A_d;
  params_ptr->U_d = _U_d;
  params_ptr->U_r = _U_r;
  params_ptr->U_s = _U_s;
  params_ptr->V_x = _V_x;
  params_ptr->A_rs = _A_rs;
  params_ptr->A_sx = _A_sx;

  params_ptr->U_dx = _U_dx;
  params_ptr->A_x = _A_x;
  params_ptr->B_x = _B_x;
  params_ptr->A_ss = _A_ss;
  params_ptr->A_upper = _A_upper;
  params_ptr->UD_data = _UD_data;
  params_ptr->A_data = A_ptr;
  params_ptr->B_data = _B_data;

  *params = params_ptr;
}

void batchParamsDestory(void* params) {
  struct BatchedFactorParams* params_ptr = (struct BatchedFactorParams*)params;
  if (params_ptr->A_d)
    free(params_ptr->A_d);
  if (params_ptr->U_d)
    free(params_ptr->U_d);
  if (params_ptr->U_r)
    free(params_ptr->U_r);
  if (params_ptr->U_s)
    free(params_ptr->U_s);
  if (params_ptr->V_x)
    free(params_ptr->V_x);
  if (params_ptr->A_rs)
    free(params_ptr->A_rs);
  if (params_ptr->A_sx)
    free(params_ptr->A_sx);
  if (params_ptr->U_dx)
    free(params_ptr->U_dx);
  if (params_ptr->A_x)
    free(params_ptr->A_x);
  if (params_ptr->B_x)
    free(params_ptr->B_x);
  if (params_ptr->A_ss)
    free(params_ptr->A_ss);
  if (params_ptr->A_upper)
    free(params_ptr->A_upper);

  free(params);
}

void batchCholeskyFactor(void* params_ptr, const struct CellComm* comm) {
  struct BatchedFactorParams* params = (struct BatchedFactorParams*)params_ptr;
  CBLAS_SIDE right = CblasRight;
  CBLAS_UPLO lower = CblasLower;
  CBLAS_TRANSPOSE trans = CblasTrans;
  CBLAS_TRANSPOSE no_trans = CblasNoTrans;
  CBLAS_DIAG non_unit = CblasNonUnit;
  double one = 1.;
  double zero = 0.;
  double minus_one = -1.;
  MKL_INT U = params->N_upper;
  MKL_INT R = params->N_r;
  MKL_INT S = params->N_s;
  MKL_INT N = R + S;
  MKL_INT D = params->L_diag;
  int64_t alen = N * N * params->L_nnz;

#ifdef _PROF
  double stime = MPI_Wtime();
#endif
  level_merge_cpu(params->A_data, alen, comm);
  dup_bcast_cpu(params->A_data, alen, comm);
#ifdef _PROF
  double etime = MPI_Wtime() - stime;
  recordCommTime(etime);
#endif

  cblas_dgemm_batch(CblasColMajor, &no_trans, &no_trans, &N, &R, &N, &one, 
    params->A_d, &N, params->U_d, &N, &zero, params->U_dx, &N, 1, &D);
  cblas_dgemm_batch(CblasColMajor, &trans, &no_trans, &R, &R, &N, &one, 
    params->U_d, &N, (const double**)(params->U_dx), &N, &zero, params->B_x, &N, 1, &D);
  cblas_dcopy(N * N * D, params->U_d[0], 1, params->UD_data, 1);

  for (int64_t i = 0; i < D; i++)
    LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', R, params->B_x[i], N);
  cblas_dtrsm_batch(CblasColMajor, &right, &lower, &trans, &non_unit, &N, &R, &one, 
    (const double**)(params->B_x), &N, params->U_dx, &N, 1, &D);

  for (int64_t i = 0; i < params->L_nnz; i += params->L_tmp) {
    MKL_INT len = params->L_nnz - i > params->L_tmp ? params->L_tmp : params->L_nnz - i;
    cblas_dgemm_batch(CblasColMajor, &no_trans, &no_trans, &N, &N, &N, &one,
      (const double**)(&params->A_x[i]), &N, &params->V_x[i], &N, &zero, params->B_x, &N, 1, &len);
    cblas_dgemm_batch(CblasColMajor, &trans, &no_trans, &N, &R, &N, &one, 
      &params->U_r[i], &N, (const double**)(params->B_x), &N, &zero, &params->A_x[i], &N, 1, &len);
    cblas_dgemm_batch(CblasColMajor, &trans, &no_trans, &S, &S, &N, &one, 
      &params->U_s[i], &N, params->A_sx, &N, &zero, &params->A_upper[i], &U, 1, &len);
  }
  cblas_dgemm_batch(CblasColMajor, &no_trans, &trans, &S, &S, &R, &minus_one,
    params->A_rs, &N, params->A_rs, &N, &one, params->A_ss, &U, 1, &D);
}

struct LastFactorParams {
  double* A_ptr;
  int64_t N_A;
};

void lastParamsCreate(void** params, double* A, int64_t N) {
  struct LastFactorParams* params_ptr = (struct LastFactorParams*)malloc(sizeof(struct LastFactorParams));
  *params = params_ptr;

  params_ptr->A_ptr = A;
  params_ptr->N_A = N;
}

void lastParamsDestory(void* params) {
  free(params);
}

void chol_decomp(void* params_ptr, const struct CellComm* comm) {
  struct LastFactorParams* params = (struct LastFactorParams*)params_ptr;
  double* A = params->A_ptr;
  int64_t N = params->N_A;
  int64_t alen = N * N;

#ifdef _PROF
  double stime = MPI_Wtime();
#endif
  level_merge_cpu(params->A_ptr, alen, comm);
  dup_bcast_cpu(params->A_ptr, alen, comm);
#ifdef _PROF
  double etime = MPI_Wtime() - stime;
  recordCommTime(etime);
#endif
  LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', N, A, N);
}
