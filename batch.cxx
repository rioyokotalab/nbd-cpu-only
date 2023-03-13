
#include "nbd.hxx"
#include "profile.hxx"

#include "mkl.h"
#include <string.h>
#include <cstdio>
#include <algorithm>

void set_work_size(int64_t Lwork, double** D_DATA, int64_t* D_DATA_SIZE) {
  if (Lwork > *D_DATA_SIZE) {
    *D_DATA_SIZE = Lwork;
    if (*D_DATA)
      free(*D_DATA);
    *D_DATA = (double*)malloc(sizeof(double) * Lwork);
  }
  else if (Lwork <= 0) {
    *D_DATA_SIZE = 0;
    if (*D_DATA)
      free(*D_DATA);
  }
}

void allocBufferedList(void** A_ptr, void** A_buffer, int64_t element_size, int64_t count) {
  *A_ptr = calloc(count, element_size);
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
    free(A_buffer);
  free(A_ptr);
}

struct BatchedFactorParams { 
  int64_t N_r, N_s, N_upper, L_diag, L_nnz, L_lower, L_rows, L_tmp;
  const double** A_d, **U_d, **U_ds, **U_r, **U_s, **V_x, **A_rs, **A_sx, **A_xlo, **A_sr;
  double** U_dx, **A_x, **B_x, **A_ss, **A_upper, *UD_data, *A_data, *B_data;
  double** X_d, **Xc_d, **Xc_y, **Xc_x, **Xo_d, **Xo_y, **B_d, **B_xlo, *X_data, *Xc_data;
};

void batchParamsCreate(void** params, int64_t R_dim, int64_t S_dim, const double* U_ptr, double* A_ptr, double* X_ptr, int64_t N_up, double** A_up, double** X_up,
  double* Workspace, int64_t Lwork, int64_t N_rows, int64_t N_cols, int64_t col_offset, const int64_t row_A[], const int64_t col_A[]) {
  
  int64_t N_dim = R_dim + S_dim;
  int64_t NNZ = col_A[N_cols] - col_A[0];
  int64_t stride = N_dim * N_dim;
  int64_t lenB = (Lwork / stride) - N_cols;
  lenB = lenB > NNZ ? NNZ : lenB;

  const double** _A_d = (const double**)malloc(sizeof(double*) * N_cols);
  const double** _U_d = (const double**)malloc(sizeof(double*) * N_cols);
  const double** _U_ds = (const double**)malloc(sizeof(double*) * N_cols);
  const double** _U_r = (const double**)malloc(sizeof(double*) * NNZ);
  const double** _U_s = (const double**)malloc(sizeof(double*) * NNZ);
  const double** _V_x = (const double**)malloc(sizeof(double*) * NNZ);
  const double** _A_rs = (const double**)malloc(sizeof(double*) * N_cols);
  const double** _A_sx = (const double**)malloc(sizeof(double*) * lenB);
  const double** _A_xlo = (const double**)malloc(sizeof(double*) * NNZ);
  const double** _A_sr = (const double**)malloc(sizeof(double*) * NNZ);

  double** _U_dx = (double**)malloc(sizeof(double*) * N_cols);
  double** _A_x = (double**)malloc(sizeof(double*) * NNZ);
  double** _B_x = (double**)malloc(sizeof(double*) * lenB);
  double** _A_ss = (double**)malloc(sizeof(double*) * N_cols);
  double** _A_upper = (double**)malloc(sizeof(double*) * NNZ);

  double** _X_d = (double**)malloc(sizeof(double*) * N_cols);
  double** _Xc_d = (double**)malloc(sizeof(double*) * N_cols);
  double** _Xc_y = (double**)malloc(sizeof(double*) * NNZ);
  double** _Xc_x = (double**)malloc(sizeof(double*) * NNZ);
  double** _Xo_d = (double**)malloc(sizeof(double*) * N_cols);
  double** _Xo_y = (double**)malloc(sizeof(double*) * NNZ);
  double** _B_d = (double**)malloc(sizeof(double*) * N_cols);
  double** _B_xlo = (double**)malloc(sizeof(double*) * NNZ);

  double* _UD_data = Workspace;
  double* _B_data = &Workspace[N_cols * stride];
  double* _Xc_data = (double*)calloc(N_rows * R_dim, sizeof(double));

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
  
  struct BatchedFactorParams* params_ptr = (struct BatchedFactorParams*)malloc(sizeof(struct BatchedFactorParams));
  params_ptr->N_r = R_dim;
  params_ptr->N_s = S_dim;
  params_ptr->N_upper = N_up;
  params_ptr->L_diag = N_cols;
  params_ptr->L_nnz = NNZ;
  params_ptr->L_lower = countL;
  params_ptr->L_rows = N_rows;
  params_ptr->L_tmp = lenB;

  params_ptr->A_d = _A_d;
  params_ptr->U_d = _U_d;
  params_ptr->U_ds = _U_ds;
  params_ptr->U_r = _U_r;
  params_ptr->U_s = _U_s;
  params_ptr->V_x = _V_x;
  params_ptr->A_rs = _A_rs;
  params_ptr->A_sx = _A_sx;
  params_ptr->A_xlo = _A_xlo;
  params_ptr->A_sr = _A_sr;

  params_ptr->U_dx = _U_dx;
  params_ptr->A_x = _A_x;
  params_ptr->B_x = _B_x;
  params_ptr->A_ss = _A_ss;
  params_ptr->A_upper = _A_upper;
  params_ptr->UD_data = _UD_data;
  params_ptr->A_data = A_ptr;
  params_ptr->B_data = _B_data;

  params_ptr->X_d = _X_d;
  params_ptr->Xc_d = _Xc_d;
  params_ptr->Xc_y = _Xc_y;
  params_ptr->Xc_x = _Xc_x;
  params_ptr->Xo_d = _Xo_d;
  params_ptr->Xo_y = _Xo_y;
  params_ptr->B_d = _B_d;
  params_ptr->B_xlo = _B_xlo;
  params_ptr->X_data = X_ptr;
  params_ptr->Xc_data = _Xc_data;

  *params = params_ptr;
}

void batchParamsDestory(void* params) {
  struct BatchedFactorParams* params_ptr = (struct BatchedFactorParams*)params;
  if (params_ptr->A_d)
    free(params_ptr->A_d);
  if (params_ptr->U_d)
    free(params_ptr->U_d);
  if (params_ptr->U_ds)
    free(params_ptr->U_ds);
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
  if (params_ptr->A_xlo)
    free(params_ptr->A_xlo);
  if (params_ptr->A_sr)
    free(params_ptr->A_sr);
  if (params_ptr->A_x)
    free(params_ptr->A_x);
  if (params_ptr->B_x)
    free(params_ptr->B_x);
  if (params_ptr->A_ss)
    free(params_ptr->A_ss);
  if (params_ptr->A_upper)
    free(params_ptr->A_upper);
  if (params_ptr->X_d)
    free(params_ptr->X_d);
  if (params_ptr->Xc_d)
    free(params_ptr->Xc_d);
  if (params_ptr->Xc_y)
    free(params_ptr->Xc_y);
  if (params_ptr->Xc_x)
    free(params_ptr->Xc_x);
  if (params_ptr->Xo_d)
    free(params_ptr->Xo_d);
  if (params_ptr->Xo_y)
    free(params_ptr->Xo_y);
  if (params_ptr->B_d)
    free(params_ptr->B_d);
  if (params_ptr->B_xlo)
    free(params_ptr->B_xlo);

  if (params_ptr->Xc_data)
    free(params_ptr->Xc_data);

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

void batchForwardULV(void* params_ptr, const struct CellComm* comm) {
  struct BatchedFactorParams* params = (struct BatchedFactorParams*)params_ptr;
  CBLAS_SIDE left = CblasLeft;
  CBLAS_UPLO lower = CblasLower;
  CBLAS_TRANSPOSE trans = CblasTrans;
  CBLAS_TRANSPOSE no_trans = CblasNoTrans;
  CBLAS_DIAG non_unit = CblasNonUnit;
  double one = 1.;
  double zero = 0.;
  double minus_one = -1.;
  MKL_INT R = params->N_r;
  MKL_INT S = params->N_s;
  MKL_INT N = R + S;
  MKL_INT D = params->L_diag;
  MKL_INT ONE = 1;

#ifdef _PROF
  double stime = MPI_Wtime();
#endif
  level_merge_cpu(params->X_data, params->L_rows * N, comm);
  neighbor_reduce_cpu(params->X_data, N, comm);
  dup_bcast_cpu(params->X_data, params->L_rows * N, comm);
#ifdef _PROF
  double etime = MPI_Wtime() - stime;
  recordCommTime(etime);
#endif

  cblas_dgemm_batch(CblasColMajor, &trans, &no_trans, &R, &ONE, &N, &one,
    params->U_d, &N, (const double**)(params->X_d), &N, &zero, params->Xc_d, &R, 1, &D);
  cblas_dgemm_batch(CblasColMajor, &trans, &no_trans, &S, &ONE, &N, &one,
    params->U_ds, &N, (const double**)(params->X_d), &N, &zero, params->Xo_d, &S, 1, &D);
  cblas_dcopy(R * D, params->Xc_d[0], 1, params->B_d[0], 1);
  cblas_dtrsm_batch(CblasColMajor, &left, &lower, &no_trans, &non_unit, &R, &ONE, &one, 
    params->A_d, &N, params->B_d, &N, 1, &D);

  for (int64_t i = 0; i < params->L_lower; i++)
    cblas_dgemm(CblasColMajor, no_trans, no_trans, R, ONE, R, minus_one, params->A_xlo[i], N, params->B_xlo[i], R, one, params->Xc_y[i], R);

#ifdef _PROF
  stime = MPI_Wtime();
#endif
  neighbor_reduce_cpu(params->Xc_data, R, comm);
  dup_bcast_cpu(params->Xc_data, params->L_rows * R, comm);
#ifdef _PROF
  etime = MPI_Wtime() - stime;
  recordCommTime(etime);
#endif

  cblas_dtrsm_batch(CblasColMajor, &left, &lower, &no_trans, &non_unit, &R, &ONE, &one, 
    params->A_d, &N, params->Xc_d, &N, 1, &D);
  for (int64_t i = 0; i < params->L_nnz; i++)
    cblas_dgemm(CblasColMajor, no_trans, no_trans, S, ONE, R, minus_one, params->A_sr[i], N, params->Xc_x[i], R, one, params->Xo_y[i], R);
}

void batchBackwardULV(void* params_ptr, const struct CellComm* comm) {
  struct BatchedFactorParams* params = (struct BatchedFactorParams*)params_ptr;
  CBLAS_SIDE left = CblasLeft;
  CBLAS_UPLO lower = CblasLower;
  CBLAS_TRANSPOSE trans = CblasTrans;
  CBLAS_TRANSPOSE no_trans = CblasNoTrans;
  CBLAS_DIAG non_unit = CblasNonUnit;
  double one = 1.;
  double zero = 0.;
  double minus_one = -1.;
  MKL_INT R = params->N_r;
  MKL_INT S = params->N_s;
  MKL_INT N = R + S;
  MKL_INT D = params->L_diag;
  MKL_INT ONE = 1;

  for (int64_t i = 0; i < params->L_nnz; i++)
    cblas_dgemm(CblasColMajor, trans, no_trans, R, ONE, S, minus_one, params->A_sr[i], N, params->Xo_y[i], R, one, params->Xc_x[i], R);
  cblas_dcopy(R * D, params->Xc_d[0], 1, params->B_d[0], 1);
  cblas_dtrsm_batch(CblasColMajor, &left, &lower, &trans, &non_unit, &R, &ONE, &one, 
    params->A_d, &N, params->Xc_d, &R, 1, &D);

#ifdef _PROF
  double stime = MPI_Wtime();
#endif
  neighbor_bcast_cpu(params->Xc_data, R, comm);
  dup_bcast_cpu(params->Xc_data, params->L_rows * R, comm);
#ifdef _PROF
  double etime = MPI_Wtime() - stime;
  recordCommTime(etime);
#endif
  
  for (int64_t i = 0; i < params->L_lower; i++)
    cblas_dgemm(CblasColMajor, trans, no_trans, R, ONE, R, minus_one, params->A_xlo[i], N, params->Xc_y[i], R, one, params->B_xlo[i], R);
  cblas_dtrsm_batch(CblasColMajor, &left, &lower, &trans, &non_unit, &R, &ONE, &one, 
    params->A_d, &N, params->B_d, &N, 1, &D);
  cblas_dgemm_batch(CblasColMajor, &no_trans, &no_trans, &N, &ONE, &R, &one,
    params->U_d, &N, (const double**)(params->B_d), &R, &zero, params->X_d, &N, 1, &D);
  cblas_dgemm_batch(CblasColMajor, &no_trans, &no_trans, &N, &ONE, &S, &one,
    params->U_ds, &N, (const double**)(params->Xo_d), &S, &one, params->X_d, &N, 1, &D);
    
#ifdef _PROF
  stime = MPI_Wtime();
#endif
  neighbor_bcast_cpu(params->X_data, N, comm);
  dup_bcast_cpu(params->X_data, params->L_rows * N, comm);
#ifdef _PROF
  etime = MPI_Wtime() - stime;
  recordCommTime(etime);
#endif
}

struct LastFactorParams {
  double* A_ptr, *X_ptr;
  int64_t N_A;
};

void lastParamsCreate(void** params, double* A, double* X, int64_t N) {
  struct LastFactorParams* params_ptr = (struct LastFactorParams*)malloc(sizeof(struct LastFactorParams));
  *params = params_ptr;

  params_ptr->A_ptr = A;
  params_ptr->X_ptr = X;
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

void chol_solve(void* params_ptr, const struct CellComm* comm) {
  struct LastFactorParams* params = (struct LastFactorParams*)params_ptr;
  double* A = params->A_ptr;
  double* X = params->X_ptr;
  int64_t N = params->N_A;

#ifdef _PROF
  double stime = MPI_Wtime();
#endif
  level_merge_cpu(params->X_ptr, N, comm);
  dup_bcast_cpu(params->X_ptr, N, comm);
#ifdef _PROF
  double etime = MPI_Wtime() - stime;
  recordCommTime(etime);
#endif
  LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'L', N, 1, A, N, X, N);
}
