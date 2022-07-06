

#include "linalg.h"

#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "string.h"
#include "inttypes.h"

#ifdef USE_MKL
#ifndef MKL_ILP64
#define MKL_ILP64
#endif
#include "mkl.h"
#else
#include "cblas.h"
#include "lapacke.h"
#endif

void matrixCreate(struct Matrix* mat, int64_t m, int64_t n) {
  int64_t size = m * n;
  if (size > 0) {
    mat->A = (double*)malloc(sizeof(double) * size);
    mat->M = m;
    mat->N = n;
  }
  else {
    mat->A = NULL;
    mat->M = 0;
    mat->N = 0;
  }
}

void matrixDestroy(struct Matrix* mat) {
  if (mat->A)
    free(mat->A);
  mat->A = NULL;
  mat->M = 0;
  mat->N = 0;
}


void cpyFromMatrix(const struct Matrix* A, double* v) {
  int64_t size = A->M * A->N;
  if (size > 0)
    memcpy(v, A->A, sizeof(double) * size);
}

void maxpby(struct Matrix* A, const double* v, double alpha, double beta) {
  int64_t size = A->M * A->N;
  if (beta == 0.)
    memset(A->A, 0, sizeof(double) * size);
  else if (beta != 1.)
    cblas_dscal(size, beta, A->A, 1);
  cblas_daxpy(size, alpha, v, 1, A->A, 1);
}

void cpyMatToMat(int64_t m, int64_t n, const struct Matrix* m1, struct Matrix* m2, int64_t y1, int64_t x1, int64_t y2, int64_t x2) {
  if (m > 0 && n > 0)
    for (int64_t j = 0; j < n; j++) {
      int64_t j1 = y1 + (x1 + j) * m1->M;
      int64_t j2 = y2 + (x2 + j) * m2->M;
      memcpy(&(m2->A)[j2], &(m1->A)[j1], sizeof(double) * m);
    }
}

void qr_with_complements(struct Matrix* Qo, struct Matrix* Qc, struct Matrix* R) {
  int64_t m = Qo->M;
  int64_t n = Qo->N;
  int64_t nc = m - n;

  if (m > 0 && n > 0 && nc >= 0) {
    double* work = (double*)malloc(sizeof(double) * (n + m * m));
    struct Matrix Q = { &work[n], m, m };
    cpyMatToMat(m, n, Qo, &Q, 0, 0, 0, 0);

    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, Q.A, m, work);
    cpyMatToMat(n, n, &Q, R, 0, 0, 0, 0);
    LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, m, n, Q.A, m, work);
    cpyMatToMat(m, n, &Q, Qo, 0, 0, 0, 0);
    if (nc > 0)
      cpyMatToMat(m, nc, &Q, Qc, 0, n, 0, 0);
    
    for (int64_t i = 0; i < n - 1; i++)
      memset(&(R->A)[i * n + i + 1], 0, sizeof(double) * (n - i - 1));

    free(work);
  }
}

void updateSubU(struct Matrix* U, const struct Matrix* R1, const struct Matrix* R2) {
  if (U->M > 0 && U->N > 0) {
    int64_t m1 = R1->N;
    int64_t m2 = R2->N;
    int64_t n = U->N;

    double* work = (double*)malloc(sizeof(double) * n * (R1->M + R2->M));
    struct Matrix ru1 = { work, R1->M, n };
    struct Matrix ru2 = { &work[n * R1->M], R2->M, n };

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, R1->M, n, m1, 1., R1->A, R1->M, U->A, U->M, 0., ru1.A, R1->M);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, R2->M, n, m2, 1., R2->A, R2->M, &(U->A)[m1], U->M, 0., ru2.A, R2->M);
    cpyMatToMat(R1->M, n, &ru1, U, 0, 0, 0, 0);
    cpyMatToMat(R2->M, n, &ru2, U, 0, 0, R1->M, 0);

    free(work);
  }
}

void lraID(double epi, struct Matrix* A, struct Matrix* U, int64_t arows[], int64_t* rnk_out) {
  int64_t rank_a = A->M < A->N ? A->M : A->N;
  int64_t rank = rank_a;
  if (*rnk_out > 0)
    rank = *rnk_out < rank ? *rnk_out : rank;

  double* work = (double*)malloc(sizeof(double) * (rank_a + rank_a + 1));
  LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'N', A->M, A->N, A->A, A->M, work, U->A, A->M, NULL, A->N, &work[rank_a]);
  if (epi > 0.) {
    int64_t r = 0;
    double sepi = work[0] * epi;
    while(r < rank && work[r] > sepi)
      r += 1;
    rank = r;
  }

  memcpy(A->A, U->A, sizeof(double) * A->M * rank);
  for (int64_t i = 0; i < rank; i++)
    cblas_dscal(A->M, work[i], &(A->A)[i * A->M], 1);

#ifdef USE_MKL
  MKL_INT* ipiv = (MKL_INT*)arows;
  MKL_INT* np = (MKL_INT*)malloc(sizeof(MKL_INT) * rank);
#else
  int* ipiv = (int*)malloc(sizeof(int) * rank * 2);
  int* np = &ipiv[rank];
#endif
  int info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, A->M, rank, A->A, A->M, ipiv);
  if (info > 0)
    rank = info - 1;
  *rnk_out = rank;

  for (int64_t i = 0; i < rank; i++)
    np[i] = i + 1;
  LAPACKE_dgetri(LAPACK_COL_MAJOR, rank, A->A, A->M, np);
  for (int64_t i = 0; i < rank; i++)
    cblas_dscal(rank, work[i], &(A->A)[i], A->M);

  double* work2 = (double*)malloc(sizeof(double) * (A->M * rank));
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A->M, rank, rank, 1., U->A, A->M, A->A, A->M, 0., work2, A->M);
  memcpy(U->A, work2, sizeof(double) * A->M * rank);
  free(work2);

#ifdef USE_MKL
  free(np);
#else
  for (int64_t i = 0; i < rank; i++)
    arows[i] = ipiv[i];
  free(ipiv);
#endif
  free(work);
}

void zeroMatrix(struct Matrix* A) {
  memset(A->A, 0, sizeof(double) * A->M * A->N);
}

void mmult(char ta, char tb, const struct Matrix* A, const struct Matrix* B, struct Matrix* C, double alpha, double beta) {
  int64_t k = (ta == 'N' || ta == 'n') ? A->N : A->M;
  if (C->M > 0 && C->N > 0 && k > 0) {
    CBLAS_TRANSPOSE tac = (ta == 'T' || ta == 't') ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE tbc = (tb == 'T' || tb == 't') ? CblasTrans : CblasNoTrans;
    cblas_dgemm(CblasColMajor, tac, tbc, C->M, C->N, k, alpha, A->A, A->M, B->A, B->M, beta, C->A, C->M);
  }
}

void chol_decomp(struct Matrix* A) {
  if (A->M > 0)
    LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', A->M, A->A, A->M);
}

void trsm_lowerA(struct Matrix* A, const struct Matrix* L) {
  if (A->M > 0 && L->M > 0 && L->N > 0)
    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, A->M, A->N, 1., L->A, L->M, A->A, A->M);
}

void utav(char tb, const struct Matrix* U, const struct Matrix* A, const struct Matrix* VT, struct Matrix* C) {
  if (tb == 'N' || tb == 'n') {
    double* work = (double*)malloc(sizeof(double) * C->M * A->N);
    struct Matrix UA = { work, C->M, A->N };
    mmult('T', 'N', U, A, &UA, 1., 0.);
    mmult('N', 'N', &UA, VT, C, 1., 0.);
    free(work);
  }
  else if (tb == 'T' || tb == 't') {
    double* work = (double*)malloc(sizeof(double) * C->M * A->N);
    struct Matrix UA = { work, C->M, A->N };
    mmult('N', 'N', U, A, &UA, 1., 0.);
    mmult('N', 'T', &UA, VT, C, 1., 0.);
    free(work);
  }
}

void mat_solve(char type, struct Matrix* X, const struct Matrix* A) {
  if (A->M > 0 && X->N > 0) {
    if (type == 'F' || type == 'f' || type == 'A' || type == 'a')
      cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, X->M, X->N, 1., A->A, A->M, X->A, X->M);
    if (type == 'B' || type == 'b' || type == 'A' || type == 'a')
      cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit, X->M, X->N, 1., A->A, A->M, X->A, X->M);
  }
}

void normalizeA(struct Matrix* A, const struct Matrix* B) {
  int64_t len_A = A->M * A->N;
  int64_t len_B = B->M * B->N;
  if (len_A > 0 && len_B > 0) {
    double nrm_A = cblas_dnrm2(len_A, A->A, 1);
    double nrm_B = cblas_dnrm2(len_B, B->A, 1);
    cblas_dscal(len_A, nrm_B / nrm_A, A->A, 1);
  }
}

void mnrm2(const struct Matrix* A, double* nrm) {
  *nrm = cblas_dnrm2(A->M * A->N, A->A, 1);
}

void matrix_mem(int64_t* bytes, const struct Matrix* A, int64_t lenA) {
  int64_t count = sizeof(struct Matrix) * lenA;
  for (int64_t i = 0; i < lenA; i++)
    count = count + sizeof(double) * A[i].M * A[i].N;
  *bytes = count;
}
