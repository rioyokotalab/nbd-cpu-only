

#include "linalg.hxx"

#if USE_MKL
#include "mkl.h"
#else
#include "cblas.h"
#include "lapacke.h"
#endif


#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <cstdlib>
#include "inttypes.h"

using namespace nbd;

double* Rvec = NULL;
int64_t Rlen = 0;

void nbd::cRandom(int64_t lenR, double min, double max, unsigned int seed) {
  if (lenR > 0) {
    if (seed > 0)
      std::srand(seed);
    if (Rlen > 0)
      free(Rvec);
    Rvec = (double*)malloc(sizeof(double) * lenR);
    Rlen = lenR;

    double range = max - min;
    for (int64_t i = 0; i < lenR; i++)
      Rvec[i] = ((double)std::rand() / RAND_MAX) * range + min;
  }
  else if (Rlen > 0) {
    free(Rvec);
    Rvec = NULL;
    Rlen = 0;
  }
}

void nbd::cMatrix(Matrix& mat, int64_t m, int64_t n) {
  int64_t size = m * n;
  int64_t size_old = mat.M * mat.N;
  if (size > 0 && size != size_old) {
    mat.A.resize(size);
    mat.M = m;
    mat.N = n;
  }
  else if (size <= 0) {
    mat.A.clear();
    mat.M = 0;
    mat.N = 0;
  }
}

void nbd::cVector(Vector& vec, int64_t n) {
  int64_t n_old = vec.N;
  if (n > 0 && n != n_old) {
    vec.X.resize(n);
    vec.N = n;
  }
  else if (n <= 0) {
    vec.X.clear();
    vec.N = 0;
  }
}

void nbd::cpyFromMatrix(const Matrix& A, double* v) {
  int64_t size = A.M * A.N;
  if (size > 0)
    std::copy(A.A.data(), A.A.data() + size, v);
}

void nbd::cpyFromVector(const Vector& A, double* v) {
  std::copy(A.X.data(), A.X.data() + A.N, v);
}

void nbd::maxpby(Matrix& A, const double* v, double alpha, double beta) {
  int64_t size = A.M * A.N;
  if (beta == 0.)
    std::fill(A.A.data(), A.A.data() + size, 0.);
  else if (beta != 1.)
    cblas_dscal(size, beta, A.A.data(), 1);
  cblas_daxpy(size, alpha, v, 1, A.A.data(), 1);
}

void nbd::vaxpby(Vector& A, const double* v, double alpha, double beta) {
  int64_t size = A.N;
  if (beta == 0.)
    std::fill(A.X.data(), A.X.data() + size, 0.);
  else if (beta != 1.)
    cblas_dscal(size, beta, A.X.data(), 1);
  cblas_daxpy(size, alpha, v, 1, A.X.data(), 1);
}

void nbd::cpyMatToMat(int64_t m, int64_t n, const Matrix& m1, Matrix& m2, int64_t y1, int64_t x1, int64_t y2, int64_t x2) {
  if (m > 0 && n > 0)
    for (int64_t j = 0; j < n; j++) {
      int64_t j1 = y1 + (x1 + j) * m1.M;
      int64_t j2 = y2 + (x2 + j) * m2.M;
      std::copy(&m1.A[j1], &m1.A[j1] + m, &m2.A[j2]);
    }
}

void nbd::cpyVecToVec(int64_t n, const Vector& v1, Vector& v2, int64_t x1, int64_t x2) {
  if (n > 0)
    std::copy(&v1.X[x1], &v1.X[x1] + n, &v2.X[x2]);
}

void nbd::updateU(Matrix& Qo, Matrix& Qc, Matrix& R) {
  int64_t m = R.M;
  int64_t n = R.N;
  int64_t nc = m - n;

  if (m > 0 && n > 0 && nc >= 0) {
    Matrix work;
    cMatrix(work, m, m);
    cpyMatToMat(m, n, R, work, 0, 0, 0, 0);
    cMatrix(R, n, n);

    Vector tau;
    cVector(tau, n);
    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, work.A.data(), m, tau.X.data());
    cpyMatToMat(n, n, work, R, 0, 0, 0, 0);
    LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, m, n, work.A.data(), m, tau.X.data());
    cpyMatToMat(m, n, work, Qo, 0, 0, 0, 0);
    if (nc > 0)
      cpyMatToMat(m, nc, work, Qc, 0, n, 0, 0);
    
    for (int64_t i = 0; i < n - 1; i++)
      std::fill(&R.A[i * n + i + 1], &R.A[(i + 1) * n], 0.);

    cMatrix(work, 0, 0);
    cVector(tau, 0);
  }
}

void nbd::updateSubU(Matrix& U, const Matrix& R1, const Matrix& R2) {
  if (U.M > 0 && U.N > 0) {
    int64_t m1 = R1.N;
    int64_t m2 = R2.N;
    int64_t n = U.N;
    Matrix ru1, ru2;
    cMatrix(ru1, R1.M, n);
    cMatrix(ru2, R2.M, n);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, R1.M, n, m1, 1., R1.A.data(), R1.M, U.A.data(), U.M, 0., ru1.A.data(), R1.M);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, R2.M, n, m2, 1., R2.A.data(), R2.M, U.A.data() + m1, U.M, 0., ru2.A.data(), R2.M);

    cMatrix(U, R1.M + R2.M, n);
    cpyMatToMat(R1.M, n, ru1, U, 0, 0, 0, 0);
    cpyMatToMat(R2.M, n, ru2, U, 0, 0, R1.M, 0);

    cMatrix(ru1, 0, 0);
    cMatrix(ru2, 0, 0);
  }
}

void nbd::lraID(double epi, Matrix& A, Matrix& U, int64_t arows[], int64_t* rnk_out) {
  int64_t rank = std::min(A.M, A.N);
  rank = *rnk_out > 0 ? std::min(*rnk_out, rank) : rank;
  if (Rlen < A.N * rank) { 
    fprintf(stderr, "Insufficent random vector: %" PRId64 " needed %" PRId64 " provided.", A.N * rank, Rlen);
    *rnk_out = 0;
    return;
  }
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A.M, rank, A.N, 1., A.A.data(), A.M, Rvec, A.N, 0., U.A.data(), A.M);

  Vector s, superb;
  cVector(s, rank);
  cVector(superb, s.N + 1);
  LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'O', 'N', A.M, rank, U.A.data(), A.M, s.X.data(), NULL, A.M, NULL, rank, superb.X.data());
  if (epi > 0.) {
    rank = 0;
    double sepi = s.X[0] * epi;
    while(rank < s.N && s.X[rank] > sepi)
      rank += 1;
  }
  *rnk_out = rank;

  for (int64_t i = 0; i < rank; i++)
    cblas_dscal(A.M, s.X[i], &U.A[i * A.M], 1);
  cblas_dcopy(A.M * rank, U.A.data(), 1, A.A.data(), 1);

  std::vector<int> ipiv(rank);
  int info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, A.M, rank, A.A.data(), A.M, ipiv.data());
  if (info > 0)
    rank = info - 1;
  cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, A.M, rank, 1., A.A.data(), A.M, U.A.data(), A.M);
  cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit, A.M, rank, 1., A.A.data(), A.M, U.A.data(), A.M);

  std::vector<int64_t> rows(A.M);
  std::iota(rows.begin(), rows.end(), 0);
  for (int64_t i = 0; i < rank; i++) {
    int64_t ri = ipiv[i] - 1;
    std::iter_swap(&rows[i], &rows[ri]);
    arows[i] = rows[i];
  }

  cVector(s, 0);
  cVector(superb, 0);
  ipiv.clear();
}

void nbd::zeroMatrix(Matrix& A) {
  std::fill(A.A.data(), A.A.data() + A.M * A.N, 0.);
}

void nbd::zeroVector(Vector& A) {
  std::fill(A.X.data(), A.X.data() + A.N, 0.);
}

void nbd::mmult(char ta, char tb, const Matrix& A, const Matrix& B, Matrix& C, double alpha, double beta) {
  int64_t k = (ta == 'N' || ta == 'n') ? A.N : A.M;
  if (C.M > 0 && C.N > 0 && k > 0) {
    CBLAS_TRANSPOSE tac = (ta == 'T' || ta == 't') ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE tbc = (tb == 'T' || tb == 't') ? CblasTrans : CblasNoTrans;
    cblas_dgemm(CblasColMajor, tac, tbc, C.M, C.N, k, alpha, A.A.data(), A.M, B.A.data(), B.M, beta, C.A.data(), C.M);
  }
}

void nbd::chol_decomp(Matrix& A) {
  if (A.M > 0)
    LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', A.M, A.A.data(), A.M);
}

void nbd::trsm_lowerA(Matrix& A, const Matrix& L) {
  if (A.M > 0 && L.M > 0 && L.N > 0)
    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, A.M, A.N, 1., L.A.data(), L.M, A.A.data(), A.M);
}

void nbd::utav(char tb, const Matrix& U, const Matrix& A, const Matrix& VT, Matrix& C) {
  Matrix work;
  cMatrix(work, C.M, A.N);
  if (tb == 'N' || tb == 'n') {
    mmult('T', 'N', U, A, work, 1., 0.);
    mmult('N', 'N', work, VT, C, 1., 0.);
  }
  else if (tb == 'T' || tb == 't') {
    mmult('N', 'N', U, A, work, 1., 0.);
    mmult('N', 'T', work, VT, C, 1., 0.);
  }
  cMatrix(work, 0, 0);
}

void nbd::mat_solve(char type, Vector& X, const Matrix& A) {
  if (A.M > 0 && X.N > 0) {
    if (type == 'F' || type == 'f' || type == 'A' || type == 'a')
      cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit, X.N, A.A.data(), A.M, X.X.data(), 1);
    if (type == 'B' || type == 'b' || type == 'A' || type == 'a')
      cblas_dtrsv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit, X.N, A.A.data(), A.M, X.X.data(), 1);
  }
}

void nbd::mvec(char ta, const Matrix& A, const Vector& X, Vector& B, double alpha, double beta) {
  if (A.M > 0 && A.N > 0) {
    CBLAS_TRANSPOSE tac = (ta == 'T' || ta == 't') ? CblasTrans : CblasNoTrans;
    cblas_dgemv(CblasColMajor, tac, A.M, A.N, alpha, A.A.data(), A.M, X.X.data(), 1, beta, B.X.data(), 1);
  }
}

void nbd::normalizeA(Matrix& A, const Matrix& B) {
  int64_t len_A = A.M * A.N;
  int64_t len_B = B.M * B.N;
  if (len_A > 0 && len_B > 0) {
    double nrm_A = cblas_dnrm2(len_A, A.A.data(), 1);
    double nrm_B = cblas_dnrm2(len_B, B.A.data(), 1);
    cblas_dscal(len_A, nrm_B / nrm_A, A.A.data(), 1);
  }
}

void nbd::vnrm2(const Vector& A, double* nrm) {
  *nrm = cblas_dnrm2(A.N, A.X.data(), 1);
}

void nbd::verr2(const Vector& A, const Vector& B, double* err) {
  Vector work;
  cVector(work, A.N);
  vaxpby(work, A.X.data(), 1., 0.);
  vaxpby(work, B.X.data(), -1., 1.);
  vnrm2(work, err);
  cVector(work, 0);
}
