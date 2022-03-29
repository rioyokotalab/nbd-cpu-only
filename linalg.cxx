

#include "linalg.hxx"
#include "minblas.h"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstdlib>

using namespace nbd;

void nbd::cMatrix(Matrix& mat, int64_t m, int64_t n) {
  mat.A.resize(m * n);
  mat.M = m;
  mat.N = n;
}

void nbd::cVector(Vector& vec, int64_t n) {
  vec.X.resize(n);
  vec.N = n;
}

void nbd::cpyFromMatrix(char trans, const Matrix& A, double* V) {
  int64_t iv = A.M;
  int64_t incv = 1;
  if (trans == 'T' || trans == 't') {
    iv = 1;
    incv = A.N;
  }
  for (int64_t j = 0; j < A.N; j++)
    Cdcopy(A.M, &A.A[j * A.M], 1, &V[j * iv], incv);
}

void nbd::cpyFromVector(const Vector& A, double* v) {
  std::copy(A.X.data(), A.X.data() + A.N, v);
}

void nbd::maxpby(Matrix& A, const double* v, double alpha, double beta) {
  int64_t size = A.M * A.N;
  if (beta == 0.)
    std::fill(A.A.data(), A.A.data() + size, 0.);
  else if (beta != 1.)
    Cdscal(size, beta, A.A.data(), 1);
  Cdaxpy(size, alpha, v, 1, A.A.data(), 1);
}

void nbd::vaxpby(Vector& A, const double* v, double alpha, double beta) {
  int64_t size = A.N;
  if (beta == 0.)
    std::fill(A.X.data(), A.X.data() + size, 0.);
  else if (beta != 1.)
    Cdscal(size, beta, A.X.data(), 1);
  Cdaxpy(size, alpha, v, 1, A.X.data(), 1);
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

void nbd::orthoBase(double epi, int64_t mrank, Matrix& A, int64_t *rnk_out) {
  Matrix U, V;
  int64_t rank = mrank;
  rank = std::min(A.M, rank);
  rank = std::min(A.N, rank);
  cMatrix(U, A.M, rank);
  cMatrix(V, A.N, rank);

  dlra(epi, A.M, A.N, rank, A.A.data(), U.A.data(), A.M, V.A.data(), A.N, rnk_out, NULL);
  rank = *rnk_out;

  if (A.N < A.M)
    cMatrix(A, A.M, A.M);
  dorth('F', A.M, rank, U.A.data(), A.M, A.A.data(), A.M);
}

void nbd::lraID(double epi, int64_t mrank, Matrix& A, Matrix& U, int64_t arows[], int64_t* rnk_out) {
  int64_t rank = mrank;
  rank = std::min(A.M, rank);
  rank = std::min(A.N, rank);
  didrow(epi, A.M, A.N, rank, A.A.data(), U.A.data(), A.M, arows, rnk_out);
}

void nbd::zeroMatrix(Matrix& A) {
  std::fill(A.A.data(), A.A.data() + A.M * A.N, 0.);
}

void nbd::zeroVector(Vector& A) {
  std::fill(A.X.data(), A.X.data() + A.N, 0.);
}

void nbd::mmult(char ta, char tb, const Matrix& A, const Matrix& B, Matrix& C, double alpha, double beta) {
  int64_t k = (ta == 'N' || ta == 'n') ? A.N : A.M;
  if (C.M > 0 && C.N > 0 && k > 0)
    Cdgemm(ta, tb, C.M, C.N, k, alpha, A.A.data(), A.M, B.A.data(), B.M, beta, C.A.data(), C.M);
}

void nbd::msample(char ta, int64_t lenR, const Matrix& A, const double* R, Matrix& C) {
  if (lenR < C.N * 100) { 
    std::cerr << "Insufficent random vector: " << C.N << " x 100 needed " << lenR << " provided." << std::endl;
    return;
  }
  int64_t k = A.M;
  int64_t inca = 1;
  if (ta == 'N' || ta == 'n') {
    k = A.N;
    inca = A.M;
  }

  int64_t rk = lenR / C.N;
  int64_t lk = k % rk;
  if (lk > 0)
    Cdgemm(ta, 'N', C.M, C.N, lk, 1., A.A.data(), A.M, R, lk, 1., C.A.data(), C.M);
  if (k > rk)
    for (int64_t i = lk; i < k; i += rk)
      Cdgemm(ta, 'N', C.M, C.N, rk, 1., &A.A[i * inca], A.M, R, rk, 1., C.A.data(), C.M);
}

void nbd::msample_m(char ta, const Matrix& A, const Matrix& B, Matrix& C) {
  int64_t k = A.M;
  if (ta == 'N' || ta == 'n')
    k = A.N;
  int64_t nrhs = std::min(B.N, C.N);
  Cdgemm(ta, 'N', C.M, nrhs, k, 1., A.A.data(), A.M, B.A.data(), B.M, 1., C.A.data(), C.M);
}

void nbd::minvl(const Matrix& A, Matrix& B) {
  Matrix work;
  cMatrix(work, A.M, A.N);
  cpyMatToMat(A.M, A.N, A, work, 0, 0, 0, 0);
  chol_decomp(work);
  dtrsml_left(B.M, B.N, work.A.data(), A.M, B.A.data(), B.M);
  dtrsmlt_left(B.M, B.N, work.A.data(), A.M, B.A.data(), B.M);
}

void nbd::invBasis(const Matrix& u, Matrix& uinv) {
  int64_t m = u.M;
  int64_t n = u.N;
  if (m > 0 && n > 0) {
    Matrix a;
    Matrix q;
    cMatrix(a, n, n);
    cMatrix(q, n, n);
    cMatrix(uinv, n, m);

    mmult('T', 'N', u, u, a, 1., 0.);
    dorth('F', n, n, a.A.data(), n, q.A.data(), n);
    mmult('T', 'T', q, u, uinv, 1., 0.);
    dtrsmr_left(n, m, a.A.data(), n, uinv.A.data(), n);
  }
}

void nbd::chol_decomp(Matrix& A) {
  if (A.M > 0)
    Cdpotrf(A.M, A.A.data(), A.M);
}

void nbd::trsm_lowerA(Matrix& A, const Matrix& L) {
  if (A.M > 0 && L.M > 0)
    dtrsmlt_right(A.M, A.N, L.A.data(), L.M, A.A.data(), A.M);
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
}

void nbd::axat(Matrix& A, Matrix& AT) {
  if (A.M > 0 && A.N > 0) {
    for (int64_t j = 0; j < A.N; j++)
      Cdaxpy(A.M, 1., &AT.A[j], AT.M, &A.A[j * A.M], 1);
    for (int64_t i = 0; i < A.M; i++)
      Cdcopy(A.N, &A.A[i], A.M, &AT.A[i * AT.M], 1);
  }
}

void nbd::madd(Matrix& A, const Matrix& B) {
  int64_t size = A.M * A.N;
  if (size > 0)
    Cdaxpy(size, 1., &B.A[0], 1, &A.A[0], 1);
}

void nbd::chol_solve(Vector& X, const Matrix& A) {
  fw_solve(X, A);
  bk_solve(X, A);
}

void nbd::fw_solve(Vector& X, const Matrix& L) {
  if (L.M > 0 && X.N > 0)
    dtrsml_left(X.N, 1, L.A.data(), L.M, X.X.data(), X.N);
}

void nbd::bk_solve(Vector& X, const Matrix& L) {
  if (L.M > 0 && X.N > 0)
    dtrsmlt_left(X.N, 1, L.A.data(), L.M, X.X.data(), X.N);
}

void nbd::mvec(char ta, const Matrix& A, const Vector& X, Vector& B, double alpha, double beta) {
  if (A.M > 0 && A.N > 0)
    Cdgemv(ta, A.M, A.N, alpha, A.A.data(), A.M, X.X.data(), 1, beta, B.X.data(), 1);
}

void nbd::pvc_fw(const Vector& X, const Matrix& Us, const Matrix& Uc, Vector& Xs, Vector& Xc) {
  mvec('T', Uc, X, Xc, 1., 0.);
  mvec('T', Us, X, Xs, 1., 0.);
}

void nbd::pvc_bk(const Vector& Xs, const Vector& Xc, const Matrix& Us, const Matrix& Uc, Vector& X) {
  mvec('N', Uc, Xc, X, 1., 0.);
  mvec('N', Us, Xs, X, 1., 1.);
}

void nbd::vnrm2(const Vector& A, double* nrm) {
  Cdnrm2(A.N, A.X.data(), 1, nrm);
}

void nbd::verr2(const Vector& A, const Vector& B, double* err) {
  Vector work;
  cVector(work, A.N);
  vaxpby(work, A.X.data(), 1., 0.);
  vaxpby(work, B.X.data(), -1., 1.);
  vnrm2(work, err);
}
