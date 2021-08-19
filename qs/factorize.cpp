
#include "factorize.h"

#include "lapacke.h"
#include "cblas.h"
#include <cstddef>

using namespace qs;

void qs::eux(Matrix& U) {
  U.LNO = U.N;
  if (U.M > U.N) {
    std::vector<double> tau(U.N);
    U.A.resize((size_t)U.LDA * U.M);
    U.N = U.M;

    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, U.M, U.N, U.A.data(), U.LDA, tau.data());
    LAPACKE_dorgqr(LAPACK_COL_MAJOR, U.M, U.M, U.N, U.A.data(), U.LDA, tau.data());
  }
}

void qs::plu(const Matrix& UX, Matrix& A) {
  int m = A.M;
  int lda = A.LDA;
  double* a = A.A.data();
  const double* ux = UX.M > 0 && UX.N > 0 ? UX.A.data() : nullptr;

  if (ux) {
    std::vector<double> work((size_t)m * m);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, m, m, 1., ux, UX.LDA, a, lda, 0., work.data(), m);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m, m, 1., work.data(), m, ux, UX.LDA, 0., a, lda);

    A.LMO = A.LNO = UX.LNO;
  }

  double* a_oc = a + (size_t)A.LMO * lda;
  double* a_cc = a_oc + A.LMO;
  double* a_co = a + A.LMO;

  int LAO = A.LMO;
  int LAC = m - LAO;

  std::vector<int> ipiv(LAC);
  LAPACKE_dgetrf(LAPACK_COL_MAJOR, LAC, LAC, a_cc, lda, ipiv.data());

  cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, LAC, LAO, 1., a_cc, lda, a_co, lda);
  cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, LAO, LAC, 1., a_cc, lda, a_oc, lda);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, LAO, LAO, LAC, -1., a_oc, lda, a_co, lda, 1., a_oc, lda);
  
}

void qs::ptrsmr(const Matrix& UX, const Matrix& A, Matrix& B) {

  if (B.M == A.LMO) {
    B.LMO = A.LMO;
    return;
  }

  int m = A.M;
  int n = B.LNO > 0 ? B.LNO : B.N;
  int lda = A.LDA;
  int ldb = B.LDA;
  const double* a = A.A.data();
  double* b = B.A.data();
  const double* ux = UX.M > 0 && UX.N > 0 ? UX.A.data() : nullptr;

  if (ux) {
    std::vector<double> work((size_t)m * n);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, n, m, 1., ux, UX.LDA, b, ldb, 0., work.data(), m);
    for (int j = 0; j < n; j++)
      cblas_dcopy(m, &work[j * m], 1, b + j * ldb, 1);

    B.LMO = UX.LNO;
  }
  
  double* b_cx = b + B.LMO;
  const double* a_oc = a + (size_t)A.LMO * lda;
  const double* a_cc = a_oc + A.LMO;

  int LAO = A.LMO;
  int LAC = m - LAO;

  cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, LAC, n, 1., a_cc, lda, b_cx, ldb);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, LAO, n, LAC, -1., a_oc, lda, b_cx, ldb, 1., b, ldb);

}

void qs::ptrsmc(const Matrix& UX, const Matrix& A, Matrix& B) {
  
  if (B.N == A.LMO) {
    B.LNO = A.LMO;
    return;
  }

  int m = B.LMO > 0 ? B.LMO : B.M;
  int n = A.M;
  int lda = A.LDA;
  int ldb = B.LDA;
  const double* a = A.A.data();
  double* b = B.A.data();
  const double* ux = UX.M > 0 && UX.N > 0 ? UX.A.data() : nullptr;

  if (ux) {
    std::vector<double> work((size_t)m * n);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, n, 1., b, ldb, ux, UX.LDA, 0., work.data(), m);
    for (int j = 0; j < n; j++)
      cblas_dcopy(m, &work[j * m], 1, b + j * ldb, 1);

    B.LNO = UX.LNO;
  }
  
  double* b_xc = b + (size_t)B.LNO * ldb;
  const double* a_co = a + A.LMO;
  const double* a_cc = a + (size_t)A.LMO * lda + A.LMO;

  int LAO = A.LMO;
  int LAC = m - LAO;

  cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, m, LAC, 1., a_cc, lda, b_xc, ldb);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, LAO, LAC, -1., b_xc, ldb, a_co, lda, 1., b, ldb);

}

void qs::pgemm(const Matrix& A, const Matrix& B, Matrix& C) {

  if (A.M == C.LMO || B.N == C.LNO)
    return;
  
  int m = C.LMO > 0 ? C.LMO : C.M;
  int n = C.LNO > 0 ? C.LNO : C.N;
  int k = A.N - A.LNO;
  int lda = A.LDA;
  int ldb = B.LDA;
  int ldc = C.LDA;

  const double* a_xc = A.A.data() + (size_t)A.LNO * lda;
  const double* b_cx = B.A.data() + B.LMO;
  double* c = C.A.data();

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, -1., a_xc, lda, b_cx, ldb, 1., c, ldc);
}

void dlu(Matrix& A) {
  int m = A.M;
  double* a = A.A.data();
  int lda = A.LDA;

  std::vector<int> ipiv(m);
  LAPACKE_dgetrf(LAPACK_COL_MAJOR, m, m, a, lda, ipiv.data());
}

