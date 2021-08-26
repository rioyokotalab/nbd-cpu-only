
#include "factorize.h"

#include "lapacke.h"
#include "cblas.h"
#include <cstddef>
#include <algorithm>

using namespace qs;

void qs::dgetrfnp(int m, int n, double* a, int lda) {
  int k = std::min(m, n);
  for (int i = 0; i < k; i++) {
    double p = 1. / a[i + (size_t)i * lda];
    int mi = m - i - 1;
    int ni = n - i - 1;

    double* ax = a + i + (size_t)i * lda + 1;
    double* ay = a + i + (size_t)i * lda + lda;
    double* an = ay + 1;

    cblas_dscal(mi, p, ax, 1);
    cblas_dger(CblasColMajor, mi, ni, -1., ax, 1, ay, lda, an, lda);
  }
}

void qs::eux(Matrix& U, Matrix& R) {
  U.LNO = U.N;
  if (U.M > U.N) {
    std::vector<double> tau(U.N);
    U.A.resize((size_t)U.LDA * U.M);

    R.M = R.N = R.LDA = U.N;
    R.A.resize((size_t)U.N * U.N);

    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, U.M, U.N, U.A.data(), U.LDA, tau.data());
    for (int i = 0; i < U.N; i++)
      cblas_dcopy(std::min(i + 1, U.M), &U.A[i * U.LDA], 1, &R.A[i * R.LDA], 1);
    LAPACKE_dorgqr(LAPACK_COL_MAJOR, U.M, U.M, U.N, U.A.data(), U.LDA, tau.data());
    U.N = U.M;
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
  dgetrfnp(LAC, LAC, a_cc, lda);

  cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, LAC, LAO, 1., a_cc, lda, a_co, lda);
  cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, LAO, LAC, 1., a_cc, lda, a_oc, lda);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, LAO, LAO, LAC, -1., a_oc, lda, a_co, lda, 1., a, lda);
  
}

void qs::ptrsmr(const Matrix& UX, const Matrix& A, Matrix& B) {

  if (B.M <= A.LMO) {
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
  
  if (B.N <= A.LMO) {
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

  if (A.N <= A.LNO || B.M <= B.LMO)
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

void qs::dlu(Matrix& A) {
  int m = A.M;
  double* a = A.A.data();
  int lda = A.LDA;

  dgetrfnp(m, m, a, lda);
}


void qs::mulrleft(const Matrix& R, Matrix& A) {
  if (A.M == R.N) {
    int m = A.M;
    int n = A.N;
    int lda = A.LDA;
    double* a = A.A.data();
    int ldr = R.LDA;
    const double* r = R.A.data();

    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, m, n, 1., r, ldr, a, lda);
  }
}

void qs::mulrright(const Matrix& R, Matrix& A) {
  if (A.N == R.M) {
    int m = A.M;
    int n = A.N;
    int lda = A.LDA;
    double* a = A.A.data();
    int ldr = R.LDA;
    const double* r = R.A.data();

    cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit, m, n, 1., r, ldr, a, lda);
  }
}
