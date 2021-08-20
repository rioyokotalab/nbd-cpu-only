
#include "solve.h"

#include "lapacke.h"
#include "cblas.h"
#include <cstddef>

using namespace qs;

void qs::uxmv(char transu, const Matrix& UX, double* X) {
  int n = UX.M; 
  std::vector<double> work(n);
  auto c_transu = transu == 'N' ? CblasNoTrans : CblasTrans;
  const double* ux = UX.A.data();
  int ldu = UX.LDA;

  cblas_dgemv(CblasColMajor, c_transu, n, n, 1., ux, ldu, X, 1, 0., work.data(), 1);
  cblas_dcopy(n, work.data(), 1, X, 1);
}

void qs::fwsvcc(const Matrix& A, double* X) {

  int n = A.M;
  int LAO = A.LMO;
  int LAC = n - LAO;
  int lda = A.LDA;

  const double* a_oc = A.A.data() + (size_t)A.LMO * lda;
  const double* a_cc = a_oc + A.LMO;
  double* X_c = X + A.LMO;

  cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit, LAC, a_cc, lda, X_c, 1);
  cblas_dgemv(CblasColMajor, CblasNoTrans, LAO, LAC, -1., a_oc, lda, X_c, 1, 1., X, 1);

}


void qs::bksvcc(const Matrix& A, double* X) {

  int n = A.M;
  int LAO = A.LMO;
  int LAC = n - LAO;
  int lda = A.LDA;

  const double* a_co = A.A.data() + A.LMO;
  const double* a_cc = A.A.data() + (size_t)A.LMO * lda + A.LMO;
  double* X_c = X + A.LMO;

  cblas_dgemv(CblasColMajor, CblasNoTrans, LAC, LAO, -1., a_co, lda, X, 1, 1., X_c, 1);
  cblas_dtrsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, LAC, a_cc, lda, X_c, 1);

}


void qs::schcc(const Matrix& A, const double* X, double* Y) {
  int m = A.M - A.LMO;
  int n = A.N - A.LNO;
  int lda = A.LDA;
  const double* a_cc = A.A.data() + (size_t)A.LNO * lda + A.LMO;
  const double* X_c = X + A.LNO;
  double* Y_c = Y + A.LMO;

  if (m > 0 && n > 0)
    cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, -1., a_cc, lda, X_c, 1, 1., Y_c, 1);
}


void qs::schco(const Matrix& A, const double* X, double* Y) {
  int m = A.M - A.LMO;
  int n = A.LNO;
  int lda = A.LDA;
  const double* a_co = A.A.data() + A.LMO;
  double* Y_c = Y + A.LMO;

  if (m > 0 && n > 0)
    cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, -1., a_co, lda, X, 1, 1., Y_c, 1);
}


void qs::schoc(const Matrix& A, const double* X, double* Y) {
  int m = A.LMO;
  int n = A.N - A.LNO;
  int lda = A.LDA;
  const double* a_oc = A.A.data() + (size_t)A.LNO * lda;
  const double* X_c = X + A.LNO;

  if (m > 0 && n > 0)
    cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, -1., a_oc, lda, X_c, 1, 1., Y, 1);
}


void qs::dgetrsnp(const Matrix& A, double* X) {

  int n = A.M;
  int lda = A.LDA;
  const double* a = A.A.data();

  cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit, n, a, lda, X, 1);
  cblas_dtrsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, a, lda, X, 1);
}
