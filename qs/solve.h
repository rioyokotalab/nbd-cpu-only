
#pragma once

#include "factorize.h"

namespace qs {

  void uxmv(char transu, const Matrix& UX, double* X);

  void fwsvcc(const Matrix& A, double* X);

  void bksvcc(const Matrix& A, double* X);

  void schcc(const Matrix& A, const double* X, double* Y);

  void schco(const Matrix& A, const double* X, double* Y);

  void schoc(const Matrix& A, const double* X, double* Y);

  void dgetrsnp(const Matrix& A, double* X);

}
