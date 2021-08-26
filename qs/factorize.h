
#pragma once

#include "qs.h"

namespace qs {

  void dgetrfnp(int m, int n, double* a, int lda);

  void eux(Matrix& U, Matrix& R);

  void plu(const Matrix& UX, Matrix& M);

  void ptrsmr(const Matrix& UX, const Matrix& A, Matrix& B);

  void ptrsmc(const Matrix& UX, const Matrix& A, Matrix& B);

  void pgemm(const Matrix& A, const Matrix& B, Matrix& C);

  void dlu(Matrix& M);

  void mulrleft(const Matrix& R, Matrix& A);

  void mulrright(const Matrix& R, Matrix& A);




}

