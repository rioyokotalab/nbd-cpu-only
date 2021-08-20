
#pragma once

#include "qs.h"

namespace qs {

  void eux(Matrix& U);

  void plu(const Matrix& UX, Matrix& M);

  void ptrsmr(const Matrix& UX, const Matrix& A, Matrix& B);

  void ptrsmc(const Matrix& UX, const Matrix& A, Matrix& B);

  void pgemm(const Matrix& A, const Matrix& B, Matrix& C);

  void dlu(Matrix& M);



}

