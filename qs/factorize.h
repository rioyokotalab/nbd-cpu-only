
#pragma once

#include <vector>

namespace qs {

  struct Matrix {
    std::vector<double> A;
    int M;
    int N;
    int LDA;
    int LMO;
    int LNO;
  };

  void eux(Matrix& U);

  void plu(const Matrix& UX, Matrix& M);

  void ptrsmr(const Matrix& UX, const Matrix& A, Matrix& B);

  void ptrsmc(const Matrix& UX, const Matrix& A, Matrix& B);

  void pgemm(const Matrix& A, const Matrix& B, Matrix& C);

  void dlu(Matrix& M);


}

