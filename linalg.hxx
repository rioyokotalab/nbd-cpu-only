
#pragma once

#include <vector>
#include <cstdint>

namespace nbd {
    
  struct Matrix {
    std::vector<double> A;
    int64_t M;
    int64_t N;
  };

  struct Vector {
    std::vector<double> X;
    int64_t N;
  };

  typedef std::vector<Matrix> Matrices;
  typedef std::vector<Vector> Vectors;

  void cMatrix(Matrix& mat, int64_t m, int64_t n);

  void cVector(Vector& vec, int64_t n);

  void cpyFromMatrix(char trans, const Matrix& A, double* v);

  void cpyFromVector(const Vector& A, double* v);

  void maxpby(Matrix& A, const double* v, double alpha, double beta);

  void vaxpby(Vector& A, const double* v, double alpha, double beta);

  void cpyMatToMat(int64_t m, int64_t n, const Matrix& m1, Matrix& m2, int64_t y1, int64_t x1, int64_t y2, int64_t x2);

  void cpyVecToVec(int64_t n, const Vector& v1, Vector& v2, int64_t x1, int64_t x2);

  void updateU(double epi, Matrix& A, const Matrix& U, int64_t *rnk_out);

  void subBase(Matrix& A, const Matrix& U);

  void lraID(double epi, int64_t mrank, Matrix& A, Matrix& U, int64_t arows[], int64_t* rnk_out);

  void zeroMatrix(Matrix& A);

  void zeroVector(Vector& A);

  void mmult(char ta, char tb, const Matrix& A, const Matrix& B, Matrix& C, double alpha, double beta);

  void msample(char ta, int64_t lenR, const Matrix& A, const double* R, Matrix& C);

  void msample_m(char ta, const Matrix& A, const Matrix& B, Matrix& C);

  void minvl(const Matrix& A, Matrix& B);

  void invBasis(const Matrix& u, Matrix& uinv);

  void chol_decomp(Matrix& A);

  void trsm_lowerA(Matrix& A, const Matrix& L);

  void utav(char tb, const Matrix& U, const Matrix& A, const Matrix& VT, Matrix& C);

  void chol_solve(Vector& X, const Matrix& A);

  void fw_solve(Vector& X, const Matrix& L);

  void bk_solve(Vector& X, const Matrix& L);

  void mvec(char ta, const Matrix& A, const Vector& X, Vector& B, double alpha, double beta);

  void pvc_fw(const Vector& X, const Matrix& Us, const Matrix& Uc, Vector& Xs, Vector& Xc);

  void pvc_bk(const Vector& Xs, const Vector& Xc, const Matrix& Us, const Matrix& Uc, Vector& X);

  void vnrm2(const Vector& A, double* nrm);

  void verr2(const Vector& A, const Vector& B, double* err);

};
