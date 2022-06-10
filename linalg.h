
#pragma once

#include "stddef.h"
#include "stdint.h"

#ifdef __cplusplus
extern "C" {
#endif
    
struct Matrix {
  double* A;
  int64_t M;
  int64_t N;
};

struct Vector {
  double* X;
  int64_t N;
};

void cRandom(int64_t lenR, double min, double max, unsigned int seed);

void cMatrix(struct Matrix* mat, int64_t m, int64_t n);

void cVector(struct Vector* vec, int64_t n);

void cpyFromMatrix(const struct Matrix* A, double* v);

void cpyFromVector(const struct Vector* A, double* v);

void maxpby(struct Matrix* A, const double* v, double alpha, double beta);

void vaxpby(struct Vector* A, const double* v, double alpha, double beta);

void cpyMatToMat(int64_t m, int64_t n, const struct Matrix* m1, struct Matrix* m2, int64_t y1, int64_t x1, int64_t y2, int64_t x2);

void cpyVecToVec(int64_t n, const struct Vector* v1, struct Vector* v2, int64_t x1, int64_t x2);

void qr_with_complements(struct Matrix* Qo, struct Matrix* Qc, struct Matrix* R);

void updateSubU(struct Matrix* U, const struct Matrix* R1, const struct Matrix* R2);

void lraID(double epi, struct Matrix* A, struct Matrix* U, int64_t arows[], int64_t* rnk_out);

void zeroMatrix(struct Matrix* A);

void zeroVector(struct Vector* A);

void mmult(char ta, char tb, const struct Matrix* A, const struct Matrix* B, struct Matrix* C, double alpha, double beta);

void chol_decomp(struct Matrix* A);

void trsm_lowerA(struct Matrix* A, const struct Matrix* L);

void utav(char tb, const struct Matrix* U, const struct Matrix* A, const struct Matrix* VT, struct Matrix* C);

void mat_solve(char type, struct Vector* X, const struct Matrix* A);

void mvec(char ta, const struct Matrix* A, const struct Vector* X, struct Vector* B, double alpha, double beta);

void normalizeA(struct Matrix* A, const struct Matrix* B);

void vnrm2(const struct Vector* A, double* nrm);
  
#ifdef __cplusplus
}
#endif
