
#pragma once

#include "umv.h"

#ifdef __cplusplus
extern "C" {
#endif

struct RightHandSides {
  int64_t Xlen;
  std::vector<Matrix> X;
  std::vector<Matrix> Xc;
  std::vector<Matrix> Xo;
};

void basisXoc(char fwbk, RightHandSides& vx, const Base& basis, int64_t level);

void svAccFw(Matrix* Xc, const Matrix* A_cc, const CSC& rels, int64_t level);

void svAccBk(Matrix* Xc, const Matrix* A_cc, const CSC& rels, int64_t level);

void svAocFw(Matrix* Xo, const Matrix* Xc, const Matrix* A_oc, const CSC& rels, int64_t level);

void svAocBk(Matrix* Xc, const Matrix* Xo, const Matrix* A_oc, const CSC& rels, int64_t level);

void permuteAndMerge(char fwbk, Matrix* px, Matrix* nx, int64_t nlevel);

void allocRightHandSides(RightHandSides st[], const Base base[], int64_t levels);

void deallocRightHandSides(RightHandSides* st, int64_t levels);

void RightHandSides_mem(int64_t* bytes, const RightHandSides* st, int64_t levels);

void solveA(RightHandSides st[], const Node A[], const Base B[], const CSC rels[], const Matrix* X, int64_t levels);

void solveSpDense(RightHandSides st[], const SpDense& sp, const Matrix* X);

void solveRelErr(double* err_out, const Matrix* X, const Matrix* ref, int64_t level);

#ifdef __cplusplus
}
#endif
