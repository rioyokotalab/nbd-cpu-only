
#pragma once

#include "umv.hxx"

namespace nbd {

  struct RightHandSides {
    int64_t Xlen;
    std::vector<Vector> X;
    std::vector<Vector> Xc;
    std::vector<Vector> Xo;
  };

  void basisXoc(char fwbk, RightHandSides& vx, const Base& basis, int64_t level);

  void svAccFw(Vector* Xc, const Matrix* A_cc, const CSC& rels, int64_t level);

  void svAccBk(Vector* Xc, const Matrix* A_cc, const CSC& rels, int64_t level);

  void svAocFw(Vector* Xo, const Vector* Xc, const Matrix* A_oc, const CSC& rels, int64_t level);

  void svAocBk(Vector* Xc, const Vector* Xo, const Matrix* A_oc, const CSC& rels, int64_t level);

  void permuteAndMerge(char fwbk, Vector* px, Vector* nx, int64_t nlevel);

  void allocRightHandSides(RightHandSides st[], const Base base[], int64_t levels);

  void deallocRightHandSides(RightHandSides* st, int64_t levels);

  void RightHandSides_mem(int64_t* bytes, const RightHandSides* st, int64_t levels);

  void solveA(RightHandSides st[], const Node A[], const Base B[], const CSC rels[], const Vector* X, int64_t levels);

  void solveSpDense(RightHandSides st[], const SpDense& sp, const Vector* X);

  void solveRelErr(double* err_out, const Vector* X, const Vector* ref, int64_t level);


};
