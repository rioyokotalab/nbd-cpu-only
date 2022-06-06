
#pragma once

#include "umv.hxx"

namespace nbd {

  struct RHS {
    Vectors X;
    Vectors Xc;
    Vectors Xo;
  };

  struct SpDense {
    int64_t Levels;
    std::vector<Node> D;
    std::vector<Base> Basis;
    const CSC *Rels;
  };

  typedef std::vector<RHS> RHSS;

  void basisXoc(char fwbk, RHS& vx, const Base& basis, int64_t level);

  void svAccFw(Vectors& Xc, const Matrices& A_cc, const CSC& rels, int64_t level);

  void svAccBk(Vectors& Xc, const Matrices& A_cc, const CSC& rels, int64_t level);

  void svAocFw(Vectors& Xo, const Vectors& Xc, const Matrices& A_oc, const CSC& rels, int64_t level);

  void svAocBk(Vectors& Xc, const Vectors& Xo, const Matrices& A_oc, const CSC& rels, int64_t level);

  void permuteAndMerge(char fwbk, Vectors& px, Vectors& nx, int64_t nlevel);

  void allocRightHandSides(RHS st[], const Base base[], int64_t levels);

  void solveA(RHS st[], const Node A[], const Base B[], const CSC rels[], const Vectors& X, int64_t levels);

  void allocSpDense(SpDense& sp, const CSC rels[], int64_t levels);

  void factorSpDense(SpDense& sp, const Cell* local, const Matrices& D);

  void solveSpDense(RHS st[], const SpDense& sp, const Vectors& X);

  void solveRelErr(double* err_out, const Vectors& X, const Vectors& ref, int64_t level);


};
