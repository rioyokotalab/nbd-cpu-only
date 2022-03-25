
#pragma once

#include "h2mv.hxx"
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

  void allocRightHandSides(RHS st[], const Base base[], int64_t levels);

  void solveA(RHS st[], const Node A[], const Base B[], const CSC rels[], const Vectors& X, int64_t levels);

  void allocSpDense(SpDense& sp, const CSC rels[], int64_t levels);

  void factorSpDense(SpDense& sp, const Cell* local, const Matrices& D, double repi, const double* R, int64_t lenR);

  void solveSpDense(RHS st[], const SpDense& sp, const Vectors& X);

  void solveH2(RHS st[], MatVec vx[], const SpDense sps[], EvalFunc ef, const Cell* root, const Base basis[], int64_t dim, const Vectors& X, int64_t levels);

  void solveRelErr(double* err_out, const Vectors& X, const Vectors& ref, int64_t level);


};
