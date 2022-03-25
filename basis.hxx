
#pragma once

#include "build_tree.hxx"

namespace nbd {

  struct Base {
    std::vector<int64_t> DIMS;
    std::vector<int64_t> DIMO;
    Matrices Uc;
    Matrices Uo;
  };

  typedef std::vector<Base> Basis;

  void sampleC1(Matrices& C1, const CSC& rels, const Matrices& A, const double* R, int64_t lenR, int64_t level);

  void sampleC2(Matrices& C2, const CSC& rels, const Matrices& A, const Matrices& C1, int64_t level);

  void orthoBasis(double repi, Matrices& C, int64_t dims_o[], int64_t level);

  void allocBasis(Basis& basis, int64_t levels);
  
  void fillDimsFromCell(Base& basis, const Cell* cell, int64_t level);

  void allocUcUo(Base& basis, const Matrices& C, int64_t level);

  void sampleA(Base& basis, double repi, const CSC& rels, const Matrices& A, const double* R, int64_t lenR, int64_t level);

  void nextBasisDims(Base& bsnext, const Base& bsprev, int64_t nlevel);

};
