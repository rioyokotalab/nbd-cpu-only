
#pragma once

#include "build_tree.hxx"

namespace nbd {

  struct Base {
    std::vector<int64_t> DIMS;
    std::vector<int64_t> DIML;
    Matrices Uc;
    Matrices Uo;
    Matrices Ulr;
  };

  typedef std::vector<Base> Basis;

  void allocBasis(Basis& basis, int64_t levels);

  void evaluateBasis(KerFunc_t ef, Matrix& Base, Cell* cell, const Body* bodies, int64_t nbodies, double epi, int64_t mrank, int64_t sp_pts);

  void evaluateLocal(KerFunc_t ef, Base& basis, Cell* cell, int64_t level, const Body* bodies, int64_t nbodies, double epi, int64_t mrank, int64_t sp_pts);

  void writeRemoteCoupling(const Base& basis, Cell* cell, int64_t level);

  void evaluateBaseAll(KerFunc_t ef, Base basis[], Cells& cells, int64_t levels, const Body* bodies, int64_t nbodies, double epi, int64_t mrank, int64_t sp_pts);
  
  void fillDimsFromCell(Base& basis, const Cell* cell, int64_t level);

  void allocUcUo(Base& basis, int64_t level);

  void nextBasisDims(Base& bsnext, const int64_t dimo[], int64_t nlevel);

};
