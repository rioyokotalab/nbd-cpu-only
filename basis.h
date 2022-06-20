
#pragma once

#include "build_tree.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Base {
  int64_t Ulen;
  std::vector<int64_t> DIMS;
  std::vector<int64_t> DIML;
  std::vector<Matrix> Uc;
  std::vector<Matrix> Uo;
  std::vector<Matrix> R;
};

void allocBasis(Base* basis, int64_t levels);

void deallocBasis(Base* basis, int64_t levels);

void basis_mem(int64_t* bytes, const Base* basis, int64_t levels);

void evaluateBasis(KerFunc_t ef, double epi, int64_t* rank, Matrix& Base, int64_t m, int64_t n1, int64_t n2, 
  int64_t cellm[], const int64_t remote[], const int64_t close[], const Body* bodies);

void evaluateLocal(KerFunc_t ef, Base& basis, Cell* cell, int64_t level, const Body* bodies, int64_t nbodies, double epi, int64_t mrank, int64_t sp_pts);

void writeRemoteCoupling(const Base& basis, Cell* cell, int64_t level);

void evaluateBaseAll(KerFunc_t ef, Base basis[], Cell* cells, int64_t levels, const Body* bodies, int64_t nbodies, double epi, int64_t mrank, int64_t sp_pts);

void orth_base_all(Base* basis, int64_t levels);

#ifdef __cplusplus
}
#endif
