
#pragma once

#include "build_tree.h"
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

struct Base {
  int64_t Ulen;
  std::vector<int64_t> DIMS;
  std::vector<int64_t> DIML;
  std::vector<int64_t> Multipoles;
  
  std::vector<Matrix> Uc;
  std::vector<Matrix> Uo;
  std::vector<Matrix> R;
};

void allocBasis(Base* basis, int64_t levels);

void deallocBasis(Base* basis, int64_t levels);

void basis_mem(int64_t* bytes, const Base* basis, int64_t levels);

void evaluateBaseAll(KerFunc_t ef, Base basis[], Cell* cells, const CSC* cellsNear, int64_t levels, const Body* bodies, int64_t nbodies, double epi, int64_t mrank, int64_t sp_pts);

#ifdef __cplusplus
}
#endif
