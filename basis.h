
#pragma once

#include "build_tree.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Base {
  int64_t Ulen;
  int64_t* Lchild;
  int64_t* DIMS;
  int64_t* DIML;
  int64_t* Multipoles;
  
  struct Matrix* Uc;
  struct Matrix* Uo;
  struct Matrix* R;
};

void allocBasis(Base* basis, int64_t levels, int64_t ncells, const struct Cell* cells, const struct CellComm* comm);

void deallocBasis(Base* basis, int64_t levels);

void basis_mem(int64_t* bytes, const Base* basis, int64_t levels);

void evaluateBaseAll(void(*ef)(double*), Base basis[], Cell* cells, const CSC* cellsNear, int64_t levels, const Body* bodies, int64_t nbodies, double epi, int64_t mrank, int64_t sp_pts);

#ifdef __cplusplus
}
#endif
