
#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

struct Base { 
  int64_t Ulen, Llen, LocalOffset, dimR, dimS, dimN;
  std::vector<int64_t> Multipoles, Offsets;
  std::vector<int64_t> Dims, DimsLr;
  struct Matrix *Uo, *Uc, *R;
  double *M_gpu, *M_cpu, *U_gpu, *U_cpu, *R_gpu, *R_cpu; 
};

struct Cell;
struct CellComm;

void initBasisLeaf(Base* skeleton, int64_t ncells, const Cell* cells, const double bodies[], const CellComm* comm, int64_t level);

void buildSkeletonsUpper(Base* skeleton, const Base* lower, const CellComm* comm);

struct SamplesFar
{ int64_t LTlen, *FarLens, *FarAvails, **FarBodies, *CloseLens, *CloseAvails, **CloseBodies, *SkeLens, **Skeletons; };

void buildSampleBodies(struct SampleBodies* sample, int64_t sp_max_far, int64_t sp_max_near, int64_t nbodies, int64_t ncells, const struct Cell* cells, 
  const struct CSC* rels, const int64_t* lt_child, const struct Base* basis_lo, int64_t level);

void buildBasis(double(*func)(double), struct Base basis[], int64_t ncells, struct Cell* cells, const struct CSC* rel_near, int64_t levels,
  const struct CellComm* comm, const double* bodies, int64_t nbodies, double epi, int64_t mrank, int64_t sp_pts, int64_t alignment);
