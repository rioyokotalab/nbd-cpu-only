
#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

struct Base { 
  int64_t Ulen, Llen, LocalOffset, dimR, dimS, dimN;
  std::vector<int64_t> Dims, DimsLr;
  struct Matrix *Uo, *Uc, *R;
  double *M_gpu, *M_cpu, *U_gpu, *U_cpu, *R_gpu, *R_cpu; 
};

void buildBasis(double(*func)(double), struct Base basis[], int64_t ncells, struct Cell* cells, const struct CSC* rel_near, int64_t levels,
  const struct CellComm* comm, const double* bodies, int64_t nbodies, double epi, int64_t mrank, int64_t sp_pts, int64_t alignment);

void basis_free(struct Base* basis);

