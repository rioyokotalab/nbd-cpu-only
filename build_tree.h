
#pragma once

#include "mpi.h"
#include "stdint.h"
#include "stddef.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Body;
struct Matrix;

struct Cell {
  int64_t CHILD;
  int64_t BODY[2];
  double R[3];
  double C[3];

  int64_t LEVEL;
  int64_t Procs[2];
  
  int64_t lenMultipole;
  int64_t* Multipole;
};

struct CSC {
  int64_t M;
  int64_t N;
  int64_t* COL_INDEX;
  int64_t* ROW_INDEX;
};

struct CellComm {
  struct CSC Comms; // P by P sparse. Entry (i, j) proc i comm with j.
  int64_t* ProcRootI; // len P. The PRI[i]'th element in comms is Entry (i, i) on row/column i.
  int64_t* ProcBoxes; // len P. Proc i hold boxes within [PB[i], PBEnd[i]) as LET.
  int64_t* ProcBoxesEnd; // len P. Proc i hold boxes within [PB[i], PBEnd[i]) as LET.

  int64_t Proc[2]; // Local Proc num used by searching. Same content stored in range [P0, P1).
  MPI_Comm* Comm_box; // len P. CB[i] is used when Proc i broadcasts.
  MPI_Comm Comm_merge; // Local comm for merged procs.
};

struct Base {
  int64_t Ulen;
  int64_t* Lchild;
  int64_t* DIMS;
  int64_t* DIML;
  int64_t* Offsets;
  int64_t* Multipoles;
  
  struct Matrix* Uo;
  struct Matrix* Uc;
  struct Matrix* R;
};

void buildTree(int64_t* ncells, struct Cell* cells, struct Body* bodies, int64_t nbodies, int64_t levels);

void traverse(char NoF, struct CSC* rels, int64_t ncells, const struct Cell* cells, double theta);

void get_level(int64_t* begin, int64_t* end, const struct Cell* cells, int64_t level, int64_t mpi_rank);

void buildComm(struct CellComm* comms, int64_t ncells, const struct Cell* cells, const struct CSC* cellFar, const struct CSC* cellNear, int64_t levels);

void cellComm_free(struct CellComm* comms, int64_t levels);

void lookupIJ(int64_t* ij, const struct CSC* rels, int64_t i, int64_t j);

void i_local(int64_t* ilocal, const struct CellComm* comm);

void i_global(int64_t* iglobal, const struct CellComm* comm);

void self_local_range(int64_t* ibegin, int64_t* iend, const struct CellComm* comm);

void content_length(int64_t* len, const struct CellComm* comm);

void relations(struct CSC rels[], int64_t ncells, const struct Cell* cells, const struct CSC* cellRel, int64_t levels);

void allocBasis(struct Base* basis, int64_t levels, int64_t ncells, const struct Cell* cells, const struct CellComm* comm);

void deallocBasis(struct Base* basis, int64_t levels);

void basis_mem(int64_t* bytes, const struct Base* basis, int64_t levels);

void evaluateBaseAll(void(*ef)(double*), struct Base basis[], int64_t ncells, struct Cell* cells, const struct CSC* rel_near, int64_t levels, const struct CellComm* comm, const struct Body* bodies, int64_t nbodies, double epi, int64_t mrank, int64_t sp_pts);

void evaluate(char NoF, struct Matrix* d, void(*ef)(double*), int64_t ncells, const struct Cell* cells, const struct Body* bodies, const struct CSC* csc, int64_t level);

void solveRelErr(double* err_out, const struct Matrix* X, const struct Matrix* ref, const struct CellComm* comm);

#ifdef __cplusplus
}
#endif

