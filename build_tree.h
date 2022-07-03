
#pragma once

#include "linalg.h"
#include "kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Cell {
  int64_t CHILD;
  int64_t BODY[2];
  double R[DIM_MAX];
  double C[DIM_MAX];

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
  int64_t Proc; // Local Proc num. Used by searching.
  struct CSC Comms; // P by P sparse. Entry (i, j) proc i comm with j.
  int64_t* ProcMerge; // len P. Proc i merge all procs within [PM[i], PMEnd[i]).
  int64_t* ProcMergeEnd; // len P. Proc i merge all procs within [PM[i], PMEnd[i]).
  int64_t* ProcBoxes; // len P. Proc i hold boxes within [PB[i], PBEnd[i]) as LET.
  int64_t* ProcBoxesEnd; // len P. Proc i hold boxes within [PB[i], PBEnd[i]) as LET.
};

void buildTree(int64_t* ncells, struct Cell* cells, struct Body* bodies, int64_t nbodies, int64_t levels, int64_t mpi_size);

void traverse(char NoF, struct CSC* rels, int64_t ncells, const struct Cell* cells, double theta);

void get_level(int64_t* begin, int64_t* end, const struct Cell* cells, int64_t level);

void buildComm(struct CellComm* comms, int64_t ncells, const struct Cell* cells, const struct CSC* cellFar, const struct CSC* cellNear, int64_t levels, int64_t mpi_rank, int64_t mpi_size);

void traverse_dist(const struct CSC* cellFar, const struct CSC* cellNear, int64_t levels);

void relations(struct CSC rels[], const struct CSC* cellRel, int64_t levels);

void evaluate(char NoF, struct Matrix* s, KerFunc_t ef, const struct Cell* cell, const struct Body* bodies, const struct CSC* csc, int64_t level);

void lookupIJ(int64_t* ij, const struct CSC* rels, int64_t i, int64_t j);

void i_local(int64_t* ilocal, int64_t iglobal, const struct CellComm* comm);

void i_global(int64_t* iglobal, int64_t ilocal, const struct CellComm* comm);

void remoteBodies(int64_t* remote, int64_t size[], int64_t nlen, const int64_t ngbs[], const struct Cell* cells, int64_t ci);

void evaluateBasis(KerFunc_t ef, double epi, int64_t* rank, struct Matrix* Base, int64_t m, int64_t n[], int64_t cellm[], const int64_t remote[], const struct Body* bodies);

void loadX(struct Matrix* X, const struct Cell* cell, const struct Body* bodies, int64_t level);

void h2MatVecReference(struct Matrix* B, KerFunc_t ef, const struct Cell* cell, const struct Body* bodies, int64_t level);

#ifdef __cplusplus
}
#endif

