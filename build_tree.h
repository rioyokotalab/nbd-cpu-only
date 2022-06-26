
#pragma once

#include "linalg.h"
#include "kernel.h"
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

struct Cell {
  int64_t CHILD;
  int64_t BODY[2];
  double R[DIM_MAX];
  double C[DIM_MAX];
  
  std::vector<int64_t> Multipole;
};

struct CSC {
  int64_t M;
  int64_t N;
  int64_t* COL_INDEX;
  int64_t* ROW_INDEX;
};

void buildTree(Cell* cells, Body* bodies, int64_t nbodies, int64_t levels);

void traverse(char NoF, CSC* rels, int64_t ncells, const Cell* cells, double theta);

void traverse_dist(const CSC* cellFar, const CSC* cellNear, int64_t levels);

void relations(CSC rels[], const CSC* cellRel, int64_t levels);

void evaluate(char NoF, Matrix* s, KerFunc_t ef, const Cell* cell, const Body* bodies, const CSC* csc, int64_t level);

void lookupIJ(int64_t* ij, const CSC* rels, int64_t i, int64_t j);

void loadX(Matrix* X, const Cell* cell, const Body* bodies, int64_t level);

void h2MatVecReference(Matrix* B, KerFunc_t ef, const Cell* cell, const Body* bodies, int64_t level);

#ifdef __cplusplus
}
#endif

