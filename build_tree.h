
#pragma once

#include "linalg.h"
#include "kernel.h"
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

struct Cell {
  int64_t NCHILD;
  int64_t BODY[2];
  Cell* CHILD;
  Cell* SIBL;
  double R[3];
  double C[3];
  int64_t ZID;
  int64_t LEVEL;
  
  std::vector<Cell*> listFar;
  std::vector<Cell*> listNear;
  std::vector<int64_t> Multipole;
};

struct CSC {
  int64_t M;
  int64_t N;
  std::vector<int64_t> COL_INDEX;
  std::vector<int64_t> ROW_INDEX;
};

void buildTree(Cell* cells, Body* bodies, int64_t nbodies, int64_t levels);

void readPartitionedBodies(const char fname[], Body* bodies, int64_t nbodies, int64_t buckets[], int64_t dim);

void buildTreeBuckets(Cell* cells, Body* bodies, const int64_t buckets[], int64_t levels);

void getList(Cell* Ci, Cell* Cj, double theta);

void findCellsAtLevel(const Cell* cells[], int64_t* len, const Cell* cell, int64_t level);

void findCellsAtLevelModify(Cell* cells[], int64_t* len, Cell* cell, int64_t level);

const Cell* findLocalAtLevel(const Cell* cell, int64_t level);

Cell* findLocalAtLevelModify(Cell* cell, int64_t level);

void traverse(Cell* cells, int64_t levels, int64_t theta);

int64_t remoteBodies(int64_t* remote, int64_t size, const Cell& cell, int64_t nbodies);

int64_t closeBodies(int64_t* remote, int64_t size, const Cell& cell);

void collectChildMultipoles(const Cell& cell, int64_t multipoles[]);

void childMultipoleSize(int64_t* size, const Cell& cell);

void relations(char NoF, CSC rels[], const Cell* cells, int64_t levels);

void evaluate(char NoF, Matrix* s, KerFunc_t ef, const Cell* cell, const Body* bodies, const CSC& csc, int64_t level);

void lookupIJ(int64_t& ij, const CSC& rels, int64_t i, int64_t j);

void loadX(Vector* X, const Cell* cell, const Body* bodies, int64_t level);

void h2MatVecReference(Vector* B, KerFunc_t ef, const Cell* cell, const Body* bodies, int64_t nbodies, int64_t level);

#ifdef __cplusplus
}
#endif

