
#pragma once
#include "linalg.hxx"
#include "kernel.h"

namespace nbd {

  struct Cell {
    int64_t NCHILD;
    int64_t NBODY;
    Cell* CHILD;
    Cell* SIBL;
    Body* BODY;
    double R[3];
    double C[3];
    int64_t ZID;
    int64_t LEVEL;
    
    std::vector<Cell*> listFar;
    std::vector<Cell*> listNear;
    std::vector<int64_t> Multipole;
  };

  typedef std::vector<Cell> Cells;

  struct CSC {
    int64_t M;
    int64_t N;
    int64_t NNZ_NEAR;
    int64_t NNZ_FAR;
    int64_t CBGN;
    std::vector<int64_t> COLS_NEAR;
    std::vector<int64_t> ROWS_NEAR;
    std::vector<int64_t> COLS_FAR;
    std::vector<int64_t> ROWS_FAR;
  };

  int64_t partition(Body* bodies, int64_t nbodies, int64_t sdim);

  int64_t buildTree(Cells& cells, Body* bodies, int64_t nbodies, int64_t levels);

  void readPartitionedBodies(const char fname[], Body* bodies, int64_t nbodies, int64_t buckets[], int64_t dim);

  void buildTreeBuckets(Cells& cells, Body* bodies, const int64_t buckets[], int64_t levels);

  void getList(Cell* Ci, Cell* Cj, double theta);

  void findCellsAtLevel(const Cell* cells[], int64_t* len, const Cell* cell, int64_t level);

  void findCellsAtLevelModify(Cell* cells[], int64_t* len, Cell* cell, int64_t level);

  const Cell* findLocalAtLevel(const Cell* cell, int64_t level);

  Cell* findLocalAtLevelModify(Cell* cell, int64_t level);

  void traverse(Cells& cells, int64_t levels, int64_t theta);

  int64_t remoteBodies(Body* remote, int64_t size, const Cell& cell, const Body* bodies, int64_t nbodies);

  void collectChildMultipoles(const Cell& cell, int64_t multipoles[]);

  void childMultipoleSize(int64_t* size, const Cell& cell);

  void relationsNear(CSC rels[], const Cells& cells);

  void evaluateLeafNear(Matrices& d, eval_func_t ef, const Cell* cell, const CSC& csc);

  void evaluateFar(Matrices& s, eval_func_t ef, const Cell* cell, const CSC& csc, int64_t level);

  void lookupIJ(char NoF, int64_t& ij, const CSC& rels, int64_t i, int64_t j);

  void loadX(Vectors& X, const Cell* cell, int64_t level);
  
  void h2MatVecReference(Vectors& B, eval_func_t ef, const Cell* root, int64_t levels);

}

