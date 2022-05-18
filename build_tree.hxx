
#pragma once
#include "kernel.hxx"

namespace nbd {

  struct Body {
    double X[3];
    double B;
  };

  typedef std::vector<Body> Bodies;

  struct Cell {
    int64_t NCHILD;
    int64_t NBODY;
    Cell* CHILD;
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
    int64_t NNZ;
    int64_t CBGN;
    std::vector<int64_t> CSC_COLS;
    std::vector<int64_t> CSC_ROWS;
  };

  struct Base;

  void loadBodiesArray(Body* bodies, int64_t nbodies, const double arr[], int64_t dim);

  void randomUniformBodies(Body* bodies, int64_t nbodies, double dmin, double dmax, int64_t dim, int seed);

  void randomSurfaceBodies(Body* bodies, int64_t nbodies, int64_t dim, int seed);

  void randomNeutralCharge(Body* bodies, int64_t nbodies, double cmax, int seed);

  int64_t partition(Body* bodies, int64_t nbodies, int64_t sdim);

  void getBounds(const Body* bodies, int64_t nbodies, double R[], double C[], int64_t dim);

  int64_t buildTree(Cells& cells, Bodies& bodies, int64_t ncrit, int64_t dim);

  void getList(Cell* Ci, Cell* Cj, int64_t dim, int64_t theta);

  void findCellsAtLevel(const Cell* cells[], int64_t* len, const Cell* cell, int64_t level);

  void findCellsAtLevelModify(Cell* cells[], int64_t* len, Cell* cell, int64_t level);

  const Cell* findLocalAtLevel(const Cell* cell, int64_t level);

  Cell* findLocalAtLevelModify(Cell* cell, int64_t level);

  void traverse(Cells& cells, int64_t levels, int64_t dim, int64_t theta);

  void remoteBodies(Bodies& remote, int64_t size, const Cell& cell, const Bodies& bodies, int64_t dim);

  void collectChildMultipoles(const Cell& cell, int64_t multipoles[]);

  void writeChildMultipoles(Cell& cell, const int64_t multipoles[], int64_t mlen);

  void childMultipoleSize(int64_t* size, const Cell& cell);

  void relationsNear(CSC rels[], const Cells& cells);

  void evaluateLeafNear(Matrices& d, EvalFunc ef, const Cell* cell, int64_t dim, const CSC& csc);

  void lookupIJ(int64_t& ij, const CSC& rels, int64_t i, int64_t j);

  void evaluateNear(Matrices d[], EvalFunc ef, const Cells& cells, int64_t dim, const CSC rels[], const Base base[], int64_t levels);

  void loadX(Vectors& X, const Cell* cell, int64_t level);

}

