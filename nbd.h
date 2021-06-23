
#pragma once

#include <vector>
#include <set>

namespace nbd {

  typedef double real_t;
  constexpr int dim = 4;

  struct Body {
    real_t X[dim];
  };

  typedef std::vector<Body> Bodies;
  typedef std::vector<real_t> Matrix;

  struct Cell {
    int NCHILD = 0;
    int NBODY = 0;
    Cell* CHILD = NULL;
    Body* BODY = NULL;
    real_t Xmin[dim];
    real_t Xmax[dim];

    std::vector<Cell*> listFar;
    std::vector<Cell*> listNear;

    Bodies inner;
    Matrix V;
  };

  typedef std::vector<Cell> Cells;

  typedef void (*eval_func_t) (real_t&);

  typedef std::set<std::pair<Cell*, Cell*>, Matrix> Matrices;


}
