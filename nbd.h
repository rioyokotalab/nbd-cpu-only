
#pragma once

#include <vector>
#include <set>
#include <cstdint>

namespace nbd {

  typedef double real_t;
  constexpr int64_t dim = 4;

  struct Body {
    real_t X[dim];
  };

  typedef std::vector<Body> Bodies;
  typedef std::vector<real_t> Matrix;

  struct Cell {
    int64_t NCHILD = 0;
    int64_t NBODY = 0;
    Cell* CHILD = NULL;
    Body* BODY = NULL;
    real_t Xmin[dim];
    real_t Xmax[dim];

    std::vector<Cell*> listFar;
    std::vector<Cell*> listNear;

    Matrix V;
  };

  typedef std::vector<Cell> Cells;

  typedef void (*eval_func_t) (real_t&);

  typedef std::set<std::pair<Cell*, Cell*>, Matrix> Matrices;
  typedef std::set<std::pair<Cell*, Cell*>, std::pair<Matrix, Matrix>> LR_Matrices;


}
