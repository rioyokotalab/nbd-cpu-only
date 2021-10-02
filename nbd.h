
#pragma once

#include <vector>
#include <algorithm>
#include <cstddef>

namespace nbd {

  typedef double real_t;
  constexpr int dim = 4;

  typedef void (*eval_func_t) (real_t&, real_t, real_t);

  struct EvalFunc {
    eval_func_t r2f;
    real_t singularity;
    real_t alpha;
  };

  struct Body {
    real_t X[dim];
  };

  typedef std::vector<Body> Bodies;

  struct Cell {
    int NCHILD = 0;
    int NBODY = 0;
    Cell* CHILD = NULL;
    Body* BODY = NULL;
    real_t C[dim];
    real_t R[dim];

    std::vector<Cell*> listFar;
    std::vector<Cell*> listNear;
  };

  typedef std::vector<Cell> Cells;

  struct Matrix {
    std::vector<real_t> A;
    int M = 0;
    int N = 0;
  };

  typedef std::vector<Matrix> Matrices;

}
