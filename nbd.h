
#pragma once

#include <vector>
#include <algorithm>
#include <cstddef>

namespace nbd {

  typedef double real_t;
  constexpr int dim = 4;

  typedef void (*eval_func_t) (real_t&);

  struct Body {
    real_t X[dim];
  };

  typedef std::vector<Body> Bodies;

  struct Cell {
    int NCHILD = 0;
    int NBODY = 0;
    Cell* CHILD = NULL;
    Body* BODY = NULL;
    real_t Xmin[dim];
    real_t Xmax[dim];

    std::vector<Cell*> listFar;
    std::vector<Cell*> listNear;
  };

  typedef std::vector<Cell> Cells;

  struct Matrix {
    std::vector<real_t> A;
    int M;
    int N;
    int LDA;
    Matrix(int m = 0, int n = 0, int lda = 0) : M(m), N(n), LDA(std::max(lda, m)) 
      { A.resize((size_t)LDA * N); }

    operator real_t*()
      { return A.data(); }
  };

  typedef std::vector<Matrix> Matrices;

}
