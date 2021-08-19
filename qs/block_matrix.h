#pragma once

#include "../nbd.h"
#include "factorize.h"

namespace qs {

  typedef std::vector<Matrix> Matrices;

  struct H2Matrix {
    int N;
    Matrices D;
  };

  void matcpy(const nbd::Matrix& m1, Matrix& m2);

  Matrices convert(const nbd::Matrices& d);

  H2Matrix build(nbd::eval_func_t eval, int dim, const nbd::Cells& cells, const nbd::Matrices& d);

  struct Level {
    std::vector<int> IND;
    std::vector<int> CHILD_IND;
    std::vector<int> NCHILD;
  };

  typedef std::vector<Level> ElimOrder;

  ElimOrder order(const nbd::Cells& cells);


};
