#pragma once

#include <vector>

namespace qs {

  struct Matrix {
    std::vector<double> A;
    int M;
    int N;
    int LDA;
    int LMO;
    int LNO;
  };

  typedef std::vector<Matrix> Matrices;

  struct H2Matrix {
    int N;
    Matrices D;
  };

  struct Level {
    std::vector<int> IND;
    std::vector<int> CHILD_IND;
    std::vector<int> NCHILD;
  };

  typedef std::vector<Level> ElimOrder;

};