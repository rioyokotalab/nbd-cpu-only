#pragma once

#include "../nbd.h"

namespace irs {

  struct BlockMatrix {
    std::vector<double*> A;
    std::vector<int> N;
  };

  void matcpy(const nbd::Matrix& m1, int m, int n, double* a);

  BlockMatrix build(nbd::eval_func_t eval, int dim, const nbd::Cells& cells, double theta);

  void dgetf2np(int m, int n, double* a, int lda);

  void dtrsml(int m, int n, const double* a, double* b);

  void dtrsmc(int m, int n, const double* a, double* b);

  void dschur(int m, int n, int k, const double* a, const double* b, double*& c);

  void elim(BlockMatrix& d);

  void dtrsvf(int n, const double* a, double* x);

  void dtrsvb(int n, const double* a, double* x);

  void dschurv(int m, int n, const double* a, const double* x, double* y);

  void solve(const BlockMatrix& d, double* x);

  void clear(BlockMatrix& d);

};
