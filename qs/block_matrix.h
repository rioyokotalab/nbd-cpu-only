#pragma once

#include "../nbd.h"
#include "qs.h"

namespace qs {

  void matcpy(const nbd::Matrix& m1, Matrix& m2);

  Matrices convert(const nbd::Matrices& d);

  H2Matrix build(nbd::eval_func_t eval, int dim, const nbd::Cells& cells, const nbd::Matrices& d);

  ElimOrder order(const nbd::Cells& cells, const nbd::Matrices& base);

  void elim(const Level& lvl, H2Matrix& h2, Matrices& base);

  void merge(const Matrix* child, int ldc, int m, int n, Matrix& d);

  void pnm(const Level& lvl, H2Matrix& h2);

  void fwd_solution(const Level& lvl, const H2Matrix& h2, const Matrices& base, double* X);

  void bkwd_solution(const Level& lvl, const H2Matrix& h2, const Matrices& base, double* X);


};
