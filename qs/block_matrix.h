#pragma once

#include "../nbd.h"
#include "qs.h"

namespace qs {

  void matcpy(const nbd::Matrix& m1, Matrix& m2);

  Matrices convert(const nbd::Matrices& d);

  H2Matrix build(nbd::eval_func_t eval, int dim, const nbd::Cells& cells, const nbd::Matrices& d);

  ElimOrder order(const nbd::Cells& cells);

  void elim(const Level& lvl, H2Matrix& h2, Matrices& base);


};
