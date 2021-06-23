
#pragma once
#include "nbd.h"

namespace nbd {

  eval_func_t r2();

  eval_func_t l3d();

  void eval(eval_func_t r2f, const Body* bi, const Body* bj, int dim, real_t* out);

  void dense_kernel(eval_func_t r2f, const Cell* ci, const Cell* cj, int dim, real_t* a, int lda);

  void mvec_kernel(eval_func_t r2f, const Cell* ci, const Cell* cj, int dim, const real_t* x_vec, real_t* b_vec);

  void P2M_L2P(eval_func_t r2f, int p, Cell* c, int dim);

  void M2L(eval_func_t r2f, const Cell* ci, const Cell* cj, int dim, Matrix& s);

}