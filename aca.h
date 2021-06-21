
#pragma once
#include "nbd.h"

namespace nbd {

#define ACA_USE_NORM
  constexpr real_t ACA_EPI = 1.e-5;

  void raca(eval_func_t r2f, const Cell* ci, const Cell* cj, int dim, int64_t max_iters, real_t* u, int64_t ldu, real_t* v, int64_t ldv, int* info = nullptr);

  void raca(int64_t m, int64_t n, int64_t max_iters, const real_t* a, int64_t lda, real_t* u, int64_t ldu, real_t* v, int64_t ldv, int* info = nullptr);

}