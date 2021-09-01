
#pragma once

#include "nbd.h"

namespace nbd {
  
  int h2solve(int max_iters, double epi, EvalFunc ef, const Cells& cells, int dim, const Matrices& base, const Matrices& d, double* x);

}
