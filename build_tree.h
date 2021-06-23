
#pragma once
#include "nbd.h"

namespace nbd {

#define PART_EQ_SIZE

  Cells buildTree(Bodies& bodies, int ncrit, int dim);

  void getList(Cell * Ci, Cell * Cj, int dim, real_t theta);

  void getBoundBox(int m, Cell * cell, Bodies& outer, int dim, real_t inner_s = 1.03, real_t outer_s = 2.97);

}

