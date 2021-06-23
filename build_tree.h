
#pragma once
#include "nbd.h"

namespace nbd {

#define PART_EQ_SIZE

  Cells buildTree(Bodies& bodies, int ncrit, int dim);

  void getList(Cell * Ci, Cell * Cj, int dim, real_t theta);

  void evaluate(eval_func_t r2f, Cells& cells, int dim, Matrices& d, Matrices& lr, int rank);

  void traverse(eval_func_t r2f, Cells& icells, Cells& jcells, int dim, Matrices& d, Matrices& lr, real_t theta, int rank);

  void getBoundBox(int m, Cell * cell, Bodies& box, int dim, real_t inner_s);

  void printTree(const Cell * cell, int level = 0, int offset = 0);


}

