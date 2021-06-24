
#pragma once
#include "nbd.h"

namespace nbd {

#define PART_EQ_SIZE

  Cells buildTree(Bodies& bodies, int ncrit, int dim);

  void getList(Cell * Ci, Cell * Cj, int dim, real_t theta, bool write_j);

  void evaluate(eval_func_t r2f, Cells& cells, const Cell* jcell_start, int dim, Matrices& d, int rank);

  void traverse(eval_func_t r2f, Cells& icells, Cells& jcells, int dim, Matrices& d, real_t theta, int rank);

  void sample_base_i(Cell* icell, Matrices& d, int ld, Matrix* base, int rank_p, const Cell* icell_start, const Cell* jcell_start);

  void sample_base_j(Cell* icell, Matrices& d, int ld, Matrix* base, int rank_p, const Cell* icell_start, const Cell* jcell_start);

  void shared_base_i(Cells& icells, Cells& jcells, Matrices& d, int ld, Matrices& base);

  void shared_base_j(Cells& icells, Cells& jcells, Matrices& d, int ld, Matrices& base, bool orth);

  void nest_base(Cell* icell, Matrix* base);

  void traverse_i(Cells& icells, Cells& jcells, Matrices& d, Matrices& base);

  void traverse_j(Cells& icells, Cells& jcells, Matrices& d, Matrices& base);

  void shared_epilogue(Matrices& d);

  void getBoundBox(int m, Cell * cell, Bodies& box, int dim, real_t inner_s);

  void printTree(const Cell * cell, int level = 0, int offset_c = 0, int offset_b = 0);


}

