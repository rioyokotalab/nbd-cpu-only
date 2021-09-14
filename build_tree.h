
#pragma once
#include "nbd.h"

namespace nbd {

#define PART_EQ_SIZE

  Cells buildTree(Bodies& bodies, int ncrit, int dim);

  void getLevel(Cell* cell);

  void getList(Cell* Ci, Cell* Cj, int dim, real_t theta, bool symm, bool x_level);

  Matrices evaluate(EvalFunc ef, const Cells& icells, const Cells& jcells, int dim, int rank, bool eval_near);

  Matrices traverse(EvalFunc ef, Cells& icells, Cells& jcells, int dim, real_t theta, int rank, bool x_level = false, bool eval_near = false);

  Matrices sample_base_i(const Cells& icells, const Cells& jcells, Matrices& d, int p);

  Matrices sample_base_j(const Cells& icells, const Cells& jcells, Matrices& d, int p);

  void sample_base_recur(Cell* cell, Matrix* base);

  void shared_base_i(const Cells& icells, const Cells& jcells, Matrices& d, Matrices& base, bool symm);

  void shared_base_j(const Cells& icells, const Cells& jcells, Matrices& d, Matrices& base);

  void nest_base(Cell* icell, Matrix* base);

  Matrices traverse_i(Cells& icells, Cells& jcells, Matrices& d, int p);

  Matrices traverse_j(Cells& icells, Cells& jcells, Matrices& d, int p);

  void shared_epilogue(Matrices& d);

  Matrices traverse_oo(const Cells& icells, const Cells& jcells, const Matrices& ibase, const Matrices& jbase, Matrices& d);

  Cells getLeaves(const Cells& cells);

}

