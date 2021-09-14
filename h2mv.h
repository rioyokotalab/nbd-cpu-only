
#pragma once
#include "nbd.h"

namespace nbd {

  std::vector<int> dfs(const Cell* cell, bool pre_order, int i = 0);

  void upwardPass(const Cells& jcells, const Matrices& jbase, const real_t* x, real_t* m);

  void upwardPassOne(const Matrices& jbase, const real_t* x, real_t* m);

  void horizontalPass(const Cells& icells, const Cells& jcells, const Matrices& ibase, const Matrices& jbase, const Matrices& d, const real_t* m, real_t* l);

  void downwardPass(const Cells& icells, const Matrices& ibase, real_t* l, real_t* b);

  void downwardPassOne(const Matrices& ibase, real_t* l, real_t* b);

  void closeQuarter(EvalFunc ef, const Cells& icells, const Cells& jcells, int dim, const real_t* x, real_t* b);

  void h2mv_complete(EvalFunc ef, const Cells& icells, const Cells& jcells, int dim, const Matrices& ibase, const Matrices& jbase, const Matrices& d, const real_t* x, real_t* b);


}
