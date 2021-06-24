
#pragma once
#include "nbd.h"

namespace nbd {

  eval_func_t r2();

  eval_func_t l2d();

  eval_func_t l3d();

  void eval(eval_func_t r2f, const Body* bi, const Body* bj, int dim, real_t* out);

  void mvec_kernel(eval_func_t r2f, const Cell* ci, const Cell* cj, int dim, const real_t* x_vec, real_t* b_vec);

  void P2Pnear(eval_func_t r2f, const Cell* ci, const Cell* cj, int dim, Matrix& a);

  void P2Pfar(eval_func_t r2f, const Cell* ci, const Cell* cj, int dim, Matrix& a, int rank);

  void SampleP2Pi(Matrix& s, const Matrix& a);

  void SampleP2Pj(Matrix& s, const Matrix& a);

  void SampleParent(Matrix& s, int rank);

  void CopyParentBasis(Matrix& sc, const Matrix& sp);

  void BasisOrth(Matrix& s);

  void BasisInvLeft(const Matrix* s, int ls, Matrix& a);

  void BasisInvRight(const Matrix& s, Matrix& a);

  void MergeS(Matrix& a);

}