
#pragma once

#include "linalg.hxx"

namespace nbd {

  typedef void (*eval_func_t) (double&, double, double);

  struct Cell;
  struct Body;

  struct EvalFunc {
    eval_func_t r2f;
    double singularity;
    double alpha;
  };

  EvalFunc r2();

  EvalFunc l2d();

  EvalFunc l3d();

  void eval(EvalFunc ef, const Body* bi, const Body* bj, int64_t dim, double* out);

  void P2P(EvalFunc ef, const Cell* ci, const Cell* cj, int64_t dim, const Vector& X, Vector& B);

  void P2Pmat(EvalFunc ef, const Cell* ci, const Cell* cj, int64_t dim, Matrix& a);

  void M2L(EvalFunc ef, const Cell* ci, const Cell* cj, int64_t dim, const double m[], double l[]);

  void M2Lc(EvalFunc ef, const Cell* ci, const Cell* cj, int64_t dim, const Vector& M, Vector& L);

  void M2Lmat_bodies(EvalFunc ef, int64_t m, int64_t n, const int64_t mi[], const int64_t mj[], const Body* bi, const Body* bj, int64_t dim, Matrix& a);

  void M2Lmat(EvalFunc ef, const Cell* ci, const Cell* cj, int64_t dim, Matrix& a);

  void P2Mmat(EvalFunc ef, Cell* ci, const Body rm[], int64_t n, int64_t dim, Matrix& u, double epi);

}