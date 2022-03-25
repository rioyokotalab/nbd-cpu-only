
#pragma once
#include "basis.hxx"

namespace nbd {

  struct MatVec {
    Vectors X;
    Vectors M;
    Vectors L;
    Vectors B;
  };

  void interTrans(char updn, MatVec& vx, const Matrices& basis, int64_t level);

  void horizontalPass(Vectors& B, const Vectors& X, EvalFunc ef, const Cell* cell, int64_t dim, int64_t level);

  void closeQuarter(Vectors& B, const Vectors& X, EvalFunc ef, const Cell* cell, int64_t dim, int64_t level);

  void permuteAndMerge(char fwbk, Vectors& px, Vectors& nx, int64_t nlevel);

  void allocMatVec(MatVec vx[], const Base base[], int64_t levels);

  void resetMatVec(MatVec vx[], const Vectors& X, int64_t levels);

  void h2MatVecLR(MatVec vx[], EvalFunc ef, const Cell* root, const Base basis[], int64_t dim, const Vectors& X, int64_t levels);

  void h2MatVecAll(MatVec vx[], EvalFunc ef, const Cell* root, const Base basis[], int64_t dim, const Vectors& X, int64_t levels);

  void h2MatVecReference(Vectors& B, EvalFunc ef, const Cell* root, int64_t dim, int64_t levels);


}
