
#pragma once

#include "nbd.h"

namespace nbd {

  void printVec(real_t* a, int n, int inc);

  void printMat(real_t* a, int m, int n, int lda);

  void initRandom(Bodies& b, int m, int dim, real_t min, real_t max, unsigned int seed = 0);

  void vecRandom(real_t* a, int n, int inc, real_t min, real_t max, unsigned int seed = 0);

  void printTree(const Cell* cell, int level = 0, int offset_c = 0, int offset_b = 0);

  void convertHmat2Dense(const Cells& icells, const Cells& jcells, const Matrices& d, real_t* a, int lda);

  real_t rel2err(const real_t* a, const real_t* ref, int m, int n, int lda, int ldref);


}
