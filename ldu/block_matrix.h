#pragma once

#include "../nbd.h"

namespace nbd {

  Matrices near(EvalFunc ef, int dim, Cells& icells, Cells& jcells, double theta);

  void dgetf2np(int m, int n, double* a, int lda);

  void dtrsml(int m, int n, const double* a, double* b);

  void dtrsmc(int m, int n, const double* a, double* b);

  void dschur(int m, int n, int k, const double* a, const double* b, double*& c);

  void dtrsvf(int n, const double* a, double* x);

  void dtrsvb(int n, const double* a, double* x);

  void dschurv(int m, int n, const double* a, const double* x, double* y);


};
