
#include "jacobi.h"

#include "cblas.h"
#include "h2mv.h"

#include <cstddef>
#include <cstring>
#include <algorithm>

using namespace nbd;

int nbd::h2solve(int max_iters, double epi, EvalFunc ef, const Cells& cells, int dim, const Matrices& base, const Matrices& d, double* x) {

  int N = cells[0].NBODY;
  double singularity = ef.singularity;
  double ins = 1. / singularity;
  ef.singularity = 0.;
  std::vector<double> x0(N), work(N);

  cblas_dcopy(N, x, 1, x0.data(), 1);
  cblas_dscal(N, ins, x0.data(), 1);

  double nrm = cblas_dnrm2(N, x0.data(), 1);
  double nrm_diff = 1.;

  int iters = 0;
  while (nrm_diff > epi && iters < max_iters) {
    h2mv_complete(ef, cells, cells, dim, base, base, d, x0.data(), work.data());
    cblas_dcopy(N, x, 1, x0.data(), 1);
    cblas_daxpy(N, -1., work.data(), 1, x0.data(), 1);
    cblas_dscal(N, ins, x0.data(), 1);
    
    double nrm_i = cblas_dnrm2(N, x0.data(), 1);
    iters += 1;

    nrm_diff = std::abs(nrm_i - nrm) / nrm;
    nrm = nrm_i;
  }

  cblas_dcopy(N, x0.data(), 1, x, 1);

  return iters;
}

