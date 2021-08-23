
#include "build_tree.h"
#include "kernel.h"
#include "aca.h"
#include "h2mv.h"
#include "test_util.h"
#include "qs/block_matrix.h"
#include "qs/factorize.h"
#include "qs/solve.h"

#include <cstdio>
#include <random>
#include <chrono>


int main(int argc, char* argv[]) {

  using namespace nbd;

  int dim = 1;
  int m = 128;
  int leaf = 64;
  int rank = 16;
  int p = 0;
  double theta = 1.01;

  Bodies b1(m);
  initRandom(b1, m, dim, 0, 1., 0);

  Cells c1 = buildTree(b1, leaf, dim);

  auto fun = dim == 2 ? l2d() : l3d();
  Matrices d, bi;
  traverse(fun, c1, c1, dim, d, theta, rank);

  {
    Matrix a_ref(m, m, m), a_rebuilt(m, m, m);
    convertHmat2Dense(fun, dim, c1, c1, d, a_rebuilt, a_rebuilt.LDA);
    P2Pnear(fun, &c1[0], &c1[0], dim, a_ref);
    printf("H-mat compress err %e\n", rel2err(&a_rebuilt[0], &a_ref[0], m, m, m, m));
  }

  traverse_i(c1, c1, d, bi, p);
  shared_epilogue(d);

  {
    Matrix a_ref(m, m, m), a_rebuilt(m, m, m);
    convertH2mat2Dense(fun, dim, c1, c1, bi, bi, d, a_rebuilt, a_rebuilt.LDA);
    P2Pnear(fun, &c1[0], &c1[0], dim, a_ref);
    printf("H2-mat compress err %e\n", rel2err(&a_rebuilt[0], &a_ref[0], m, m, m, m));
  }


  std::vector<double> x(m), b(m);
  vecRandom(&x[0], m, 1, 0, 1);

  h2mv_complete(fun, c1, c1, dim, bi, bi, d, &x[0], &b[0]);

  std::vector<double> b_ref(m);

  mvec_kernel(fun, &c1[0], &c1[0], dim, 1., &x[0], 1, 0., &b_ref[0], 1);

  printf("H2-vec vs direct m-vec err %e\n", rel2err(&b[0], &b_ref[0], m, 1, m, m));

  printTree(&c1[0], dim);

  using namespace qs;

  ElimOrder eo = order(c1, bi);

  H2Matrix h2 = build(fun, dim, c1, d);

  qs::Matrices base = convert(bi);
  
  for (int i = 0; i < eo.size(); i++) {
    if (i > 0)
      pnm(eo[i], h2);
    if (i < eo.size() - 1)
      elim(eo[i], h2, base);
  }

  dlu(h2.D[0]);

  for (int i = 0; i < eo.size() - 1; i++) {
    fwd_solution(eo[i], h2, base, &b[0]);
  }

  dgetrsnp(h2.D[0], &b[0]);

  for (int i = eo.size() - 2; i >= 0; i--) {
    bkwd_solution(eo[i], h2, base, &b[0]);
  }

  printf("H2-vec vs direct m-vec err %e\n", rel2err(&b[0], &x[0], m, 1, m, m));


  return 0;
}
