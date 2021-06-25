
#include "build_tree.h"
#include "kernel.h"
#include "aca.h"
#include "h2mv.h"
#include "test_util.h"

#include <cstdio>
#include <random>
#include <chrono>


int main(int argc, char* argv[]) {

  using namespace nbd;

  int m = 512;
  int dim = 2;
  int leaf = 128;

  Bodies b1(m);
  auto fun = l2d();

  initRandom(b1, m, dim, 0, 1, 199);

  Cells c1 = buildTree(b1, leaf, dim);

  Matrices d, bi;
  traverse(fun, c1, c1, dim, d, 1.0, 32);

  Matrix a_ref(m, m, m), a_rebuilt(m, m, m);
  convertHmat2Dense(c1, c1, d, a_rebuilt, a_rebuilt.LDA);
  P2Pnear(l2d(), &c1[0], &c1[0], dim, a_ref);
  /*traverse_i(c1, c1, d, bi);
  shared_epilogue(d);

  printTree(&c1[0]);

  std::vector<double> x(m), b(m);
  vecRandom(&x[0], m, 1, 0, 1);

  h2mv_complete(c1, c1, bi, bi, d, &x[0], &b[0]);

  std::vector<double> b_ref(m);

  mvec_kernel(fun, &c1[0], &c1[0], dim, &x[0], &b_ref[0]);*/

  printf("%e\n", rel2err(&a_rebuilt[0], &a_ref[0], m, m, m, m));

  return 0;
}
