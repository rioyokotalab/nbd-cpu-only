
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

  int m = 2048;
  int dim = 3;
  int leaf = 128;

  Bodies b1(m);
  auto fun = l3d();

  initRandom(b1, m, dim, 0, 1., 0);

  Cells c1 = buildTree(b1, leaf, dim);

  Matrices d, bi;
  traverse(fun, c1, c1, dim, d, 0.8, 128);

  {
    Matrix a_ref(m, m, m), a_rebuilt(m, m, m);
    convertHmat2Dense(c1, c1, d, a_rebuilt, a_rebuilt.LDA);
    P2Pnear(fun, &c1[0], &c1[0], dim, a_ref);
    printf("%e\n", rel2err(&a_rebuilt[0], &a_ref[0], m, m, m, m));
  }

  printTree(&c1[0], dim);
  traverse_i(c1, c1, d, bi, 10);
  shared_epilogue(d);

   {
    Matrix a_ref(m, m, m), a_rebuilt(m, m, m);
    convertH2mat2Dense(c1, c1, bi, bi, d, a_rebuilt, a_rebuilt.LDA);
    P2Pnear(fun, &c1[0], &c1[0], dim, a_ref);
    printf("%e\n", rel2err(&a_rebuilt[0], &a_ref[0], m, m, m, m));

  }


  std::vector<double> x(m), b(m);
  vecRandom(&x[0], m, 1, 0, 1);

  h2mv_complete(c1, c1, bi, bi, d, &x[0], &b[0]);

  std::vector<double> b_ref(m);

  mvec_kernel(fun, &c1[0], &c1[0], dim, &x[0], &b_ref[0]);

  printf("%e\n", rel2err(&b[0], &b_ref[0], m, 1, m, m));

  return 0;
}
