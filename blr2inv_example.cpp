
#include "build_tree.h"
#include "kernel.h"
#include "lra.h"
#include "h2mv.h"
#include "jacobi.h"
#include "test_util.h"
#include "h2inv.h"

#include <cstdio>
#include <random>
#include <chrono>


int main(int argc, char* argv[]) {

  using namespace nbd;

  int dim = 1;
  int m = 1024;
  int leaf = 128;
  int rank = 60;
  int p = 20;
  double theta = 0.9;

  Bodies b1(m);
  initRandom(b1, m, dim, 0, 1., 0);

  Cells c1 = getLeaves(buildTree(b1, leaf, dim));

  auto fun = l2d();
  Matrices d, bi;
  d = traverse(fun, c1, c1, dim, theta, rank);
  bi = traverse_i(c1, c1, d, p);
  shared_epilogue(d);

  std::vector<double> x(m), b(m);
  vecRandom(&x[0], m, 1, 0, 1);

  mvec_kernel(fun, &c1[0], &c1[0], dim, 1., &x[0], 1, 0., &b[0], 1);

  Matrix dA;
  near(c1, c1, d);

  //int iters = h2solve(100, 1.e-14, fun, c1, dim, bi, d, &b[0]);
  //printf("%d H2-vec vs direct m-vec err %e\n", iters, rel2err(&b[0], &x[0], m, 1, m, m));

  printTree(&c1[0], dim);
  
  return 0;
}
