
#include "build_tree.h"
#include "kernel.h"
#include "lra.h"
#include "h2mv.h"
#include "test_util.h"
#include "jacobi.h"
#include "timer.h"

#include <cstdio>
#include <random>
#include <chrono>


int main(int argc, char* argv[]) {

  using namespace nbd;

  int dim = 3;
  int m = 4000;
  int leaf = 400;
  int rank = 80;
  int p = 20;
  double theta = 1.01;

  start("bodies");
  Bodies b1(m);
  initRandom(b1, m, dim, 0., 1., 0);
  stop("bodies");

  start("build tree");
  Cells c1 = buildTree(b1, leaf, dim);
  stop("build tree");

  start("build H");
  auto fun = dim == 2 ? l2d() : l3d();
  Matrices d, bi;
  d = traverse(fun, c1, c1, dim, theta, rank);
  stop("build H");

  start("build H2");
  bi = traverse_i(c1, c1, d, p);
  shared_epilogue(d);
  stop("build H2");

  std::vector<double> x(m), b(m);
  vecRandom(&x[0], m, 1, 0, 1);

  start("Mvec");
  h2mv_complete(fun, c1, c1, dim, bi, bi, d, &x[0], &b[0]);
  stop("Mvec");

  start("Solv");
  int iters = h2solve(100, 1.e-14, fun, c1, dim, bi, d, &b[0]);
  stop("Solv");

  printf("solve using jacobi-iterative err = %e, iters = %d\n", rel2err(&b[0], &x[0], m, 1, m, m), iters);

  return 0;
}
