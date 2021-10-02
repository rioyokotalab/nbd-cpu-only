
#include "build_tree.h"
#include "kernel.h"
#include "h2mv.h"
#include "blr2.h"
#include "test_util.h"
#include "timer.h"
#include "omp.h"

#include <cstdio>
#include <random>
#include <chrono>


int main(int argc, char* argv[]) {

  using namespace nbd;

  int dim = 2;
  int m = 8192;
  int leaf = 128;
  int p = 10;
  double theta = 0.7;

  Bodies b1(m);
  initRandom(b1, m, dim, 0, 1., 0);

  Cells c1 = getLeaves(buildTree(b1, leaf, dim));
  getList(&c1[0], &c1[0], dim, theta, true);

  omp_set_num_threads(4);
  int num_threads = omp_get_max_threads();
  printf("threads: %d\n", num_threads);

  auto fun = dim == 2 ? l2d() : l3d();
  Node n = node(fun, dim, &c1[0], &c1[0]);

  start("build H2");
  Base bi = base_i(1.e-13, p, n);
  split_A(n, bi, bi);
  stop("build H2");

  start("factor H2");
  factor_A(n);
  stop("factor H2");

  std::vector<double> x(m), b(m);
  vecRandom(&x[0], m, 1, 0, 1);

  closeQuarter(fun, c1, c1, dim, &x[0], &b[0]);

  start("solution");
  auto xi = base_fw(bi, &b[0]);
  A_fw(n, xi);

  Matrix last = merge_D(n);
  solve_D(last, xi[n.M]);

  A_bk(n, xi);
  base_bk(bi, &b[0]);
  stop("solution");

  printf("solution err %e\n", rel2err(&b[0], &x[0], m, 1, m, m));
  
  return 0;
}
