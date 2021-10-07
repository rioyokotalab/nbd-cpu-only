
#include "build_tree.h"
#include "kernel.h"
#include "h2mv.h"
#include "bsp_umv.h"
#include "test_util.h"
#include "timer.h"

#include <cstdio>
#include <random>
#include <chrono>


int main(int argc, char* argv[]) {

  using namespace nbd;

  int dim = 2;
  int m = atoi(argv[1]);
  int leaf = 128;
  double theta = 0.7;

  Bodies b1(m);
  initRandom(b1, m, dim, 0, 1., 0);

  Cells c1 = getLeaves(buildTree(b1, leaf, dim));
  getList(&c1[0], &c1[0], dim, theta, true);

  auto fun = dim == 2 ? l2d() : l3d();

  std::vector<double> x(m), b(m);
  vecRandom(&x[0], m, 1, 0, 1);

  closeQuarter(fun, c1, c1, dim, &x[0], &b[0]);

  start("construct");
  std::vector<Node> H;
  H.reserve(100);
  H.emplace_back(fun, dim, &c1[0], &c1[0]);
  stop("construct");

  start("factor");
  std::vector<Base> B;
  Matrix last;
  std::vector<int> ipiv;
  B.reserve(100);
  h2_factor(1.e-6, H, B, last, ipiv);
  stop("factor");

  start("solution");
  h2_solve(H.size(), &H[0], &B[0], last, ipiv.data(), &b[0]);
  stop("solution");

  printf("solution err %e\n", rel2err(&b[0], &x[0], m, 1, m, m));
  
  return 0;
}
