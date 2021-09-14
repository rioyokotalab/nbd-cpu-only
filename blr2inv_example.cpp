
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
  d = traverse(fun, c1, c1, dim, theta, rank, false, true);
  bi = traverse_i(c1, c1, d, p);
  shared_epilogue(d);

  std::vector<double> x(m), b(m);
  vecRandom(&x[0], m, 1, 0, 1);

  mvec_kernel(fun, &c1[0], &c1[0], dim, 1., &x[0], 1, 0., &b[0], 1);

  Matrix dA;
  near(c1, c1, d, dA);
  near_solve(dA, &b[0], &b[0]);

  std::vector<double> m0(m), l0(m);
  upwardPassOne(bi, &b[0], &m0[0]);

  Matrices d_oo = traverse_oo(c1, c1, bi, bi, d);

  Matrix& fi = d[0];
  dgetf2np(fi.M, fi.N, fi, fi.LDA);

  std::vector<int> ma(bi.size());
  for (int i = 0; i < bi.size(); i++)
    ma[i] = bi[i].N;

  std::fill(l0.begin(), l0.end(), 0.);
  spmv(m / leaf, m / leaf, &d_oo[1 + c1.size()], c1.size(), &ma[1], &ma[1], &m0[0], &l0[0]);
  near_solve(fi, &l0[0], &l0[0]);
  
  std::fill(m0.begin(), m0.end(), 0.);
  spmv(m / leaf, m / leaf, &d[1 + c1.size()], c1.size(), &ma[1], &ma[1], &l0[0], &m0[0]);

  std::fill(l0.begin(), l0.end(), 0.);
  downwardPassOne(bi, &m0[0], &l0[0]);

  near_solve(dA, &l0[0], &l0[0]);

  for (int i = 0; i < m; i++)
    b[i] += -l0[i];

  printf("H2-vec vs direct m-vec err %e\n", rel2err(&b[0], &x[0], m, 1, m, m));

  printTree(&c1[0], dim);
  
  return 0;
}
