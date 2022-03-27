
#include "build_tree.hxx"
#include "kernel.hxx"
#include "h2mv.hxx"
#include "basis.hxx"
#include "solver.hxx"
#include "dist.hxx"

#include <cstdio>
#include <cstdlib>
#include <random>

int main(int argc, char* argv[]) {

  using namespace nbd;

  int64_t dim = 2;
  int64_t m = 4096;
  int64_t leaf = 128;
  int64_t theta = 1;

  std::vector<double> my_min(dim + 1, 0.);
  std::vector<double> my_max(dim + 1, 1.);
  EvalFunc fun = dim == 2 ? l2d() : l3d();
  fun.singularity = m * 1.e3;

  Bodies body(m);
  randomBodies(body, m, &my_min[0], &my_max[0], dim, 1234);
  Cells cell;
  int64_t levels = buildTree(cell, body, leaf, &my_min[0], &my_max[0], dim);

  initComm(&argc, &argv);

  traverse(cell, levels, dim, theta);

  Basis basis;
  allocBasis(basis, levels);
  evaluateBaseAll(fun, &basis[0], cell, levels, body, 100, 2000, dim);

  std::vector<CSC> cscs(levels + 1);
  relationsNear(&cscs[0], cell);
  std::vector<Matrices> d(levels + 1);
  evaluateNear(&d[0], fun, cell, dim, &cscs[0], &basis[0], levels);

  std::srand(100);
  std::vector<double> R(1 << 16);
  for (int64_t i = 0; i < R.size(); i++)
    R[i] = -1. + 2. * ((double)std::rand() / RAND_MAX);

  const Cell* local = &cell[0];
  std::vector<SpDense> sp(levels + 1);
  for (int64_t i = 0; i <= levels; i++) {
    local = findLocalAtLevel(local, i);
    allocSpDense(sp[i], &cscs[0], i);
    factorSpDense(sp[i], local, d[i], 100, &R[0], R.size());
  }

  std::vector<MatVec> vx(levels + 1);
  allocMatVec(&vx[0], &basis[0], levels);

  Vectors X;
  loadX(X, local, levels);

  h2MatVecAll(&vx[0], fun, &cell[0], &basis[0], dim, X, levels);

  Vectors Bref(X.size());
  h2MatVecReference(Bref, fun, &cell[0], dim, levels);

  int64_t mpi_rank;
  commRank(&mpi_rank, NULL, NULL);
  double err;
  solveRelErr(&err, vx[levels].B, Bref, levels);
  printf("H2-vec vs direct m-vec %lld ERR: %e\n", mpi_rank, err);

  RHSS rhs(levels + 1);
  solveH2(&rhs[0], &vx[0], &sp[0], fun, &cell[0], &basis[0], dim, Bref, levels);

  solveRelErr(&err, rhs[levels].X, X, levels);
  printf("H2-solve %lld ERR: %e\n", mpi_rank, err);

  closeComm();
  return 0;
}
