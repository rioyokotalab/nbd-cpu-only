

#include "solver.hxx"
#include "dist.hxx"
#include "h2mv.hxx"
#include "minblas.h"

#include <random>
#include <cstdio>
#include <cmath>

using namespace nbd;

int main(int argc, char* argv[]) {

  initComm(&argc, &argv);

  int64_t Nbody = 40000;
  int64_t Ncrit = 100;
  int64_t theta = 1;
  int64_t dim = 2;
  EvalFunc ef = dim == 2 ? l2d() : l3d();

  std::srand(100);
  std::vector<double> R(1 << 16);
  for (int64_t i = 0; i < R.size(); i++)
    R[i] = -1. + 2. * ((double)std::rand() / RAND_MAX);

  std::vector<double> my_min(dim + 1, 0.);
  std::vector<double> my_max(dim + 1, 1.);

  Bodies body(Nbody);
  randomBodies(body, Nbody, &my_min[0], &my_max[0], dim, 1234);
  Cells cell;
  int64_t levels = buildTree(cell, body, Ncrit, &my_min[0], &my_max[0], dim);
  traverse(cell, levels, dim, theta);
  const Cell* lcleaf = &cell[0];
  lcleaf = findLocalAtLevel(lcleaf, levels);

  std::vector<CSC> rels(levels + 1);
  relationsNear(&rels[0], cell);

  Matrices A(rels[levels].NNZ);
  evaluateLeafNear(A, ef, &cell[0], dim, rels[levels]);

  SpDense sp;
  allocSpDense(sp, &rels[0], levels);
  factorSpDense(sp, lcleaf, A, 1.e-7, &R[0], R.size());

  Vectors X, Xref;
  loadX(X, lcleaf, levels);
  loadX(Xref, lcleaf, levels);

  RHSS rhs(levels + 1);
  solveSpDense(&rhs[0], sp, X);

  DistributeVectorsList(rhs[levels].X, levels);
  for (int64_t i = 0; i < X.size(); i++)
    zeroVector(X[i]);
  closeQuarter(X, rhs[levels].X, ef, lcleaf, dim, levels);

  int64_t mpi_rank;
  commRank(&mpi_rank, NULL, NULL);
  double err;
  solveRelErr(&err, X, Xref, levels);
  printf("%lld ERR: %e\n", mpi_rank, err);

  int64_t* flops = getFLOPS();
  double gf = flops[0] * 1.e-9;
  printf("%lld GFLOPS: %f\n", mpi_rank, gf);
  closeComm();
  return 0;
}
