
#include "build_tree.hxx"
#include "kernel.hxx"
#include "h2mv.hxx"
#include "basis.hxx"
#include "solver.hxx"
#include "dist.hxx"

#include "omp.h"
#include <iostream>
#include <cstdlib>
#include <random>

int main(int argc, char* argv[]) {

  using namespace nbd;

  int64_t m = argc > 1 ? atol(argv[1]) : 8192;
  int64_t leaf = argc > 2 ? atol(argv[2]) : 256;
  int64_t theta = argc > 3 ? atol(argv[3]) : 1;
  int64_t dim = argc > 4 ? atol(argv[4]) : 3;

  double lr_epi = 1.e-12;
  double ds_epi = 1.e-4;
  int64_t lr_max = 150;
  int64_t rank_max = 40;
  int64_t sp_pts = 3000;
  //omp_set_num_threads(4);

  EvalFunc fun = dim == 2 ? l2d() : l3d();
  fun.singularity = m * 1.e3;

  std::vector<double> my_min(dim + 1, 0.);
  std::vector<double> my_max(dim + 1, 1.);

  Bodies body(m);
  randomBodies(body, m, &my_min[0], &my_max[0], dim, 1234);
  Cells cell;
  int64_t levels = buildTree(cell, body, leaf, &my_min[0], &my_max[0], dim);

  initComm(&argc, &argv);
  traverse(cell, levels, dim, theta);

  std::vector<CSC> cscs(levels + 1);
  relationsNear(&cscs[0], cell);

  double ctime = 0.;
  startTimer(&ctime);
  Basis basis;
  allocBasis(basis, levels);
  evaluateBaseAll(fun, &basis[0], cell, levels, body, lr_epi, lr_max, sp_pts, dim);
  stopTimer(&ctime);

  std::vector<Matrices> d(levels + 1);
  evaluateNear(&d[0], fun, cell, dim, &cscs[0], &basis[0], levels);

  std::srand(100);
  std::vector<double> R(1 << 20);
  for (int64_t i = 0; i < R.size(); i++)
    R[i] = -1. + 2. * ((double)std::rand() / RAND_MAX);

  double ftime = 0.;
  startTimer(&ftime);
  const Cell* local = &cell[0];
  std::vector<SpDense> sp(levels + 1);
  for (int64_t i = 0; i <= levels; i++) {
    local = findLocalAtLevel(local, i);
    allocSpDense(sp[i], &cscs[0], i);
    factorSpDense(sp[i], local, d[i], ds_epi, rank_max, &R[0], R.size());
  }
  stopTimer(&ftime);

  std::vector<MatVec> vx(levels + 1);
  allocMatVec(&vx[0], &basis[0], levels);

  Vectors X;
  loadX(X, local, levels);

  double mvtime = 0.;
  startTimer(&mvtime);
  h2MatVecAll(&vx[0], fun, &cell[0], &basis[0], dim, X, levels);
  stopTimer(&mvtime);

  Vectors Bref(X.size());
  h2MatVecReference(Bref, fun, &cell[0], dim, levels);

  int64_t mpi_rank, mpi_size;
  commRank(&mpi_rank, &mpi_size, NULL);
  double err_mv;
  solveRelErr(&err_mv, vx[levels].B, Bref, levels);

  double stime = 0.;
  startTimer(&stime);
  RHSS rhs(levels + 1);
  solveH2(&rhs[0], &vx[0], &sp[0], fun, &cell[0], &basis[0], dim, Bref, levels);
  stopTimer(&stime);

  double err_s;
  solveRelErr(&err_s, rhs[levels].X, X, levels);

  if (mpi_rank == 0) {
    std::cout << "H2: " << m << "," << leaf << "," << theta << "," << dim
	    << "," << mpi_size << "," << ctime << "," << ftime << ","
	    << mvtime << "," << stime << "," << err_mv << "," << err_s << std::endl;
  }

  closeComm();
  return 0;
}
