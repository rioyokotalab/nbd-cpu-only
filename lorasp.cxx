

#include "solver.hxx"
#include "dist.hxx"

#include <random>
#include <cstdio>
#include <cmath>
#include <iostream>

using namespace nbd;

int main(int argc, char* argv[]) {

  initComm(&argc, &argv);

  int64_t Nbody = argc > 1 ? atol(argv[1]) : 8192;
  double theta = argc > 2 ? atof(argv[2]) : 1;
  int64_t leaf_size = 256;
  int64_t dim = 3;

  double fac_epi = 1.e-5;
  int64_t fac_rank_max = 50;
  double lr_epi = 1.e-11;
  int64_t lr_rank_max = 100;
  int64_t sp_pts = 4000;

  int64_t mpi_rank;
  int64_t mpi_size;
  int64_t levels = (int64_t)std::log2(Nbody / leaf_size);
  int64_t Nleaf = (int64_t)1 << levels;

  commRank(&mpi_rank, &mpi_size, NULL);
  
  EvalFunc ef = l3d();//yukawa3d();
  ef.singularity = 1.e-3 / Nbody;

  std::srand(100);
  std::vector<double> R(1 << 18);
  for (int64_t i = 0; i < R.size(); i++)
    R[i] = -1. + 2. * ((double)std::rand() / RAND_MAX);
  
  Bodies body(Nbody);
  std::vector<int64_t> buckets(Nleaf);
  //readPartitionedBodies(DATA, body.data(), Nbody, buckets.data(), dim);
  randomSurfaceBodies(body.data(), Nbody, dim, 1234);
  //randomUniformBodies(body.data(), Nbody, 0., 1., dim, 1234);
  randomNeutralCharge(body.data(), Nbody, 1., 0);

  Cells cell;
  //buildTreeBuckets(cell, body.data(), Nbody, buckets.data(), levels, dim);
  buildTree(cell, body, levels, dim);
  traverse(cell, levels, dim, theta);
  const Cell* lcleaf = &cell[0];
  lcleaf = findLocalAtLevel(lcleaf, levels);

  std::vector<CSC> rels(levels + 1);
  relationsNear(&rels[0], cell);

  Matrices A(rels[levels].NNZ_NEAR);
  evaluateLeafNear(A, ef, &cell[0], dim, rels[levels]);

  SpDense sp;
  allocSpDense(sp, &rels[0], levels);

  double construct_time, construct_comm_time;
  startTimer(&construct_time, &construct_comm_time);
  evaluateBaseAll(ef, &sp.Basis[0], cell, levels, body, lr_epi, lr_rank_max, sp_pts, &R[0], R.size(), dim);
  for (int64_t i = 0; i <= levels; i++)
    evaluateFar(sp.D[i].S, ef, &cell[0], dim, rels[i], i);
  stopTimer(&construct_time, &construct_comm_time);

  double factor_time, factor_comm_time;
  startTimer(&factor_time, &factor_comm_time);
  factorSpDense(sp, lcleaf, A, fac_epi, fac_rank_max, &R[0], R.size());
  stopTimer(&factor_time, &factor_comm_time);

  Vectors X, Xref;
  loadX(X, lcleaf, levels);
  loadX(Xref, lcleaf, levels);

  Vectors B(X.size());
  h2MatVecReference(B, ef, &cell[0], dim, levels);

  RHSS rhs(levels + 1);

  double solve_time, solve_comm_time;
  startTimer(&solve_time, &solve_comm_time);
  solveSpDense(&rhs[0], sp, B);
  stopTimer(&solve_time, &solve_comm_time);

  DistributeVectorsList(rhs[levels].X, levels);

  double err;
  solveRelErr(&err, rhs[levels].X, Xref, levels);

  if (mpi_rank == 0) {
    std::cout << "LORASP: " << Nbody << "," << (int64_t)(Nbody / Nleaf) << "," << theta << "," << dim << "," << mpi_size << std::endl;
    std::cout << "Construct: " << construct_time << " COMM: " << construct_comm_time << std::endl;
    std::cout << "Factorize: " << factor_time << " COMM: " << factor_comm_time << std::endl;
    std::cout << "Solution: " << solve_time << " COMM: " << solve_comm_time << std::endl;
    std::cout << "Err: " << err << std::endl;
  }
  closeComm();
  return 0;
}
