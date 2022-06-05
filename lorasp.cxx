

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

  double epi = 1.e-10;
  int64_t rank_max = 100;
  int64_t sp_pts = 2000;

  int64_t mpi_rank;
  int64_t mpi_size;
  int64_t levels = (int64_t)std::log2(Nbody / leaf_size);
  int64_t Nleaf = (int64_t)1 << levels;

  commRank(&mpi_rank, &mpi_size, NULL);
  
  KerFunc_t ef = laplace3d;
  set_kernel_constants(1.e-3 / Nbody, 1.);

  cRandom((int64_t)1.e5, -1, 1, 100);
  
  std::vector<Body> body(Nbody);
  std::vector<int64_t> buckets(Nleaf);
  //readPartitionedBodies(DATA, body.data(), Nbody, buckets.data(), dim);
  mesh_unit_sphere(body.data(), Nbody);
  //uniform_unit_cube(body.data(), Nbody, 3, 1234);
  body_neutral_charge(body.data(), Nbody, 1., 0);

  Cells cell;
  //buildTreeBuckets(cell, body.data(), Nbody, buckets.data(), levels, dim);
  buildTree(cell, body.data(), Nbody, levels);
  traverse(cell, levels, theta);
  const Cell* lcleaf = &cell[0];
  lcleaf = findLocalAtLevel(lcleaf, levels);

  std::vector<CSC> rels(levels + 1);
  relationsNear(&rels[0], cell);

  Matrices A(rels[levels].NNZ_NEAR);
  evaluateLeafNear(A, ef, &cell[0], rels[levels]);

  SpDense sp;
  allocSpDense(sp, &rels[0], levels);

  double construct_time, construct_comm_time;
  startTimer(&construct_time, &construct_comm_time);
  evaluateBaseAll(ef, &sp.Basis[0], cell, levels, body.data(), Nbody, epi, rank_max, sp_pts);
  for (int64_t i = 0; i <= levels; i++)
    evaluateFar(sp.D[i].S, ef, &cell[0], rels[i], i);
  stopTimer(&construct_time, &construct_comm_time);

  double factor_time, factor_comm_time;
  startTimer(&factor_time, &factor_comm_time);
  factorSpDense(sp, lcleaf, A, epi, rank_max);
  stopTimer(&factor_time, &factor_comm_time);
  cRandom(0, 0, 0, 0);

  Vectors X, Xref;
  loadX(X, lcleaf, levels);
  loadX(Xref, lcleaf, levels);

  Vectors B(X.size());
  h2MatVecReference(B, ef, &cell[0], levels);

  RHSS rhs(levels + 1);

  double solve_time, solve_comm_time;
  startTimer(&solve_time, &solve_comm_time);
  solveSpDense(&rhs[0], sp, B);
  stopTimer(&solve_time, &solve_comm_time);

  DistributeVectorsList(rhs[levels].X, levels);

  double err;
  solveRelErr(&err, rhs[levels].X, Xref, levels);
  int64_t dim = 3;

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
