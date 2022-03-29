

#include "solver.hxx"
#include "dist.hxx"
#include "h2mv.hxx"
#include "minblas.h"

#include <random>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <chrono>
#include "mpi.h"
#include "omp.h"

using namespace nbd;

int main(int argc, char* argv[]) {

  initComm(&argc, &argv);

  int64_t Nbody = argc > 1 ? atol(argv[1]) : 8192;
  int64_t Ncrit = argc > 2 ? atol(argv[2]) : 256;
  int64_t theta = argc > 3 ? atol(argv[3]) : 1;
  int64_t dim = argc > 4 ? atol(argv[4]) : 3;
  double epi = 1.e-4;
  int64_t rank_max = 40;
  //omp_set_num_threads(4);
  
  EvalFunc ef = dim == 2 ? l2d() : l3d();
  ef.singularity = Nbody * 1.e3;

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
  MPI_Barrier(MPI_COMM_WORLD);
  auto start_time = std::chrono::system_clock::now();
  factorSpDense(sp, lcleaf, A, epi, rank_max, &R[0], R.size());
  MPI_Barrier(MPI_COMM_WORLD);
  auto stop_time = std::chrono::system_clock::now();
  double factor_time_process = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();

  double total_factor_time;
  MPI_Reduce(&factor_time_process, &total_factor_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  total_factor_time = total_factor_time / mpi_size; 

  Vectors X, Xref;
  loadX(X, lcleaf, levels);
  loadX(Xref, lcleaf, levels);

  RHSS rhs(levels + 1);

  MPI_Barrier(MPI_COMM_WORLD);
  auto start_solve = std::chrono::system_clock::now();
  solveSpDense(&rhs[0], sp, X);
  MPI_Barrier(MPI_COMM_WORLD);
  auto  stop_solve = std::chrono::system_clock::now();
  double solve_time_process = std::chrono::duration_cast<std::chrono::milliseconds>(stop_solve - start_solve).count();
  double total_solve_time = 0;
  MPI_Reduce(&solve_time_process, &total_solve_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  total_solve_time = total_solve_time / mpi_size;

  DistributeVectorsList(rhs[levels].X, levels);
  for (int64_t i = 0; i < X.size(); i++)
    zeroVector(X[i]);
  closeQuarter(X, rhs[levels].X, ef, lcleaf, dim, levels);

  int64_t mpi_rank;
  commRank(&mpi_rank, NULL, NULL);
  double err;
  solveRelErr(&err, X, Xref, levels);
  //printf("%lld ERR: %e\n", mpi_rank, err);

  int64_t* flops = getFLOPS();
  double gf = flops[0] * 1.e-9;
  //printf("%lld GFLOPS: %f\n", mpi_rank, gf);

  if (mpi_rank == 0) {
    std::cout << "LORASP: " << Nbody << "," << Ncrit << "," << theta << "," << dim
	    << "," << mpi_size << "," << total_factor_time << ","
	    << total_solve_time << "," << err << "," << gf << std::endl;
	    
  }
  closeComm();
  return 0;
}
