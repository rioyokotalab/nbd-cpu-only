

#include "solver.h"
#include "dist.h"

#include <random>
#include <cstdio>
#include <cmath>
#include <iostream>

int main(int argc, char* argv[]) {

  initComm(&argc, &argv);

  int64_t Nbody = argc > 1 ? atol(argv[1]) : 8192;
  double theta = argc > 2 ? atof(argv[2]) : 1;
  int64_t leaf_size = 256;

  double epi = 1.e-10;
  int64_t rank_max = 100;
  int64_t sp_pts = 2000;

  int64_t levels = (int64_t)std::log2(Nbody / leaf_size);
  int64_t Nleaf = (int64_t)1 << levels;
  
  KerFunc_t ef = laplace3d;
  set_kernel_constants(1.e-3 / Nbody, 1.);

  cRandom((int64_t)1.e6, -1, 1, 100);
  
  std::vector<Body> body(Nbody);
  std::vector<int64_t> buckets(Nleaf);
  mesh_unit_sphere(body.data(), Nbody);
  //uniform_unit_cube(body.data(), Nbody, 3, 1234);
  body_neutral_charge(body.data(), Nbody, 1., 0);

  std::vector<Cell> cell(Nleaf + Nleaf - 1);
  CSC cellNear, cellFar;
  buildTree(cell.data(), body.data(), Nbody, levels);
  traverse('N', &cellNear, cell.size(), cell.data(), theta);
  traverse('F', &cellFar, cell.size(), cell.data(), theta);

  traverse_dist(&cellFar, &cellNear, levels);

  SpDense sp;
  allocSpDense(sp, &cellFar, &cellNear, levels);

  double construct_time, construct_comm_time;
  startTimer(&construct_time, &construct_comm_time);
  evaluateBaseAll(ef, &sp.Basis[0], cell.data(), &cellNear, levels, body.data(), Nbody, epi, rank_max, sp_pts);
  stopTimer(&construct_time, &construct_comm_time);

  evaluate('N', sp.D[levels].A.data(), ef, &cell[0], &cellNear, body.data(), &sp.RelsNear[levels], levels);
  for (int64_t i = 0; i <= levels; i++)
    evaluate('F', sp.D[i].S.data(), ef, &cell[0], &cellFar, body.data(), &sp.RelsFar[i], i);

  double factor_time, factor_comm_time;
  startTimer(&factor_time, &factor_comm_time);
  factorSpDense(sp);
  stopTimer(&factor_time, &factor_comm_time);
  cRandom(0, 0, 0, 0);

  int64_t xlen = (int64_t)1 << levels;
  contentLength(&xlen, levels);
  std::vector<Matrix> X(xlen), Xref(xlen);
  loadX(X.data(), cell.data(), body.data(), levels);
  loadX(Xref.data(), cell.data(), body.data(), levels);

  std::vector<Matrix> B(xlen);
  h2MatVecReference(B.data(), ef, &cell[0], body.data(), levels);

  std::vector<RightHandSides> rhs(levels + 1);

  double solve_time, solve_comm_time;
  startTimer(&solve_time, &solve_comm_time);
  solveSpDense(&rhs[0], sp, B.data());
  stopTimer(&solve_time, &solve_comm_time);

  DistributeMatricesList(rhs[levels].X.data(), levels);

  double err;
  solveRelErr(&err, rhs[levels].X.data(), Xref.data(), levels);
  int64_t mpi_rank;
  int64_t mpi_levels;
  int64_t dim = 3;
  commRank(&mpi_rank, &mpi_levels);

  int64_t mem_basis, mem_A, mem_X;
  basis_mem(&mem_basis, &sp.Basis[0], levels);
  node_mem(&mem_A, &sp.D[0], levels);
  RightHandSides_mem(&mem_X, &rhs[0], levels);

  if (mpi_rank == 0) {
    std::cout << "LORASP: " << Nbody << "," << (int64_t)(Nbody / Nleaf) << "," << theta << "," << dim << "," << (int64_t)(1 << mpi_levels) << std::endl;
    std::cout << "Construct: " << construct_time << " COMM: " << construct_comm_time << std::endl;
    std::cout << "Factorize: " << factor_time << " COMM: " << factor_comm_time << std::endl;
    std::cout << "Solution: " << solve_time << " COMM: " << solve_comm_time << std::endl;
    std::cout << "Basis Memory: " << (double)mem_basis * 1.e-9 << " GiB." << std::endl;
    std::cout << "Matrix Memory: " << (double)mem_A * 1.e-9 << " GiB." << std::endl;
    std::cout << "Vector Memory: " << (double)mem_X * 1.e-9 << " GiB." << std::endl;
    std::cout << "Err: " << err << std::endl;
  }

  deallocSpDense(&sp);
  deallocRightHandSides(&rhs[0], levels);
  for (int64_t i = 0; i < xlen; i++) {
    matrixDestroy(&X[i]);
    matrixDestroy(&Xref[i]);
    matrixDestroy(&B[i]);
  }
  free(cellFar.COL_INDEX);
  free(cellNear.COL_INDEX);
  
  closeComm();
  return 0;
}
