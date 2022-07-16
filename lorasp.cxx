
#include "nbd.h"
#include "profile.h"

#include <cstdio>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  double prog_time = MPI_Wtime();

  int64_t Nbody = argc > 1 ? atol(argv[1]) : 8192;
  double theta = argc > 2 ? atof(argv[2]) : 1;
  int64_t leaf_size = 256;

  double epi = 1.e-10;
  int64_t rank_max = 100;
  int64_t sp_pts = 2000;

  int64_t levels = (int64_t)std::log2(Nbody / leaf_size);
  int64_t Nleaf = (int64_t)1 << levels;
  
  auto ef = laplace3d;
  set_kernel_constants(1.e-3 / Nbody, 1.);
  
  std::vector<Body> body(Nbody);
  std::vector<int64_t> buckets(Nleaf);
  mesh_unit_sphere(body.data(), Nbody);
  //uniform_unit_cube(body.data(), Nbody, 3, 1234);
  body_neutral_charge(body.data(), Nbody, 1., 0);

  int64_t ncells = Nleaf + Nleaf - 1;
  std::vector<Cell> cell(ncells);
  CSC cellNear, cellFar;
  buildTree(&ncells, cell.data(), body.data(), Nbody, levels);
  traverse('N', &cellNear, cell.size(), cell.data(), theta);
  traverse('F', &cellFar, cell.size(), cell.data(), theta);

  std::vector<CellComm> cell_comm(levels + 1);
  buildComm(cell_comm.data(), ncells, cell.data(), &cellFar, &cellNear, levels);

  std::vector<CSC> rels_far(levels + 1);
  std::vector<CSC> rels_near(levels + 1);
  std::vector<Base> basis(levels + 1);
  relations(&rels_near[0], ncells, cell.data(), &cellNear, levels);
  relations(&rels_far[0], ncells, cell.data(), &cellFar, levels);

  double construct_time, construct_comm_time;
  startTimer(&construct_time, &construct_comm_time);
  buildBasis(ef, &basis[0], ncells, cell.data(), &rels_near[0], levels, cell_comm.data(), body.data(), Nbody, epi, rank_max, sp_pts);
  stopTimer(&construct_time, &construct_comm_time);
  
  std::vector<Node> nodes(levels + 1);
  allocNodes(&nodes[0], &basis[0], &rels_near[0], &rels_far[0], cell_comm.data(), levels);

  evalD(ef, nodes[levels].A, ncells, &cell[0], body.data(), &rels_near[levels], levels);
  for (int64_t i = 0; i <= levels; i++)
    evalS(ef, nodes[i].S, &basis[i], body.data(), &rels_far[i], &cell_comm[i]);

  double factor_time, factor_comm_time;
  startTimer(&factor_time, &factor_comm_time);
  factorA(&nodes[0], &basis[0], &rels_near[0], &rels_far[0], cell_comm.data(), levels);
  stopTimer(&factor_time, &factor_comm_time);

  int64_t body_local[2];
  local_bodies(body_local, ncells, cell.data(), levels);
  std::vector<double> X(body_local[1] - body_local[0]);
  loadX(X.data(), body_local, body.data());

  std::vector<double> B(body_local[1] - body_local[0]);
  mat_vec_reference(ef, body_local[0], body_local[1], &B[0], Nbody, body.data());

  std::vector<RightHandSides> rhs(levels + 1);
  allocRightHandSides(&rhs[0], &basis[0], levels);

  double solve_time, solve_comm_time;
  startTimer(&solve_time, &solve_comm_time);
  solveA(&rhs[0], &nodes[0], &basis[0], &rels_near[0], &B[0], cell_comm.data(), levels);
  stopTimer(&solve_time, &solve_comm_time);

  double err;
  solveRelErr(&err, B.data(), X.data(), X.size());

  int64_t dim = 3;

  int64_t mem_basis, mem_A, mem_X;
  basis_mem(&mem_basis, &basis[0], levels);
  node_mem(&mem_A, &nodes[0], levels);
  rightHandSides_mem(&mem_X, &rhs[0], levels);

  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  prog_time = MPI_Wtime() - prog_time;
  double cm_time;
  getCommTime(&cm_time);

  if (mpi_rank == 0) {
    std::cout << "LORASP: " << Nbody << "," << (int64_t)(Nbody / Nleaf) << "," << theta << "," << dim << "," << mpi_size << std::endl;
    std::cout << "Construct: " << construct_time << " s. COMM: " << construct_comm_time << " s." << std::endl;
    std::cout << "Factorize: " << factor_time << " s. COMM: " << factor_comm_time << " s." << std::endl;
    std::cout << "Solution: " << solve_time << " s. COMM: " << solve_comm_time << " s." << std::endl;
    std::cout << "Basis Memory: " << (double)mem_basis * 1.e-9 << " GiB." << std::endl;
    std::cout << "Matrix Memory: " << (double)mem_A * 1.e-9 << " GiB." << std::endl;
    std::cout << "Vector Memory: " << (double)mem_X * 1.e-9 << " GiB." << std::endl;
    std::cout << "Err: " << err << std::endl;
    std::cout << "Program: " << prog_time << " s. COMM: " << cm_time << " s." << std::endl;
  }

  for (int64_t i = 0; i <= levels; i++) {
    csc_free(&rels_far[i]);
    csc_free(&rels_near[i]);
    basis_free(&basis[i]);
    node_free(&nodes[i]);
    rightHandSides_free(&rhs[i]);
    cellComm_free(&cell_comm[i]);
  }
  free(cellFar.ColIndex);
  free(cellNear.ColIndex);
  
  MPI_Finalize();
  return 0;
}
