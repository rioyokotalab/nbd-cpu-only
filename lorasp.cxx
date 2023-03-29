
#include "nbd.hxx"
#include "profile.hxx"

#include "omp.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char* argv[]) {
  init_libs(&argc, &argv);

  double prog_time = MPI_Wtime();

  int64_t Nbody = argc > 1 ? atol(argv[1]) : 8192;
  double theta = argc > 2 ? atof(argv[2]) : 1e0;
  int64_t leaf_size = argc > 3 ? atol(argv[3]) : 256;
  double epi = argc > 4 ? atof(argv[4]) : 1e-10;
  int64_t rank_max = argc > 5 ? atol(argv[5]) : 100;
  int64_t sp_pts = argc > 6 ? atol(argv[6]) : 2000;
  const char* fname = argc > 7 ? argv[7] : NULL;

  leaf_size = Nbody < leaf_size ? Nbody : leaf_size;
  int64_t levels = (int64_t)log2((double)Nbody / leaf_size);
  int64_t Nleaf = (int64_t)1 << levels;
  int64_t ncells = Nleaf + Nleaf - 1;
  
  //double(*func)(double) = laplace3d_cpu();
  double(*func)(double) = yukawa3d_cpu();
  //double(*func)(double) = gauss_cpu();
  set_kernel_constants(1.e-9, 1);
  
  double* body = (double*)malloc(sizeof(double) * Nbody * 3);
  double* Xbody = (double*)malloc(sizeof(double) * Nbody);
  struct Cell* cell = (struct Cell*)malloc(sizeof(struct Cell) * ncells);
  struct CSC cellNear, cellFar;
  struct CSC* rels_far = (struct CSC*)malloc(sizeof(struct CSC) * (levels + 1));
  struct CSC* rels_near = (struct CSC*)malloc(sizeof(struct CSC) * (levels + 1));
  struct CellComm* cell_comm = (struct CellComm*)calloc(levels + 1, sizeof(struct CellComm));
  struct Base* basis = (struct Base*)calloc(levels + 1, sizeof(struct Base));
  struct Node* nodes = (struct Node*)malloc(sizeof(struct Node) * (levels + 1));
  struct RightHandSides* rhs = (struct RightHandSides*)malloc(sizeof(struct RightHandSides) * (levels + 1));

  if (fname == NULL) {
    mesh_unit_sphere(body, Nbody);
    //mesh_unit_cube(body, Nbody);
    //uniform_unit_cube(body, Nbody, 2, 1234);
    double c[3] = { 0, 0, 0 };
    double r[3] = { sqrt(1.e-3 * Nbody), sqrt(1.e-3 * Nbody), sqrt(1.e-3 * Nbody) };
    magnify_reloc(body, Nbody, c, c, r, 1.);
    buildTree(&ncells, cell, body, Nbody, levels);
  }
  else {
    int64_t* buckets = (int64_t*)malloc(sizeof(int64_t) * Nleaf);
    read_sorted_bodies(&Nbody, Nleaf, body, buckets, fname);
    //buildTreeBuckets(cell, body, buckets, levels);
    buildTree(&ncells, cell, body, Nbody, levels);
    free(buckets);
  }
  body_neutral_charge(Xbody, Nbody, 1., 999);

  traverse('N', &cellNear, ncells, cell, theta);
  traverse('F', &cellFar, ncells, cell, theta);

  buildComm(cell_comm, ncells, cell, &cellFar, &cellNear, levels);
  buildCommGPU(cell_comm, levels);
  relations(rels_near, ncells, cell, &cellNear, levels, cell_comm);
  relations(rels_far, ncells, cell, &cellFar, levels, cell_comm);

  double construct_time, construct_comm_time;
  startTimer(&construct_time, &construct_comm_time);
  buildBasis(func, basis, ncells, cell, &cellNear, levels, cell_comm, body, Nbody, epi, rank_max, sp_pts, 4);
  stopTimer(&construct_time, &construct_comm_time);

  double* Workspace = NULL;
  int64_t Lwork = 0;
  allocNodes(nodes, &Workspace, &Lwork, basis, rels_near, rels_far, cell_comm, levels);

  evalD(func, nodes[levels].A, ncells, cell, body, &cellNear, levels);
  for (int64_t i = 0; i <= levels; i++)
    evalS(func, nodes[i].S, &basis[i], &rels_far[i], &cell_comm[i]);

  int64_t lenX = rels_near[levels].N * basis[levels].dimN;
  double* X1 = (double*)calloc(lenX, sizeof(double));
  double* X2 = (double*)calloc(lenX, sizeof(double));

  loadX(X1, basis[levels].dimN, Xbody, ncells, cell, levels);
  allocRightHandSidesMV(rhs, basis, cell_comm, levels);
  matVecA(rhs, nodes, basis, rels_near, X1, cell_comm, levels);

  double cerr = 0.;
  if (Nbody < 20000) {
    int64_t body_local[2];
    local_bodies(body_local, ncells, cell, levels);
    std::vector<double> X3(lenX);
    mat_vec_reference(func, body_local[0], body_local[1], &X3[0], Nbody, body, Xbody);

    int64_t ibegin = 0, iend = ncells;
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    get_level(&ibegin, &iend, cell, levels, mpi_rank);

    for (int64_t i = 0; i < (iend - ibegin); i++) {
      int64_t b0 = cell[i + ibegin].Body[0];
      int64_t b1 = cell[i + ibegin].Body[1];
      const double* x3 = &X3[b0 - body_local[0]];
      double* x2 = &X2[i * basis[levels].dimN];
      for (int64_t j = 0; j < (b1 - b0); j++)
        x2[j] = x3[j];
    }
    solveRelErr(&cerr, X1, X2, lenX);
    std::iter_swap(&X1, &X2);
  }
  
  factorA_mov_mem('S', nodes, basis, levels);
  double factor_time, factor_comm_time;
  startTimer(&factor_time, &factor_comm_time);
  factorA(nodes, basis, cell_comm, levels);
  stopTimer(&factor_time, &factor_comm_time);

  int64_t factor_flops[3];
  get_factor_flops(factor_flops);
  int64_t sum_flops = factor_flops[0] + factor_flops[1] + factor_flops[2];
  double percent[3];
  for (int i = 0; i < 3; i++)
    percent[i] = (double)factor_flops[i] / (double)sum_flops * (double)100;

  int64_t lbegin = 0;
  content_length(NULL, NULL, &lbegin, &cell_comm[levels]);
  cudaMemcpy(&nodes[levels].X_ptr[lbegin * basis[levels].dimN], X1, lenX * sizeof(double), cudaMemcpyHostToDevice);

  double solve_time, solve_comm_time;
  startTimer(&solve_time, &solve_comm_time);

  for (int64_t i = levels; i > 0; i--)
    batchForwardULV(&nodes[i].params, &cell_comm[i]);
  chol_solve(&nodes[0].params, &cell_comm[0]);
  for (int64_t i = 1; i <= levels; i++)
    batchBackwardULV(&nodes[i].params, &cell_comm[i]);
  stopTimer(&solve_time, &solve_comm_time);

  cudaMemcpy(X1, &nodes[levels].X_ptr[lbegin * basis[levels].dimN], lenX * sizeof(double), cudaMemcpyDeviceToHost);

  loadX(X2, basis[levels].dimN, Xbody, ncells, cell, levels);
  double err;
  solveRelErr(&err, X1, X2, lenX);

  int64_t mem_basis, mem_A, mem_X;
  basis_mem(&mem_basis, basis, levels);
  node_mem(&mem_A, nodes, levels);
  rightHandSides_mem(&mem_X, rhs, levels);

  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  prog_time = MPI_Wtime() - prog_time;
  double cm_time;
  getCommTime(&cm_time);

  if (mpi_rank == 0)
    printf("LORASP: %d,%d,%lf,%d,%d,%d\nConstruct: %lf s. COMM: %lf s.\n"
      "Factorize: %lf s. COMM: %lf s.\n"
      "Solution: %lf s. COMM: %lf s.\n"
      "Factorization GFLOPS: %lf GFLOPS/s.\n"
      "GEMM: %lf%%, POTRF: %lf%%, TRSM: %lf%%\n"
      "Basis Memory: %lf GiB.\n"
      "Matrix Memory: %lf GiB.\n"
      "Vector Memory: %lf GiB.\n"
      "Err: Compress %e, Factor %e\n"
      "Program: %lf s. COMM: %lf s.\n",
      (int)Nbody, (int)(Nbody / Nleaf), theta, 3, (int)mpi_size, omp_get_max_threads(),
      construct_time, construct_comm_time, factor_time, factor_comm_time, solve_time, solve_comm_time, (double)sum_flops * 1.e-9 / factor_time,
      percent[0], percent[1], percent[2], (double)mem_basis * 1.e-9, (double)mem_A * 1.e-9, (double)mem_X * 1.e-9, cerr, err, prog_time, cm_time);

  for (int64_t i = 0; i <= levels; i++) {
    csc_free(&rels_far[i]);
    csc_free(&rels_near[i]);
    basis_free(&basis[i]);
    node_free(&nodes[i]);
    rightHandSides_free(&rhs[i]);
    cellComm_free(&cell_comm[i]);
  }
  csc_free(&cellFar);
  csc_free(&cellNear);
  
  free(body);
  free(Xbody);
  free(cell);
  free(rels_far);
  free(rels_near);
  free(cell_comm);
  free(basis);
  free(nodes);
  free(rhs);
  free(X1);
  free(X2);
  set_work_size(0, &Workspace, &Lwork);

  fin_libs();
  return 0;
}
