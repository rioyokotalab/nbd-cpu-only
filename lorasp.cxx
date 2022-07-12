
#include "kernel.h"
#include "linalg.h"
#include "umv.h"
#include "dist.h"

#include <cstdio>
#include <cmath>
#include <iostream>
#include <algorithm>

void loadX(struct Matrix* X, const struct Cell* cell, const struct Body* bodies, int64_t level) {
  int64_t xlen = (int64_t)1 << level;
  contentLength(&xlen, level);
  int64_t len = (int64_t)1 << level;
  const struct Cell* leaves = &cell[len - 1];

  for (int64_t i = 0; i < xlen; i++) {
    int64_t gi = i;
    iGlobal(&gi, i, level);
    const struct Cell* ci = &leaves[gi];

    struct Matrix* Xi = &X[i];
    int64_t nbegin = ci->BODY[0];
    int64_t ni = ci->BODY[1] - nbegin;
    matrixCreate(Xi, ni, 1);
    for (int64_t n = 0; n < ni; n++)
      Xi->A[n] = bodies[n + nbegin].B;
  }
}

void h2MatVecReference(struct Matrix* B, void(*ef)(double*), const struct Cell* cell, const struct Body* bodies, int64_t level) {
  int64_t nbodies = cell->BODY[1];
  int64_t xlen = (int64_t)1 << level;
  contentLength(&xlen, level);
  int64_t len = (int64_t)1 << level;
  const struct Cell* leaves = &cell[len - 1];

  for (int64_t i = 0; i < xlen; i++) {
    int64_t gi = i;
    iGlobal(&gi, i, level);
    const struct Cell* ci = &leaves[gi];

    struct Matrix* Bi = &B[i];
    int64_t ibegin = ci->BODY[0];
    int64_t m = ci->BODY[1] - ibegin;
    matrixCreate(Bi, m, 1);

    int64_t block = 500;
    int64_t last = nbodies % block;
    struct Matrix X;
    struct Matrix Aij;
    matrixCreate(&X, block, 1);
    matrixCreate(&Aij, m, block);
    zeroMatrix(&X);
    zeroMatrix(&Aij);

    if (last > 0) {
      for (int64_t k = 0; k < last; k++)
        X.A[k] = bodies[k].B;
      gen_matrix(ef, m, last, &bodies[ibegin], bodies, Aij.A, NULL, NULL);
      mmult('N', 'N', &Aij, &X, Bi, 1., 0.);
    }
    else
      zeroMatrix(Bi);

    for (int64_t j = last; j < nbodies; j += block) {
      for (int64_t k = 0; k < block; k++)
        X.A[k] = bodies[k + j].B;
      gen_matrix(ef, m, block, &bodies[ibegin], &bodies[j], Aij.A, NULL, NULL);
      mmult('N', 'N', &Aij, &X, Bi, 1., 1.);
    }

    matrixDestroy(&Aij);
    matrixDestroy(&X);
  }
}

void traverse_dist(const struct CSC* cellFar, const struct CSC* cellNear, int64_t levels) {
  int64_t mpi_rank, mpi_levels;
  commRank(&mpi_rank, &mpi_levels);

  configureComm(levels, NULL, 0);
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nodes = i > mpi_levels ? (int64_t)1 << (i - mpi_levels) : 1;
    int64_t lvl_diff = i < mpi_levels ? mpi_levels - i : 0;
    int64_t my_rank = mpi_rank >> lvl_diff;
    int64_t gbegin = my_rank * nodes;

    int64_t offc = (int64_t)(1 << i) - 1;
    int64_t nc = offc + gbegin;
    int64_t nbegin = cellNear->COL_INDEX[nc];
    int64_t nlen = cellNear->COL_INDEX[nc + nodes] - nbegin;
    int64_t fbegin = cellFar->COL_INDEX[nc];
    int64_t flen = cellFar->COL_INDEX[nc + nodes] - fbegin;
    int64_t* ngbs = (int64_t*)malloc(sizeof(int64_t) * (nlen + flen));

    for (int64_t j = 0; j < nlen; j++) {
      int64_t ngb = cellNear->ROW_INDEX[nbegin + j] - offc;
      ngb /= nodes;
      ngbs[j] = ngb;
    }
    for (int64_t j = 0; j < flen; j++) {
      int64_t ngb = cellFar->ROW_INDEX[fbegin + j] - offc;
      ngb /= nodes;
      ngbs[j + nlen] = ngb;
    }

    std::sort(ngbs, &ngbs[nlen + flen]);
    const int64_t* iter = std::unique(ngbs, &ngbs[nlen + flen]);
    int64_t size = iter - ngbs;
    configureComm(i, &ngbs[0], size);
    free(ngbs);
  }
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  initComm();

  int64_t Nbody = argc > 1 ? atol(argv[1]) : 8192;
  double theta = argc > 2 ? atof(argv[2]) : 1;
  int64_t leaf_size = 256;

  double epi = 1.e-10;
  int64_t rank_max = 100;
  int64_t sp_pts = 2000;

  int64_t levels = (int64_t)std::log2(Nbody / leaf_size);
  int64_t Nleaf = (int64_t)1 << levels;

  int64_t mpi_rank;
  int64_t mpi_levels;
  commRank(&mpi_rank, &mpi_levels);
  
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

  traverse_dist(&cellFar, &cellNear, levels);
  std::vector<CellComm> cell_comm(levels + 1);
  buildComm(cell_comm.data(), ncells, cell.data(), &cellFar, &cellNear, levels);

  SpDense sp;
  allocSpDense(sp, levels);
  relations(&sp.RelsNear[0], ncells, cell.data(), &cellNear, levels);
  relations(&sp.RelsFar[0], ncells, cell.data(), &cellFar, levels);
  allocBasis(sp.Basis.data(), levels, ncells, cell.data(), cell_comm.data());

  double construct_time, construct_comm_time;
  startTimer(&construct_time, &construct_comm_time);
  evaluateBaseAll(ef, &sp.Basis[0], ncells, cell.data(), &sp.RelsNear[0], levels, cell_comm.data(), body.data(), Nbody, epi, rank_max, sp_pts);
  stopTimer(&construct_time, &construct_comm_time);
  
  allocNodes(sp.D.data(), sp.Basis.data(), sp.RelsNear.data(), sp.RelsFar.data(), levels);

  evaluate('N', sp.D[levels].A.data(), ef, ncells, &cell[0], body.data(), &sp.RelsNear[levels], levels);
  for (int64_t i = 0; i <= levels; i++)
    evaluate('F', sp.D[i].S.data(), ef, ncells, &cell[0], body.data(), &sp.RelsFar[i], i);

  double factor_time, factor_comm_time;
  startTimer(&factor_time, &factor_comm_time);
  factorSpDense(sp);
  stopTimer(&factor_time, &factor_comm_time);

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
  solveRelErr(&err, rhs[levels].X.data(), Xref.data(), &cell_comm[levels]);

  int64_t dim = 3;

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
  cellComm_free(cell_comm.data(), levels);
  
  closeComm();
  MPI_Finalize();
  return 0;
}
