
#include "geometry.hxx"
#include "kernel.hxx"

#include "omp.h"
#include "mpi.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

// Uncomment the following line to print output in CSV format
#define OUTPUT_CSV

// ScaLAPACK Fortran Interface
extern "C" {
  /* Cblacs declarations: https://netlib.org/blacs/BLACS/QRef.html */
  void Cblacs_pinfo(int*, int*);
  void Cblacs_get(int CONTEXT, int WHAT, int*VALUE);
  void Cblacs_gridinit(int*, const char*, int, int);
  // returns the co-ordinates of the process number PNUM in PROW and PCOL.
  void Cblacs_pcoord(int CONTEXT, int PNUM, int* PROW, int* PCOL);
  void Cblacs_gridexit(int CONTEXT);
  void Cblacs_barrier(int, const char*);
  void Cblacs_exit(int CONTINUE);
  void Cblacs_gridmap(int* CONTEXT, int* USERMAP, const int LDUMAP,
                      const int NPROW, const int NPCOL);
  void Cblacs_gridinfo(int CONTEXT, int *NPROW, int *NPCOL,
                       int *MYROW, int *MYCOL);

  // calculate the number of rows and cols owned by process IPROC.
  // IPROC :: (local input) INTEGER
  //          The coordinate of the process whose local array row or
  //          column is to be determined.
  // ISRCPROC :: The coordinate of the process that possesses the first
  //             row or column of the distributed matrix. Global input.
  int numroc_(const int* N, const int* NB, const int* IPROC, const int* ISRCPROC,
              const int* NPROCS);

  // init descriptor for scalapack matrices.
  void descinit_(int *desc,
                 const int *m,  const int *n, const int *mb, const int *nb,
                 const int *irsrc, const int *icsrc,
                 const int *BLACS_CONTEXT,
                 const int *lld, int *info);

  // set values of the descriptor without error checking.
  void descset_(int *desc, const int *m,  const int *n, const int *mb,
                const int *nb, const int *irsrc, const int *icsrc, const int *BLACS_CONTEXT,
                const int *lld, int *info);
  // Compute reference eigenvalues
  void pdsyev_(char*, char*, int*, double*, int*, int*, int*,
               double*, double*, int*, int*, int*, double*, int*, int*);
  // Compute selected eigenvalues
  void pdsyevx_(char*, char*, char*, int*, double*, int*, int*, int*,
                double*, double*, int*, int*, double*, int*, int*,
                double*, double*, double*, int*, int*, int*,
                double*, int*, int*, int*, int*, int*, double*, int*);
}

// Translate global indices to local indices. INDXGLOB is the global index for the
// row/col. Returns the local FORTRAN-style index. NPROCS is the number of processes
// in that row or col.
int indxg2l(int INDXGLOB, int NB, int NPROCS) {
  return NB * ((INDXGLOB - 1) / ( NB * NPROCS)) + (INDXGLOB - 1) % NB + 1;
}
int indxl2g(int indxloc, int nb, int iproc, int isrcproc, int nprocs) {
  return nprocs * nb * ((indxloc - 1) / nb) +
    (indxloc-1) % nb + ((nprocs + iproc - isrcproc) % nprocs) * nb + 1;
}
int indxg2p(int INDXGLOB, int NB, int ISRCPROC, int NPROCS) {
  return (ISRCPROC + (INDXGLOB - 1) / NB) % NPROCS;
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int mpi_rank, mpi_size, mpi_grid[2] = {0};
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Dims_create(mpi_size, 2, mpi_grid);

  // Parse Input
  int Nbody = argc > 1 ? atol(argv[1]) : 1024;
  int NB = argc > 2 ? atol(argv[2]) : 32;
  // Choice of Kernel
  // 0: Laplace
  // 1: Yukawa
  // 2: Gaussian
  // 3: Toeplitz (to be paired with geom = 0)
  int64_t kernel = argc > 3 ? atol(argv[3]) : 0;
  // Choice of Geometry
  // 0: 1D Line
  // 1: 2D Unit Circle
  // 2: 2D Unit Square
  // 3: 2D Unit Square Grid
  // 4: 3D Unit Sphere
  // 5: 3D Unit Cube
  int64_t geom = argc > 4 ? atol(argv[4]) : 0;
  // Eigenvalue computation parameters
  double abs_tol = argc > 5 ? atof(argv[5]) : 1e-3;
  const int64_t k_begin = argc > 6 ? atol(argv[6]) : 1;
  const int64_t k_end = argc > 7 ? atol(argv[7]) : k_begin;
  bool print_csv_header = argc > 8 ? (atol(argv[8]) == 1) : true;

  // Initialize
  double* body = (double*)malloc(sizeof(double) * Nbody * 3);
  constexpr double singularity = 1.e-3;
  constexpr double alpha = 1.;
  constexpr double p = 0.5;
  std::string kernel_name, geom_name;
  EvalDouble* eval;
  Laplace2D laplace(singularity);
  Yukawa3D yukawa(singularity, alpha);
  Gaussian gaussian(alpha);
  Toeplitz toeplitz(p);
  switch (kernel) {
    case 0: {
      kernel_name = "Laplace";
      eval = &laplace;
      break;
    }
    case 1: {
      kernel_name = "Yukawa";
      eval = &yukawa;
      break;
    }
    case 2: {
      kernel_name = "Gaussian";
      eval = &gaussian;
      break;
    }
    case 3: {
      kernel_name = "Toeplitz";
      eval = &toeplitz;
      break;
    }
    default: {
      kernel_name = "Laplace";
      eval = &laplace;
      break;
    }
  }
  switch (geom) {
    case 0: {
      geom_name = "Line";
      mesh_line(body, Nbody);
      break;
    }
    case 1: {
      geom_name = "UnitCircle";
      mesh_unit_circle(body, Nbody);
      break;
    }
    case 2: {
      geom_name = "UnitSquare";
      mesh_unit_square(body, Nbody);
      break;
    }
    case 3: {
      geom_name = "UnitSquareGrid";
      uniform_unit_cube(body, Nbody, 2);
      double c[3] = { 0, 0, 0 };
      double r[3] = { 1, 1, 0 };
      magnify_reloc(body, Nbody, c, c, r, sqrt(Nbody));
      break;
    }
    case 4: {
      geom_name = "UnitSphere";
      mesh_unit_sphere(body, Nbody);
      break;
    }
    case 5:{
      geom_name = "UnitCube";
      mesh_unit_cube(body, Nbody);
      break;
    }
    default: {
      geom_name = "Line";
      mesh_line(body, Nbody);
      break;
    }
  }
  
  // BLACS Initialization
  int ZERO = 0;
  int ONE = 1;
  int blacs_context, blacs_rank, blacs_nprocs, blacs_prow, blacs_pcol;
  // Get BLACS context
  Cblacs_get(0, 0, &blacs_context);
  // Initialize BLACS grid
  Cblacs_gridinit(&blacs_context, "Row", mpi_grid[0], mpi_grid[1]);
  // Get information about grid and local processes
  Cblacs_pinfo(&blacs_rank, &blacs_nprocs);
  Cblacs_gridinfo(blacs_context, &mpi_grid[0], &mpi_grid[1], &blacs_prow, &blacs_pcol);
  const int local_nrows = numroc_(&Nbody, &NB, &blacs_prow, &ZERO, &mpi_grid[0]);
  const int local_ncols = numroc_(&Nbody, &NB, &blacs_pcol, &ZERO, &mpi_grid[1]);
  const int local_stride = local_nrows;

  // Initialize local matrix
  int info, desc[9];
  descinit_(desc, &Nbody, &Nbody, &NB, &NB, &ZERO, &ZERO, &blacs_context, &local_stride, &info);
  std::vector<double> A(local_nrows * local_ncols);
  for (int64_t i = 0; i < local_nrows; i++) {
    for (int64_t j = 0; j < local_ncols; j++) {
      const int g_row = indxl2g(i + 1, NB, blacs_prow, ZERO, mpi_grid[0]) - 1;
      const int g_col = indxl2g(j + 1, NB, blacs_pcol, ZERO, mpi_grid[1]) - 1;
      const double *bi = body + 3 * g_row;
      const double *bj = body + 3 * g_col;
      const double x = bi[0] - bj[0];
      const double y = bi[1] - bj[1];
      const double z = bi[2] - bj[2];
      const double d = std::sqrt(x * x + y * y + z * z);
      A[i + j * local_stride] = (*eval)(d);
    }
  }

  // Compute selected eigenvalues with pdsyevx
  char jobz = 'N';
  char range = 'I';
  char uplo = 'L';
  int il = k_begin;
  int iu = k_end;
  int m;
  std::vector<double> computed_ev(Nbody);
  int LWORK; double *WORK;
  int NNP, LIWORK; int *IWORK;
  double pdsyevx_time = 0;
  MPI_Barrier(MPI_COMM_WORLD);
  pdsyevx_time = MPI_Wtime();
  // PDSYEVX Work Query
  {
    LWORK = -1;
    LIWORK = -1;
    WORK = new double[1];
    IWORK = new int[1];
    pdsyevx_(&jobz, &range, &uplo, &Nbody, A.data(), &ONE, &ONE, desc, nullptr, nullptr,
             &il, &iu, &abs_tol, &m, nullptr, computed_ev.data(), nullptr, nullptr, nullptr,
             nullptr, nullptr, WORK, &LWORK, IWORK, &LIWORK, nullptr, nullptr, nullptr, &info);
    if (info != 0) {
      printf("Process-%d: Error in pdsyevx workspace query, info=%d\n", mpi_rank, info);
    }
    LWORK = (int)WORK[0];
    // Manual LIWORK calculation
    // For some reason workspace query gives a very big LIWORK, which requires 9.5 GB of memory
    // Use minimal LIWORK instead based on ScaLAPACK reference:
    // LIWORK >= 6 * NNP
    // Where NNP = MAX( N, NPROW*NPCOL + 1, 4 )
    NNP = Nbody;
    NNP = std::max(NNP, mpi_grid[0] * mpi_grid[1] + 1);
    NNP = std::max(NNP, (int)4);
    LIWORK = std::min(6 * NNP, IWORK[0]);
    delete[] WORK;
    delete[] IWORK;
  }
  // PDSYEVX Computation
  {
    WORK = new double[LWORK];
    IWORK = new int[LIWORK];
    pdsyevx_(&jobz, &range, &uplo, &Nbody, A.data(), &ONE, &ONE, desc, nullptr, nullptr,
             &il, &iu, &abs_tol, &m, nullptr, computed_ev.data(), nullptr, nullptr, nullptr,
             nullptr, nullptr, WORK, &LWORK, IWORK, &LIWORK, nullptr, nullptr, nullptr, &info);
    delete[] WORK;
    delete[] IWORK;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  pdsyevx_time = MPI_Wtime() - pdsyevx_time;

  // Print outputs
  if (mpi_rank == 0) {
#ifndef OUTPUT_CSV
    printf("NProcs=%d NThreads=%d N=%d NB=%d Kernel=%s Geometry=%s\n"
           "K_Begin=%d K_End=%d Abs_Tol=%.1e PDSYEVX_Time=%.5lf\n",
           blacs_nprocs, omp_get_max_threads(), Nbody, NB, kernel_name.c_str(), geom_name.c_str(),
           (int)k_begin, (int)k_end, abs_tol, pdsyevx_time);
#else
    if (print_csv_header == 1) {
      printf("nprocs,nthreads,N,NB,kernel,geometry,k_begin,k_end,abs_tol,pdsyevx_time\n");
    }
    printf("%d,%d,%d,%d,%s,%s,%d,%d,%.1e,%.5lf\n",
           blacs_nprocs, omp_get_max_threads(), Nbody, NB, kernel_name.c_str(), geom_name.c_str(),
           (int)k_begin, (int)k_end, abs_tol, pdsyevx_time);
#endif
  }

  free(body);

  Cblacs_gridexit(blacs_context);
  Cblacs_exit(1);
  MPI_Finalize();
  return 0;
}
