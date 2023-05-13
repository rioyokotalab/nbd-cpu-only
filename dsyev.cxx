
#include "geometry.hxx"
#include "kernel.hxx"
#include "nbd.hxx"

#include "omp.h"
#include "mpi.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#ifdef USE_MKL
#include "mkl.h"
#else
#include "cblas.h"
#include "lapacke.h"
#endif

// Uncomment the following line to print output in CSV format
#define OUTPUT_CSV

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int mpi_rank, mpi_size, mpi_grid[2] = {0};
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  if (mpi_size > 1) {
    printf("Only single process is allowed here\n");
    return 0;
  }

  // Parse Input
  int Nbody = argc > 1 ? atol(argv[1]) : 1024;
  // Choice of Kernel
  // 0: Laplace
  // 1: Yukawa
  // 2: Gaussian
  // 3: Toeplitz (to be paired with geom = 0)
  int64_t kernel = argc > 2 ? atol(argv[2]) : 0;
  // Choice of Geometry
  // 0: 1D Line
  // 1: 2D Unit Circle
  // 2: 2D Unit Square
  // 3: 2D Unit Square Grid
  // 4: 3D Unit Sphere
  // 5: 3D Unit Cube
  int64_t geom = argc > 3 ? atol(argv[3]) : 0;
  bool print_csv_header = argc > 4 ? (atol(argv[4]) == 1) : true;

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

  // Initialize Matrix
  std::vector<double> Adata(Nbody * Nbody);
  std::vector<double> ref_ev(Nbody);
  struct Matrix A { Adata.data(), Nbody, Nbody, Nbody };
  gen_matrix(*eval, A.M, A.N, body, body, A.A, A.LDA);

  // Compute all eigenvalues with dsyev
  MPI_Barrier(MPI_COMM_WORLD);
  double dsyev_time = MPI_Wtime();

  compute_all_eigenvalues(&A, ref_ev.data());

  MPI_Barrier(MPI_COMM_WORLD);
  dsyev_time = MPI_Wtime() - dsyev_time;

  // Print outputs
  if (mpi_rank == 0) {
#ifndef OUTPUT_CSV
    printf("NThreads=%d N=%d Kernel=%s Geometry=%s DSYEV_Time=%.5lf\n",
           omp_get_max_threads(), Nbody, kernel_name.c_str(), geom_name.c_str(), dsyev_time);
#else
    if (print_csv_header == 1) {
      printf("nthreads,N,kernel,geometry,dsyev_time\n");
    }
    printf("%d,%d,%s,%s,%.5lf\n",
           omp_get_max_threads(), Nbody, kernel_name.c_str(), geom_name.c_str(), dsyev_time);
#endif
  }

  free(body);
  MPI_Finalize();
  return 0;
}
