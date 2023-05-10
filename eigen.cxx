
#include "geometry.hxx"
#include "kernel.hxx"
#include "nbd.hxx"
#include "profile.hxx"

#include "omp.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include <limits>

// Uncomment the following line to print output in CSV format
#define OUTPUT_CSV
// Uncomment the following line to enable debug output
#define DEBUG_OUTPUT

constexpr double EPS = std::numeric_limits<double>::epsilon();

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  double prog_time = MPI_Wtime();
  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  int64_t Nbody = argc > 1 ? atol(argv[1]) : 8192;
  double theta = argc > 2 ? atof(argv[2]) : 0;
  int64_t leaf_size = argc > 3 ? atol(argv[3]) : 256;
  double epi = argc > 4 ? atof(argv[4]) : 1e-8;
  int64_t rank_max = argc > 5 ? atol(argv[5]) : 100;
  int64_t sp_pts = argc > 6 ? atol(argv[6]) : 2000;
  // Choice of Kernel
  // 0: Laplace
  // 1: Yukawa
  // 2: Gaussian
  // 3: Toeplitz (to be paired with geom = 0)
  int64_t kernel = argc > 7 ? atol(argv[7]) : 0;
  // Choice of Geometry
  // 0: 1D Line
  // 1: 2D Unit Circle
  // 2: 2D Unit Square
  // 3: 3D Unit Sphere
  // 4: 3D Unit Cube
  int64_t geom = argc > 8 ? atol(argv[8]) : 0;
  // Eigenvalue computation params
  double ev_tol = argc > 9 ? atof(argv[9]) : 1e-6;
  int64_t k_begin = argc > 10 ? atol(argv[10]) : 1;
  int64_t k_end = argc > 11 ? atol(argv[11]) : k_begin;
  double left = argc > 12 ? atof(argv[12]) : (double)-Nbody;
  double right = argc > 13 ? atof(argv[13]) : (double)Nbody;
  const char* fname = argc > 14 ? argv[14] : NULL;

  leaf_size = Nbody < leaf_size ? Nbody : leaf_size;
  int64_t levels = (int64_t)log2((double)Nbody / leaf_size);
  int64_t Nleaf = (int64_t)1 << levels;
  int64_t ncells = Nleaf + Nleaf - 1;

  // Initialize
  double* body = (double*)malloc(sizeof(double) * Nbody * 3);
  double* Xbody = (double*)malloc(sizeof(double) * Nbody);
  struct Cell* cell = (struct Cell*)malloc(sizeof(struct Cell) * ncells);
  struct CSC cellNear, cellFar;
  struct CSC* rels_far = (struct CSC*)malloc(sizeof(struct CSC) * (levels + 1));
  struct CSC* rels_near = (struct CSC*)malloc(sizeof(struct CSC) * (levels + 1));
  struct CellComm* cell_comm = (struct CellComm*)calloc(levels + 1, sizeof(struct CellComm));
  struct Base* basis = (struct Base*)calloc(levels + 1, sizeof(struct Base));
  struct Node* nodes = (struct Node*)malloc(sizeof(struct Node) * (levels + 1));

#ifdef DEBUG_OUTPUT
  if (mpi_rank == 0) {
    printf("Initializing global tree structure...\n");
  }
#endif
  constexpr double singularity = 1.e-3;
  constexpr double alpha = 1.;
  constexpr double p = 0.5;
  EvalDouble* eval;
  Laplace3D laplace(singularity);
  Yukawa3D yukawa(singularity, alpha);
  Gaussian gaussian(alpha);
  Toeplitz toeplitz(p);
  switch (kernel) {
    case 0: eval = &laplace;   break;
    case 1: eval = &yukawa;    break;
    case 2: eval = &gaussian;  break;
    case 3: eval = &toeplitz;  break;
    default: eval = &laplace;  break;
  }
  switch (geom) {
    case 0: mesh_line(body, Nbody);        break;
    case 1: mesh_unit_circle(body, Nbody); break;
    case 2: mesh_unit_square(body, Nbody); break;
    case 3: mesh_unit_sphere(body, Nbody); break;
    case 4: mesh_unit_cube(body, Nbody);   break;
    default: mesh_line(body, Nbody);       break;
  }
  buildTree(&ncells, cell, body, Nbody, levels);
  body_neutral_charge(Xbody, Nbody, 1., 999);

  traverse('N', &cellNear, ncells, cell, theta);
  traverse('F', &cellFar, ncells, cell, theta);

#ifdef DEBUG_OUTPUT
  if (mpi_rank == 0) {
    printf("Initializing local content...\n");
  }
#endif
  struct CommTimer timer;
  buildComm(cell_comm, ncells, cell, &cellFar, &cellNear, levels);
  for (int64_t i = 0; i <= levels; i++) {
    cell_comm[i].timer = &timer;
  }
  relations(rels_near, &cellNear, levels, cell_comm);
  relations(rels_far, &cellFar, levels, cell_comm);

  int64_t lbegin = 0, llen = 0;
  content_length(&llen, NULL, &lbegin, &cell_comm[levels]);
  int64_t gbegin = lbegin;
  i_global(&gbegin, &cell_comm[levels]);

  MPI_Barrier(MPI_COMM_WORLD);
#ifdef DEBUG_OUTPUT
  if (mpi_rank == 0) {
    printf("Building basis...\n");
  }
#endif
  double construct_time = MPI_Wtime(), construct_comm_time;
  buildBasis(*eval, basis, cell, &cellNear, levels, cell_comm, body, Nbody, epi, rank_max, sp_pts, 4);

  MPI_Barrier(MPI_COMM_WORLD);
  construct_time = MPI_Wtime() - construct_time;
  construct_comm_time = timer.get_comm_timing();

#ifdef DEBUG_OUTPUT
  if (mpi_rank == 0) {
    printf("Allocating nodes...\n");
  }
#endif
  double* Workspace = NULL;
  int64_t Lwork = 0;
  allocNodes(nodes, &Workspace, &Lwork, basis, rels_near, rels_far, cell_comm, levels);

#ifdef DEBUG_OUTPUT
  if (mpi_rank == 0) {
    printf("Generating coupling matrices...\n");
  }
#endif
  evalD(*eval, nodes[levels].A, &cellNear, cell, body, &cell_comm[levels]);
  for (int64_t i = 0; i <= levels; i++)
    evalS(*eval, nodes[i].S, &basis[i], &rels_far[i], &cell_comm[i]);

  int64_t lenX = rels_near[levels].N * basis[levels].dimN;
  double* X1 = (double*)calloc(lenX, sizeof(double));
  double* X2 = (double*)calloc(lenX, sizeof(double));
  loadX(X1, basis[levels].dimN, Xbody, 0, llen, &cell[gbegin]);

#ifdef DEBUG_OUTPUT
  if (mpi_rank == 0) {
    printf("Performing matvecA...\n");
  }
#endif  
  MPI_Barrier(MPI_COMM_WORLD);
  double matvec_time = MPI_Wtime(), matvec_comm_time;

  matVecA(nodes, basis, rels_near, X1, cell_comm, levels);

  matvec_time = MPI_Wtime() - matvec_time;
  matvec_comm_time = timer.get_comm_timing();

  double cerr = 0.;
  if (Nbody < 20000) {
#ifdef DEBUG_OUTPUT
    if (mpi_rank == 0) {
      printf("Generating matvec reference for construction error...\n");
    }
#endif
    int64_t body_local[2] = { cell[gbegin].Body[0], cell[gbegin + llen - 1].Body[1] };
    std::vector<double> X3(lenX);
    mat_vec_reference(*eval, body_local[0], body_local[1], &X3[0], Nbody, body, Xbody);
    loadX(X2, basis[levels].dimN, &X3[0], body_local[0], llen, &cell[gbegin]);

    solveAbsErr(&cerr, X1, X2, lenX);
    std::iter_swap(&X1, &X2);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  auto inertia = [&](const double shift, bool& is_singular)
  {
    EvalDouble* eval_shifted;
    Laplace3D laplace(singularity, -shift);
    Yukawa3D yukawa(singularity, alpha, -shift);
    Gaussian gaussian(alpha, -shift);
    Toeplitz toeplitz(p, -shift);
    switch (kernel) {
      case 0: eval_shifted = &laplace;   break;
      case 1: eval_shifted = &yukawa;    break;
      case 2: eval_shifted = &gaussian;  break;
      case 3: eval_shifted = &toeplitz;  break;
      default: eval_shifted = &laplace;  break;
    }
    // Reset
    for (int64_t level = 1; level <= levels; level++) {
      int64_t R = nodes[level].params.N_r, S = nodes[level].params.N_s, N = R + S;
      memset(nodes[level].params.A_data, 0, sizeof(double) * N * N * nodes[level].params.L_nnz);
    }
    memset(nodes[0].params.A_data, 0, sizeof(double) * nodes[0].params.N_r * nodes[0].params.N_r);
    // Fill shifted H2-Matrix
    evalD(*eval_shifted, nodes[levels].A, &cellNear, cell, body, &cell_comm[levels]);
    for (int64_t i = 0; i <= levels; i++)
      evalS(*eval_shifted, nodes[i].S, &basis[i], &rels_far[i], &cell_comm[i]);
    // Factorize shifted matrix
    MPI_Barrier(MPI_COMM_WORLD);
    for (int64_t i = levels; i > 0; i--)
      batchCholeskyFactor(&nodes[i].params, &cell_comm[i]);
    chol_decomp(&nodes[0].params, &cell_comm[0]);
    MPI_Barrier(MPI_COMM_WORLD);
    // Count number of negative entries in diagonal of U
    int64_t local_count = 0;
    for (int64_t level = 0; level <= levels; level++) {
      int64_t local_begin, local_len;
      content_length(&local_len, NULL, &local_begin, &cell_comm[level]);
      if (level == 0) {
        if (mpi_rank == 0) {
          for (int64_t i = 0; i < basis[level].Dims[0]; i++) {
            const double di = nodes[level].params.A_data[i * (basis[level].dimN + 1)];
            if(std::isnan(di) || std::abs(di) < EPS) is_singular = true;
            local_count += (di < 0 ? 1 : 0);
          }
        }
      }
      else {
        auto c = cell_comm[level].Comm_share;
        int rank_share = 0;
        if (c != MPI_COMM_NULL) {
          MPI_Comm_rank(c, &rank_share);
        }
        if (rank_share > 0) continue;
        for (int64_t i = 0; i < local_len; i++) {
          int64_t dim_i = basis[level].Dims[i + local_begin] - basis[level].DimsLr[i + local_begin];
          for (int64_t j = 0; j < dim_i; j++) {
            const double di = nodes[level].params.A_x[i][j * (basis[level].dimR + 1)];
            if(std::isnan(di) || std::abs(di) < EPS) is_singular = true;
            local_count += (di < 0 ? 1 : 0);
          }
        }
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, &local_count, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    return local_count;
  };
  // Check whether the interval [left,right] contain eigenvalue(s) of interest
#ifdef DEBUG_OUTPUT
  if (mpi_rank == 0) {
    printf("Checking whether starting interval contains target eigenvalue(s)...\n");
  }
#endif
  double bisection_time, bisection_comm_time;
  bool ss = false;
  const auto vl = inertia(left, ss);
  const auto vr = inertia(right, ss);
  if (!(k_begin > vl && k_end <= vr)) {
    if (mpi_rank == 0) {
      printf("Error: starting interval does not contain all target eigenvalues:"
             " inertia(left=%.3lf)=%d, inertia(right=%.3lf)=%d, target eigenvalue=[%d, %d]\n",
             left, (int)vl, right, (int)vr, (int)k_begin, (int)k_end);
    }
  }
  else {
    // Initialize global intervals
    const auto num_ev = k_end - k_begin + 1;
    std::vector<int64_t> bisect_k (num_ev);
    std::vector<double> bisect_left(num_ev, left), bisect_right(num_ev, right), bisect_ev(num_ev);
    for (int64_t idx = 0; idx < num_ev; idx++) {
      bisect_k[idx] = k_begin + idx;
    }
    int64_t idx_begin = 0, idx_end = num_ev - 1;
    MPI_Barrier(MPI_COMM_WORLD);
    bisection_time = MPI_Wtime();

#ifdef DEBUG_OUTPUT
    if (mpi_rank == 0) {
      printf("Starting bisection: inertia(left=%.3lf)=%d, inertia(right=%.3lf)=%d, k=[%d, %d]\n",
             left, (int)vl, right, (int)vr, (int)k_begin, (int)k_end);
    }
#endif
    auto compute_eigenvalue = [&](const int64_t idx) {
      bool is_singular = false;
      while ((bisect_right[idx] - bisect_left[idx]) >= ev_tol) {
        const auto mid = (bisect_left[idx] + bisect_right[idx]) / 2.;
        const auto vmid = inertia(mid, is_singular);
        if (is_singular) {
          printf("Shifted matrix becomes singular (shift=%.8lf)\n", mid);
          break;
        }
        // Update intervals accordingly
        for (int64_t i = 0; i < num_ev; i++) {
          const auto ki = bisect_k[i];
          if (ki <= vmid && mid < bisect_right[i]) {
            bisect_right[i] = mid;
          }
          if (ki > vmid && mid > bisect_left[i]) {
            bisect_left[i] = mid;
          }
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);  // Is this necessary?
      return (bisect_left[idx] + bisect_right[idx]) / 2.;
    };
    // Compute the smallest and largest eigenvalues first to narrow search interval
#ifdef DEBUG_OUTPUT
    if (mpi_rank == 0) {
      printf("Computing the %d-th eigenvalue...\n", (int)bisect_k[idx_begin]);
    }
#endif
    bisect_ev[idx_begin] = compute_eigenvalue(idx_begin);
    if (idx_begin < idx_end) {
#ifdef DEBUG_OUTPUT
      if (mpi_rank == 0) {
        printf("Computing the %d-th eigenvalue...\n", (int)bisect_k[idx_end]);
      }
#endif
      bisect_ev[idx_end] = compute_eigenvalue(idx_end);
    }
    idx_begin++;
    idx_end--;
    for (int64_t idx = idx_begin; idx <= idx_end; idx++) {
#ifdef DEBUG_OUTPUT
      if (mpi_rank == 0) {
        printf("Computing the %d-th eigenvalue...\n", (int)bisect_k[idx]);
      }
#endif
      bisect_ev[idx] = compute_eigenvalue(idx);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    bisection_time = MPI_Wtime() - bisection_time;
    bisection_comm_time = timer.get_comm_timing();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  prog_time = MPI_Wtime() - prog_time;

  if (mpi_rank == 0) {
#ifdef OUTPUT_CSV
#else
    // printf("mpi_nprocs=%d nthreads=%d N=%d leaf_size=%d height=%d theta=%.1lf epi=%.1e\n"
    //        "construct_min_rank=%d construct_max_rank=%d construct_error=%.3e "
    //        "construct_time=%.5lf construct_comm_time=%.5lf\n"
    //        "matvec_time=%.5lf matvec_comm_time=%.5lf\n"
    //        "factor_time=%.5lf factor_comm_time=%.5lf factor_gflops=%.3lf "
    //        "local_basis_memory=%.5lf local_matrix_memory=%.5lf\n"
    //        "total_time=%.5lf\n",
    //        (int)mpi_size, omp_get_max_threads(), (int)Nbody, (int)leaf_size, (int)levels, theta, epi,
    //        -1, -1, cerr, construct_time, construct_comm_time,
    //        matvec_time, matvec_comm_time, factor_time, factor_comm_time,
    //        (double)sum_flops * 1.e-9 / factor_time, percent[0], percent[1], percent[2],
    //        (double)mem_A[0] * 1.e-9, (double)mem_A[1] * 1.e-9, prog_time);
#endif
  }

  // Deallocate
#ifdef DEBUG_OUTPUT
  if (mpi_rank == 0) {
    printf("Deallocating...\n");
  }
#endif
  for (int64_t i = 0; i <= levels; i++) {
    csc_free(&rels_far[i]);
    csc_free(&rels_near[i]);
    basis_free(&basis[i]);
    node_free(&nodes[i]);
  }
  cellComm_free(cell_comm, levels);
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
  free(X1);
  free(X2);
  set_work_size(0, &Workspace, &Lwork);

  MPI_Finalize();
  return 0;
}
