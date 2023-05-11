
#include "geometry.hxx"
#include "kernel.hxx"
#include "nbd.hxx"
#include "profile.hxx"

#include "omp.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Uncomment the following line to print output in CSV format
#define OUTPUT_CSV
// Uncomment the following line to print the computed eigenvalues(s)
// #define PRINT_EV
// Uncomment the following line to enable debug output
// #define DEBUG_OUTPUT

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
  // 3: 2D Unit Square Grid
  // 4: 3D Unit Sphere
  // 5: 3D Unit Cube
  int64_t geom = argc > 8 ? atol(argv[8]) : 0;
  // Eigenvalue computation params
  double ev_tol = argc > 9 ? atof(argv[9]) : 1e-6;
  int64_t k_begin = argc > 10 ? atol(argv[10]) : 1;
  int64_t k_end = argc > 11 ? atol(argv[11]) : k_begin;
  double left = argc > 12 ? atof(argv[12]) : (double)-Nbody;
  double right = argc > 13 ? atof(argv[13]) : (double)Nbody;
  bool compute_ev_acc = argc > 14 ? (atol(argv[14]) == 1) : false;
  bool print_csv_header = argc > 15 ? (atol(argv[15]) == 1) : true;
  const char* fname = argc > 16 ? argv[16] : NULL;

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
  std::string kernel_name, geom_name;
  EvalDouble* eval;
  Laplace3D laplace(singularity);
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
  // TODO Obtain the following
  int64_t construct_min_rank = -1;
  int64_t construct_max_rank = -1;

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

  double construct_error = -1;
  if (Nbody < 20000) {
#ifdef DEBUG_OUTPUT
    if (mpi_rank == 0) {
      printf("Calculating construction error...\n");
    }
#endif
    int64_t body_local[2] = { cell[gbegin].Body[0], cell[gbegin + llen - 1].Body[1] };
    std::vector<double> X3(lenX);
    mat_vec_reference(*eval, body_local[0], body_local[1], &X3[0], Nbody, body, Xbody);
    loadX(X2, basis[levels].dimN, &X3[0], body_local[0], llen, &cell[gbegin]);

    solveAbsErr(&construct_error, X1, X2, lenX);
    std::iter_swap(&X1, &X2);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  int64_t factor_flops[3], local_mem[3], global_mem[3];
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
            if (std::isnan(di) || std::abs(di) < EPS) is_singular = true;
            if (di < 0.) local_count++;
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
            if (std::isnan(di) || std::abs(di) < EPS) is_singular = true;
            if (di < 0.) local_count++;
          }
        }
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, &local_count, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
#ifdef DEBUG_OUTPUT
    int64_t dense_cnt = 0;
    {
      std::vector<double> A_mat(Nbody * Nbody);
      struct Matrix A { A_mat.data(), Nbody, Nbody, Nbody };
      // Dense LDL Inertia
      gen_matrix(*eval_shifted, Nbody, Nbody, body, body, A.A, A.LDA);
      ldl_decomp(&A);
      for (int64_t i = 0; i < A.M; i++) {
        const double di = A.A[i + i * A.LDA];
        if (di < 0.) dense_cnt++;
      }
      if (mpi_rank == 0) {
        printf("shift=%.8lf HSS_Inertia=%d vs. Dense_LDL_Inertia=%d\n",
               -shift, (int)local_count, (int)dense_cnt);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    return local_count;

  };
  // Check whether the interval [left,right] contain eigenvalue(s) of interest
#ifdef DEBUG_OUTPUT
  if (mpi_rank == 0) {
    printf("Checking whether starting interval contains target eigenvalue(s)...\n");
  }
#endif
  const auto num_ev = k_end - k_begin + 1;
  std::vector<int64_t> bisect_k (num_ev);
  std::vector<double> bisect_left(num_ev, left), bisect_right(num_ev, right), bisect_ev(num_ev);
  double bisection_time, bisection_comm_time;
  bool ss;
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

    // Get local and global memory usage
    Profile profile;
    for (int64_t i = 0; i <= levels; i++) {
      profile.record_factor(basis[i].dimR, basis[i].dimN,
                            nodes[i].params.L_nnz, nodes[i].params.L_diag, nodes[i].params.L_rows);
    }
    profile.get_profile(factor_flops, local_mem);
    MPI_Reduce(local_mem, global_mem, 3, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  prog_time = MPI_Wtime() - prog_time;

  if (mpi_rank == 0) {
    const double local_mem_usage = ((double)local_mem[0] + (double)local_mem[1]) * 1.e-9;  // in GiB
    const double global_mem_usage = ((double)global_mem[0] + (double)global_mem[1]) * 1.e-9;  // in GiB
    std::vector<double> bisect_ev_abs_err(num_ev, -1), bisect_ev_rel_err(num_ev, -1), ref_ev(num_ev, -1);
    double max_ev_abs_err = -1;
    std::string max_ev_abs_ok = "YES";
    if (compute_ev_acc && Nbody < 20000) {
      std::vector<double> Adense_mat(Nbody * Nbody), ref_ev_all(Nbody);
      struct Matrix Adense { Adense_mat.data(), Nbody, Nbody, Nbody };
      gen_matrix(*eval, Nbody, Nbody, body, body, Adense.A, Adense.LDA);
      compute_all_eigenvalues(&Adense, ref_ev_all.data());
      for (int64_t i = 0; i < num_ev; i++) {
        const auto k = k_begin + i;
        ref_ev[i] = ref_ev_all[k - 1];
        bisect_ev_abs_err[i] = std::abs(ref_ev[i] - bisect_ev[i]);
        bisect_ev_rel_err[i] = bisect_ev_abs_err[i] / ref_ev[i];
      }
      max_ev_abs_err = *std::max_element(bisect_ev_abs_err.begin(), bisect_ev_abs_err.end());
      max_ev_abs_ok = max_ev_abs_err < (0.5 * ev_tol) ? "YES" : "NO";
    }
#ifdef OUTPUT_CSV
    if (print_csv_header) {
      printf("mpi_nprocs,nthreads,N,leaf_size,theta,epi,rank_max,sample_size,kernel,geometry,"
             "height,construct_min_rank,construct_max_rank,construct_error,construct_time,construct_comm_time,"
             "construct_local_mem,construct_global_mem,matvec_time,matvec_comm_time,"
             "k_begin,k_end,start_left,start_right,ev_tol,"
             "bisection_time,bisection_comm_time,prog_time,max_ev_abs_err,max_ev_abs_ok");
#ifdef PRINT_EV
      printf(",k,ref_ev,bisect_ev,ev_abs_err,ev_rel_err,ev_abs_ok");
#endif
      printf("\n");
    }

#ifdef PRINT_EV
    for (int64_t i = 0; i < num_ev; i++) {
      const auto k = k_begin + i;
      const std::string ev_abs_ok = bisect_ev_abs_err[i] < (0.5 * ev_tol) ? "YES" : "NO";
      printf("%d,%d,%d,%d,%.1lf,%.1e,%d,%d,%s,%s,"
             "%d,%d,%d,%.3e,%.3lf,%.3lf,"
             "%.5lf,%.5lf,%.3lf,%.3lf,"
             "%d,%d,%.3lf,%.3lf,%.1e,"
             "%.5lf,%.5lf,%.5lf,%.3e,%s,"
             "%d,%.5lf,%.5lf,%.3e,%.3e,%s\n",
             mpi_size, omp_get_max_threads(), (int)Nbody, (int)leaf_size, theta, epi, (int)rank_max,
             (int)sp_pts, kernel_name.c_str(), geom_name.c_str(), (int)levels, (int)construct_min_rank,
             (int)construct_max_rank, construct_error, construct_time, construct_comm_time,
             local_mem_usage, global_mem_usage, matvec_time, matvec_comm_time, (int)k_begin, (int)k_end,
             left, right, ev_tol, bisection_time, bisection_comm_time, prog_time, max_ev_abs_err,
             max_ev_abs_ok.c_str(), (int)k, ref_ev[i], bisect_ev[i], bisect_ev_abs_err[i],
             bisect_ev_rel_err[i], ev_abs_ok.c_str());
    }
#else
    printf("%d,%d,%d,%d,%.1lf,%.1e,%d,%d,%s,%s,"
           "%d,%d,%d,%.3e,%.3lf,%.3lf,"
           "%.5lf,%.5lf,%.3lf,%.3lf,"
           "%d,%d,%.3lf,%.3lf,%.1e,"
           "%.5lf,%.5lf,%.5lf,%.3e,%s\n",
           mpi_size, omp_get_max_threads(), (int)Nbody, (int)leaf_size, theta, epi, (int)rank_max,
           (int)sp_pts, kernel_name.c_str(), geom_name.c_str(), (int)levels, (int)construct_min_rank,
           (int)construct_max_rank, construct_error, construct_time, construct_comm_time,
           local_mem_usage, global_mem_usage, matvec_time, matvec_comm_time, (int)k_begin, (int)k_end,
           left, right, ev_tol, bisection_time, bisection_comm_time, prog_time, max_ev_abs_err,
           max_ev_abs_ok.c_str());
#endif
#else
    printf("MPI_NProcs=%d NThreads=%d N=%d Leaf_Size=%d Theta=%.1lf Epi=%.1e Rank_Max=%d Sample_Size=%d\n"
           "Kernel=%s Geometry=%s Height=%d Basis_Min_Rank=%d Basis_Max_Rank=%d Construct_Error=%.3e\n"
           "Construct_Time=%.3lf Construct_Comm_Time=%.3lf Local_Memory=%.3lf GiB Global_Memory=%.3lf GiB\n"
           "MatVec_Time=%.3lf MatVec_Comm_Time=%.3lf\n"
           "Bisection_Time=%.5lf Bisection_Comm_Time=%.5lf EV_Max_Abs_Err=%.3e EV_Max_Abs_OK=%s\n"
           "Prog_Time=%.5lf\n",
           mpi_size, omp_get_max_threads(), (int)Nbody, (int)leaf_size, theta, epi, (int)rank_max,
           (int)sp_pts, kernel_name.c_str(), geom_name.c_str(), (int)levels, (int)construct_min_rank,
           (int)construct_max_rank, construct_error, construct_time, construct_comm_time,
           local_mem_usage, global_mem_usage, matvec_time, matvec_comm_time, bisection_time,
           bisection_comm_time, max_ev_abs_err, max_ev_abs_ok.c_str(), prog_time);
    for (int64_t i = 0; i < num_ev; i++) {
      const auto k = k_begin + i;
      const std::string ev_abs_ok = bisect_ev_abs_err[i] < (0.5 * ev_tol) ? "YES" : "NO";
      printf("K=%d Ref_EV=%.5lf Bisection_EV=%.5lf EV_Abs_Err=%.3e EV_Rel_Err=%.3e EV_Abs_OK=%s\n",
             (int)k, ref_ev[i], bisect_ev[i], bisect_ev_abs_err[i], bisect_ev_rel_err[i], ev_abs_ok.c_str());
    }
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
