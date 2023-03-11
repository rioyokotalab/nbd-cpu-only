
#include "nbd.hxx"
#include "profile.hxx"

#include "stdio.h"

void matrix_mem(int64_t* bytes, const struct Matrix* A, int64_t lenA) {
  int64_t count = sizeof(struct Matrix) * lenA;
  for (int64_t i = 0; i < lenA; i++)
    count = count + sizeof(double) * A[i].M * A[i].N;
  *bytes = count;
}

void basis_mem(int64_t* bytes, const struct Base* basis, int64_t levels) {
  int64_t count = sizeof(struct Base) * levels;
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nodes = basis[i].Ulen;
    int64_t bytes_o, bytes_c, bytes_r;
    matrix_mem(&bytes_o, &basis[i].Uo[0], nodes);
    matrix_mem(&bytes_c, &basis[i].Uc[0], nodes);
    matrix_mem(&bytes_r, &basis[i].R[0], nodes);

    count = count + bytes_o + bytes_c + bytes_r;
  }
  *bytes = count;
}

void node_mem(int64_t* bytes, const struct Node* node, int64_t levels) {
  int64_t count = sizeof(struct Node) * levels;
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nnz = node[i].lenA;
    int64_t bytes_a;
    matrix_mem(&bytes_a, &node[i].A[0], nnz);
    count = count + bytes_a;
  }
  *bytes = count;
}

void rightHandSides_mem(int64_t* bytes, const struct RightHandSides* rhs, int64_t levels) {
  int64_t count = sizeof(struct RightHandSides) * levels;
  for (int64_t i = 0; i <= levels; i++) {
    int64_t len = rhs[i].Xlen;
    int64_t bytes_x, bytes_o, bytes_c, bytes_b;
    matrix_mem(&bytes_x, &rhs[i].X[0], len);
    matrix_mem(&bytes_o, &rhs[i].Xo[0], len);
    matrix_mem(&bytes_c, &rhs[i].Xc[0], len);
    matrix_mem(&bytes_b, &rhs[i].B[0], len);
    count = count + bytes_x + bytes_o + bytes_c + bytes_b;
  }
  *bytes = count;
}

int64_t gemm_flops = 0;
int64_t potrf_flops = 0;
int64_t trsm_flops = 0;

void record_factor_flops(int64_t dimr, int64_t dims, int64_t nnz, int64_t ndiag) {
  if (dims == 0 && nnz == 1)
    potrf_flops = potrf_flops + dimr * dimr * dimr / 3;
  else {
    int64_t dimn = dimr + dims;
    int64_t fgemm = 4 * dimn * dimn * dimn * nnz;
    int64_t fsplit = 2 * dimn * dimr * (dimn + dimr) * ndiag;
    int64_t fchol = dimr * dimr * dimr * ndiag / 3;
    int64_t ftrsm = dimn * dimr * dimr * ndiag;
    int64_t fschur = 2 * dims * dims * dimr * ndiag;
    gemm_flops = gemm_flops + fgemm + fsplit + fschur;
    potrf_flops = potrf_flops + fchol;
    trsm_flops = trsm_flops + ftrsm;
  }
}

void get_factor_flops(int64_t flops[3]) {
  flops[0] = gemm_flops;
  flops[1] = potrf_flops;
  flops[2] = trsm_flops;
  MPI_Allreduce(MPI_IN_PLACE, flops, 3, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  gemm_flops = 0;
  potrf_flops = 0;
  trsm_flops = 0;
}

double tot_cm_time = 0.;

void startTimer(double* wtime, double* cmtime) {
  MPI_Barrier(MPI_COMM_WORLD);
  *wtime = MPI_Wtime();
  *cmtime = tot_cm_time;
}

void stopTimer(double* wtime, double* cmtime) {
  MPI_Barrier(MPI_COMM_WORLD);
  double etime = MPI_Wtime();
  double time[2] = { etime - *wtime, tot_cm_time - *cmtime };
  MPI_Allreduce(MPI_IN_PLACE, time, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  int mpi_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  *wtime = time[0] / mpi_size;
  *cmtime = time[1] / mpi_size;
}

void recordCommTime(double cmtime) {
  tot_cm_time = tot_cm_time + cmtime;
}

void getCommTime(double* cmtime) {
#ifndef _PROF
  printf("Communication time not recorded: Profiling macro not defined, compile lib with -D_PROF.\n");
  *cmtime = -1;
#else
  *cmtime = tot_cm_time;
#endif
}
