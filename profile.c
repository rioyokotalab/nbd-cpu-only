
#include "nbd.h"
#include "profile.h"

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
    int64_t nnz_f = node[i].lenS;
    int64_t bytes_a, bytes_s;
    matrix_mem(&bytes_a, &node[i].A[0], nnz);
    matrix_mem(&bytes_s, &node[i].S[0], nnz_f);
    count = count + bytes_a + bytes_s;
  }
  *bytes = count;
}

void rightHandSides_mem(int64_t* bytes, const struct RightHandSides* rhs, int64_t levels) {
  int64_t count = sizeof(struct RightHandSides) * levels;
  for (int64_t i = 0; i <= levels; i++) {
    int64_t len = rhs[i].Xlen;
    int64_t bytes_x, bytes_o, bytes_c, bytes_b;
    matrix_mem(&bytes_x, &rhs[i].X[0], len);
    matrix_mem(&bytes_o, &rhs[i].XoL[0], len);
    matrix_mem(&bytes_c, &rhs[i].XcM[0], len);
    matrix_mem(&bytes_b, &rhs[i].B[0], len);
    count = count + bytes_x + bytes_o + bytes_c + bytes_b;
  }
  *bytes = count;
}

int64_t factor_flops = 0;

void record_factor_flops(int64_t dimr, int64_t dims, int64_t nnz, int64_t ndiag) {
  int64_t dimn = dimr + dims;
  int64_t fgemm = 2 * dimn * dimn * dimn * nnz;
  int64_t fsplit = 2 * dimn * dimr * (dimn + dimr) * ndiag;
  int64_t fchol = dimr * dimr * dimr * ndiag / 3;
  int64_t ftrsm = dimn * dimr * dimr * ndiag / 2;
  int64_t fschur = 2 * dims * dims * dimr * ndiag;
  factor_flops = factor_flops + fgemm + fsplit + fchol + ftrsm + fschur;
}

void get_factor_flops(int64_t* flops) {
  *flops = factor_flops;
  MPI_Allreduce(MPI_IN_PLACE, flops, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  factor_flops = 0;
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
