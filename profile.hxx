
#pragma once

#include <cstdint>
#include <cstddef>

struct Profile {
  int64_t gemm_flops = 0;
  int64_t potrf_flops = 0;
  int64_t trsm_flops = 0;

  int64_t bytes_matrix = 0;
  int64_t bytes_basis = 0;
  int64_t bytes_vector = 0;

  void record_factor(int64_t dimr, int64_t dimn, int64_t nnz, int64_t ndiag, int64_t nrows) {
    if (dimr == 0 && nnz == 1) {
      potrf_flops += + dimn * dimn * dimn / 3;
      bytes_matrix += dimn * dimn * sizeof(double);
      bytes_vector += dimn * sizeof(double);
    }
    else {
      int64_t dims = dimn - dimr;
      int64_t fgemm = 4 * dimn * dimn * dimn * nnz;
      int64_t fsplit = 2 * dimn * dimr * (dimn + dimr) * ndiag;
      int64_t fchol = dimr * dimr * dimr * ndiag / 3;
      int64_t ftrsm = dimn * dimr * dimr * ndiag;
      int64_t fschur = 2 * dims * dims * dimr * ndiag;
      gemm_flops += + fgemm + fsplit + fschur;
      potrf_flops += fchol;
      trsm_flops += ftrsm;
      bytes_matrix += dimn * dimn * nnz * sizeof(double);
      bytes_basis += dimn * dimn * nrows * sizeof(double);
      bytes_vector += dimn * nrows * sizeof(double);
    }
  }

  void get_profile(int64_t flops[3], int64_t bytes[3]) {
    flops[0] = gemm_flops;
    flops[1] = potrf_flops;
    flops[2] = trsm_flops;
    bytes[0] = bytes_matrix;
    bytes[1] = bytes_basis;
    bytes[2] = bytes_vector;

    gemm_flops = 0;
    potrf_flops = 0;
    trsm_flops = 0;
    bytes_matrix = 0;
    bytes_basis = 0;
    bytes_vector = 0;
  }
};






