
#include "nbd.hxx"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cinttypes>

double _singularity = 1.e-8;
double _alpha = 1.;
__constant__ double _singularity_gpu = 1.e-8;
__constant__ double _alpha_gpu = 1.;

double laplace3d(double r2) {
  double r = sqrt(r2) + _singularity;
  return 1. / r;
}

__device__ double __laplace3d(double r2) {
  double r = sqrt(r2) + _singularity_gpu;
  return 1. / r;
}
__constant__ double(*laplace3d_gpu)(double) = __laplace3d;

void laplace3d_addr(double(**ef_cpu)(double), double(**ef_gpu)(double)) {
  *ef_cpu = laplace3d;
  cudaMemcpyFromSymbol(ef_gpu, (const void*)&laplace3d_gpu, sizeof(double(*)(double)));
}

double yukawa3d(double r2) {
  double r = sqrt(r2) + _singularity;
  return exp(_alpha * -r) / r;
}

__device__ double __yukawa3d(double r2) {
  double r = sqrt(r2) + _singularity_gpu;
  return exp(_alpha_gpu * -r) / r;
}
__constant__ double(*yukawa3d_gpu)(double) = __yukawa3d;

void yukawa3d_addr(double(**ef_cpu)(double), double(**ef_gpu)(double)) {
  *ef_cpu = yukawa3d;
  cudaMemcpyFromSymbol(ef_gpu, (const void*)&yukawa3d_gpu, sizeof(void (*)(double*)));
}

void set_kernel_constants(double singularity, double alpha) {
  _singularity = singularity;
  _alpha = alpha;
  cudaMemcpyToSymbol(_singularity_gpu, &singularity, sizeof(double));
  cudaMemcpyToSymbol(_alpha_gpu, &alpha, sizeof(double));
}

void gen_matrix(double(*func)(double), int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda) {
  if (m > 0)
    for (int64_t x = 0; x < n; x++) {
      const double* bjj = &bj[x * 3];
      for (int64_t y = 0; y < m; y++) {
        const double* bii = &bi[y * 3];
        double dX = bii[0] - bjj[0];
        double dY = bii[1] - bjj[1];
        double dZ = bii[2] - bjj[2];
        double r2 = dX * dX + dY * dY + dZ * dZ;
        Aij[y + x * lda] = func(r2);
      }
    }
}

void mat_vec_reference(double(*func)(double), int64_t begin, int64_t end, double B[], int64_t nbodies, const double* bodies, const double Xbodies[]) {
  int64_t m = end - begin;
  int64_t n = nbodies;
  for (int64_t i = 0; i < m; i++) {
    int64_t y = begin + i;
    const double* bi = &bodies[y * 3];
    double s = 0.;
    for (int64_t j = 0; j < n; j++) {
      const double* bj = &bodies[j * 3];
      double dX = bi[0] - bj[0];
      double dY = bi[1] - bj[1];
      double dZ = bi[2] - bj[2];
      double r2 = dX * dX + dY * dY + dZ * dZ;
      s = s + func(r2) * Xbodies[j];
    }
    B[i] = s;
  }
}

__global__ void gen_matrix_kernel(double(*func)(double), int m, int n, const double* bi, const double* bj, double Aij[], int lda) {
  if (m > 0)
    for (int x = 0; x < n; x++) {
      const double* bjj = &bj[x * 3];
      for (int y = 0; y < m; y++) {
        const double* bii = &bi[y * 3];
        double dX = bii[0] - bjj[0];
        double dY = bii[1] - bjj[1];
        double dZ = bii[2] - bjj[2];
        double r2 = dX * dX + dY * dY + dZ * dZ;
        Aij[y + x * lda] = func(r2);
      }
    }
}

void gen_matrix_gpu(double(*func)(double), int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda, cudaStream_t stream) {
  double(*ef1)(double) = NULL;
  double(*ef2)(double) = NULL;
  laplace3d_addr(&ef1, &ef2);
  gen_matrix_kernel <<< 1, 1, 0, stream >>> (ef2, m, n, bi, bj, Aij, lda);
}
