
#include "nbd.hxx"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cinttypes>

double _singularity = 1.e-8;
double _alpha = 1.;

void laplace3d(double* r2) {
  double _r2 = *r2;
  double r = sqrt(_r2) + _singularity;
  *r2 = 1. / r;
}

void yukawa3d(double* r2) {
  double _r2 = *r2;
  double r = sqrt(_r2) + _singularity;
  *r2 = exp(_alpha * -r) / r;
}

void set_kernel_constants(double singularity, double alpha) {
  _singularity = singularity;
  _alpha = alpha;
}

void gen_matrix(void(*ef)(double*), int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda) {
  if (m > 0)
    for (int64_t x = 0; x < n; x++) {
      const double* bjj = &bj[x * 3];
      for (int64_t y = 0; y < m; y++) {
        const double* bii = &bi[y * 3];
        double dX = bii[0] - bjj[0];
        double dY = bii[1] - bjj[1];
        double dZ = bii[2] - bjj[2];
        double r2 = dX * dX + dY * dY + dZ * dZ;
        Aij[y + x * lda] = r2;
      }
    }

  if (m > 0)
    for (int64_t x = 0; x < n; x++)
      for (int64_t y = 0; y < m; y++)
        ef(&Aij[y + x * lda]);
}

void mat_vec_reference(void(*ef)(double*), int64_t begin, int64_t end, double B[], int64_t nbodies, const double* bodies, const double Xbodies[]) {
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
      ef(&r2);
      s = s + r2 * Xbodies[j];
    }
    B[i] = s;
  }
}

/*__global__ void gen_matrix_kernel(struct Functor* func, int m, int n, const double* bi, const double* bj, double Aij[], int lda) {
  if (m > 0)
    for (int x = 0; x < n; x++) {
      const double* bjj = &bj[x * 3];
      for (int y = 0; y < m; y++) {
        const double* bii = &bi[y * 3];
        double dX = bii[0] - bjj[0];
        double dY = bii[1] - bjj[1];
        double dZ = bii[2] - bjj[2];
        double r2 = dX * dX + dY * dY + dZ * dZ;
        Aij[y + x * lda] = func->operator()(r2);
        printf("%d %d %e %e\n", y, x, r2, Aij[y + x * lda]);
      }
    }
}

void gen_matrix_gpu(struct Functor* func, int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda, cudaStream_t stream) {
  gen_matrix_kernel <<< 1, 1, 0, stream >>> (func, m, n, bi, bj, Aij, lda);
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}
*/