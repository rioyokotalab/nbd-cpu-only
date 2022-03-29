
#include "minblas.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <chrono>


void compress_using_aca(int m, int n, int r, double* a, int lda);

void compress_using_id(int m, int n, int r, double* a, int lda);

double rel2err(const double* a, const double* ref, int m, int n, int M, int ldref) {
  double err = 0., nrm = 0.;
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      double e = ref[i + j * ldref] - a[i + j * M];
      err += e * e;
      nrm += a[i + j * M] * a[i + j * M];
    }
  }

  return std::sqrt(err / nrm);
}


int main(int argc, char* argv[]) {

  int r = argc > 1 ? atoi(argv[1]) : 16;
  int m = argc > 2 ? atoi(argv[2]) : 32;
  int n = argc > 3 ? atoi(argv[3]) : m;

  r = std::min(r, m);
  r = std::min(r, n);

  std::srand(199);
  std::vector<double> left(m * r), right(n * r);

  for(auto& i : left)
    i = ((double)std::rand() / RAND_MAX) * 100;
  for(auto& i : right)
    i = ((double)std::rand() / RAND_MAX) * 100;

  std::vector<double> a(m * n);

  for(int j = 0; j < n; j++) {
    for(int i = 0; i < m; i++) {
      double e = 0.;
      for(int k = 0; k < r; k++)
        e += left[i + k * m] * right[j + k * n];
      a[i + j * m] = e;
    }
  }

  compress_using_aca(m, n, r, a.data(), m);

  compress_using_id(m, n, r, a.data(), m);

  return 0;
}


void compress_using_aca(int m, int n, int r, double* a, int lda) {
  int rp = r + 8;
  std::vector<double> u(m * rp);
  std::vector<double> v(n * rp);
  std::vector<double> b(m * n);
  std::vector<double> c(m * n);

  for (int j = 0; j < n; j++)
    for (int i = 0; i < m; i++)
      c[i + j * m] = a[i + j * lda];

  int64_t iters;
  dlra(1.e-12, m, n, rp, c.data(), u.data(), m, v.data(), n, &iters, NULL);

  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      double e = 0.;
      for (int k = 0; k < iters; k++)
        e += u[i + k * m] * v[j + k * n];
      b[i + j * lda] = e;
    }
  }

  printf("aca rel err: %e, aca iters %lld\n", rel2err(b.data(), a, m, n, m, lda), iters);
}


void compress_using_id(int m, int n, int r, double* a, int lda) {
  int rp = r + 8;
  std::vector<double> u(m * rp);
  std::vector<double> v(n * rp);
  std::vector<double> b(m * n);
  std::vector<int64_t> pu(rp);
  std::vector<double> c(m * n);

  for (int j = 0; j < n; j++)
    for (int i = 0; i < m; i++)
      c[i + j * m] = a[i + j * lda];

  int64_t iters;
  didrow(1.e-12, m, n, rp, c.data(), u.data(), m, pu.data(), &iters);

  for (int i = 0; i < iters; i++) {
    int row = pu[i];
    for (int j = 0; j < n; j++) {
      v[j + i * m] = a[row + j * lda];
    }
  }

  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      double e = 0.;
      for (int k = 0; k < iters; k++)
        e += u[i + k * m] * v[j + k * n];
      b[i + j * lda] = e;
    }
  }

  printf("id rel err: %e, aca iters %lld\n", rel2err(b.data(), a, m, n, m, lda), iters);
}
