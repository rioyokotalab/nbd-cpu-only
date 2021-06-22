
#include "aca.h"
#include "kernel.h"
#include "svd.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <chrono>

using namespace nbd;

void compress_using_tsvd_aca(int m, int n, int r, double* a, int lda);

void compress_using_tsvd(int m, int n, double* a, int lda);

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

  compress_using_tsvd_aca(m, n, r, a.data(), m);
  compress_using_tsvd(m, n, a.data(), m);


  return 0;
}


void compress_using_tsvd_aca(int m, int n, int r, double* a, int lda) {
  int rp = r + 8;
  std::vector<double> u(m * rp), v(n * rp);

  int iters;
  daca(m, n, rp, a, lda, u.data(), m, v.data(), n, &iters);

  std::vector<double> s(iters);
  dlr_svd(m, n, iters, u.data(), m, v.data(), n, s.data());

  double err = 0., nrm = 0.;
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      double e = 0.;
      for (int k = 0; k < iters; k++)
        e += u[i + k * m] * v[j + k * n] * s[k];
      e -= a[i + j * lda];
      err += e * e;
      nrm += a[i + j * lda] * a[i + j * lda];
    }
  }
  printf("aca_svd rel err: %e, aca iters %d\n", std::sqrt(err / nrm), iters);
}


void compress_using_tsvd(int m, int n, double* a, int lda) {

  int iters;
  std::vector<double> b(m * n), full_u(m * std::min(m, n)), full_v(n * std::min(m, n));
  std::vector<double> full_s(std::min(m, n));

  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      b[i + j * m] = a[i + j * lda];
    }
  }

  dsvd(m, n, b.data(), m, full_s.data(), full_u.data(), m, full_v.data(), n);
  dtsvd(m, n, std::min(m, n), full_s.data(), &iters);

  double err = 0., nrm = 0.;
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      double e = 0.;
      for (int k = 0; k < iters; k++)
        e += full_u[i + k * m] * full_v[j + k * n] * full_s[k];
      e -= a[i + j * lda];
      err += e * e;
      nrm += a[i + j * lda] * a[i + j * lda];
    }
  }

  printf("svd rel err: %e, svd rank %d\n", std::sqrt(err / nrm), iters);
}