
#include "build_tree.h"
#include "kernel.h"
#include "aca.h"
#include "test_util.h"

#include <cstdio>
#include <random>
#include <chrono>

int main(int argc, char* argv[]) {

  int m = 64;
  int dim = 2;
  int p = 10;

  using namespace nbd;

  Bodies b1(m), b2(m);

  std::srand(199);
  for (auto& i : b1) {
    for (int x = 0; x < dim; x++)
	    i.X[x] = ((real_t)std::rand() / RAND_MAX);
  }

  for (auto& i : b2) {
    for (int x = 0; x < dim; x++)
      i.X[x] = ((real_t)std::rand() / RAND_MAX) + 10;
  }

  Cells c1 = buildTree(b1, m, dim);
  Cells c2 = buildTree(b2, m, dim);

  P2M_L2P(l3d(), p, &c1[0], dim);
  P2M_L2P(l3d(), p, &c2[0], dim);

  Matrix s;
  M2L(l3d(), &c1[0], &c2[0], dim, s);

  Matrix ref(m * m);
  dense_kernel(l3d(), &c1[0], &c2[0], dim, &ref[0], m);

  Matrix us(m * p);
  for (int j = 0; j < p; j++) {
    for (int i = 0; i < m; i++) {
      double e = 0.;
      for (int k = 0; k < p; k++)
        e += c1[0].V[i + k * m] * s[k + j * p];
      us[i + j * m] = e;
    }
  }

  Matrix a(m * m);
  for (int j = 0; j < m; j++) {
    for (int i = 0; i < m; i++) {
      double e = 0.;
      for (int k = 0; k < p; k++)
        e += us[i + k * m] * c2[0].V[j + k * m];
      a[i + j * m] = e;
    }
  }

  double err = 0., nrm = 0.;
  for (int j = 0; j < m; j++) {
    for (int i = 0; i < m; i++) {
      double e = ref[i + j * m];
      e -= a[i + j * m];
      printf("%d %d, %e\n", i, j, e);
      err += e * e;
      nrm += ref[i + j * m] * ref[i + j * m];
    }
  }

  printf("far field rel err: %e\n", std::sqrt(err / nrm));
  


  return 0;
}
