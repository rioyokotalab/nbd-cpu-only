
#include "build_tree.h"
#include "kernel.h"
#include "aca.h"

#include <cstdio>
#include <random>
#include <chrono>

int main(int argc, char* argv[]) {

  int m = 16, n = m;
  int leaf = m / 2;

  using namespace nbd;

  Bodies b(m);

  std::srand(199);
  for (auto& i : b) {
    for (int x = 0; x < 2; x++)
	  i.X[x] = ((real_t)std::rand() / RAND_MAX) * 100;
  }

  Cells c = buildTree(b, leaf, 2);

  /*Dense d(m, m);
  Dense u(m, m);
  Dense v(m, m);

  dense_kernel(l3d(), &c[0], &c[0], 2, d.elements, d.ld);
  d.print();

  raca(l3d(), &c[0], &c[0], 2, m, u.elements, u.ld, v.elements, v.ld);
  printf("\n");
  u.print();
  printf("\n");
  v.print();*/

  return 0;
}
