
#include "build_tree.h"
#include "kernel.h"
#include "aca.h"
#include "test_util.h"

#include <cstdio>
#include <random>
#include <chrono>

int main(int argc, char* argv[]) {

  using namespace nbd;

  int m = 64;
  int dim = 2;
  int leaf = 16;

  Bodies b1(m);
  auto fun = l2d();

  std::srand(199);
  for (auto& i : b1) {
    for (int x = 0; x < dim; x++)
	    i.X[x] = 1 * ((real_t)std::rand() / RAND_MAX);
  }

  Cells c1 = buildTree(b1, leaf, dim);

  Matrices d, lr;
  traverse(fun, c1, c1, dim, d, lr, 1., 8);
  
  for (auto& i : d)
    printMat(i, i.M, i.N, i.LDA);

  printTree(&c1[0]);

  return 0;
}
