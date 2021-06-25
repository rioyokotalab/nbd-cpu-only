
#include "test_util.h"

#include <iostream>
#include <random>
#include <cmath>

using namespace nbd;

void nbd::printVec(real_t* a, int n, int inc) {
  for (int i = 0; i < n; i++)
    std::cout << a[i * inc] << " ";
  std::cout << std::endl;
}

void nbd::printMat(real_t* a, int m, int n, int lda) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++)
      std::cout << a[i + j * lda] << " ";
    std::cout << std::endl;
  }
}


void nbd::initRandom(Bodies& b, int m, int dim, real_t min, real_t max, unsigned int seed) {
  if (seed)
    std::srand(seed);
  for (auto& i : b) {
    for (int x = 0; x < dim; x++)
      i.X[x] = (max - min) * ((real_t)std::rand() / RAND_MAX) + min;
  }
}

void nbd::vecRandom(real_t* a, int n, int inc, real_t min, real_t max, unsigned int seed) {
  if (seed)
    std::srand(seed);
  for (int i = 0; i < n; i++) {
    a[i * inc] = (max - min) * ((real_t)std::rand() / RAND_MAX) + min;
  }
}


void nbd::printTree(const Cell* cell, int level, int offset_c, int offset_b) {
  for (int i = 0; i < level; i++)
    printf("  ");
  printf("%d: <%d, %d>", offset_c, offset_b, offset_b + cell->NBODY);
  printf(" <Far: ");
  for (auto& c : cell->listFar)
    printf("%d ", offset_c + (int)(c - cell));
  printf("> <Near: ");
  for (auto& c : cell->listNear)
    printf("%d ", offset_c + (int)(c - cell));
  printf(">\n");
  for (auto c = cell->CHILD; c != cell->CHILD + cell->NCHILD; c++)
    printTree(c, level + 1, offset_c + (int)(c - cell), offset_b + (int)(c->BODY - cell->BODY));
}

real_t nbd::rel2err(const real_t* a, const real_t* ref, int m, int n, int lda, int ldref) {
  real_t err = 0., nrm = 0.;
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      real_t e = ref[i + j * ldref] - a[i + j * lda];
      err += e * e;
      nrm += a[i + j * lda] * a[i + j * lda];
    }
  }

  return std::sqrt(err / nrm);
}

