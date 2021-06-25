
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


void nbd::convertHmat2Dense(const Cells& icells, const Cells& jcells, const Matrices& d, real_t* a, int lda) {
  auto j_begin = jcells[0].BODY;
  auto i_begin = icells[0].BODY;
  int ld = (int)icells.size();

  for (auto& i : icells) {
    auto y = &i - icells.data();
    auto yi = i.BODY - i_begin;
    for (auto& j : i.listNear) {
      auto _x = j - &jcells[0];
      auto xi = j->BODY - j_begin;
      const Matrix& m = d[y + _x * ld];
      for (int jj = 0; jj < m.N; jj++)
        for (int ii = 0; ii < m.M; ii++)
          a[ii + yi + (jj + xi) * lda] = m.A[ii + (size_t)jj * m.LDA];
    }
    for (auto& j : i.listFar) {
      auto _x = j - &jcells[0];
      auto xi = j->BODY - j_begin;
      const Matrix& m = d[y + _x * ld];
      for (int jj = 0; jj < m.N; jj++)
        for (int ii = 0; ii < m.M; ii++) {
          double e = 0.;
          for (int k = 0; k < m.R; k++)
            e += m.A[ii + (size_t)k * m.LDA] * m.B[jj + (size_t)k * m.LDB];
          a[ii + yi + (jj + xi) * lda] = e;
        }
    }
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

