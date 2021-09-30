
#pragma once

#include "kernel.h"
#include <iostream>
#include <random>
#include <cmath>

using namespace nbd;

Matrix mat(int m, int n) {
  Matrix a;
  a.M = m;
  a.N = n;
  a.A.resize(m * n);
  return a;
}

void initRandom(Bodies& b, int m, int dim, double min, double max, unsigned int seed) {
  if (seed)
    std::srand(seed);
  for (auto& i : b) {
    for (int x = 0; x < dim; x++)
      i.X[x] = (max - min) * ((double)std::rand() / RAND_MAX) + min;
  }
}

void vecRandom(double* a, int n, int inc, double min, double max, unsigned int seed = 0) {
  if (seed)
    std::srand(seed);
  for (int i = 0; i < n; i++) {
    a[i * inc] = (max - min) * ((double)std::rand() / RAND_MAX) + min;
  }
}

void mulUV(const Matrix& uv, double* a, int lda) {
  int r = uv.A.size() / (uv.M + uv.N);
  const double* vt = &uv.A[r * uv.M];
  for (int jj = 0; jj < uv.N; jj++)
    for (int ii = 0; ii < uv.M; ii++) {
      double e = 0.;
      for (int k = 0; k < r; k++)
        e += uv.A[ii + (size_t)k * uv.M] * vt[jj + (size_t)k * uv.N];
      a[ii + (size_t)jj * lda] = e;
    }
}

void convertHmat2Dense(EvalFunc ef, int dim, const Cells& icells, const Cells& jcells, const Matrices& d, double* a, int lda) {
  auto j_begin = jcells[0].BODY;
  auto i_begin = icells[0].BODY;
  int ld = (int)icells.size();

#pragma omp parallel for
  for (int y = 0; y < icells.size(); y++) {
    auto i = icells[y];
    auto yi = i.BODY - i_begin;
    for (auto& j : i.listNear) {
      auto _x = j - &jcells[0];
      auto xi = j->BODY - j_begin;
      Matrix m;
      P2Pnear(ef, &icells[y], &jcells[_x], dim, m);
      for (int jj = 0; jj < m.N; jj++)
        for (int ii = 0; ii < m.M; ii++)
          a[ii + yi + (jj + xi) * lda] = m.A[ii + (size_t)jj * m.M];
    }
    for (auto& j : i.listFar) {
      auto _x = j - &jcells[0];
      auto xi = j->BODY - j_begin;
      const Matrix& m = d[y + _x * ld];
      mulUV(m, a + yi + xi * lda, lda);
    }
  }
}


void convertFullBase(const Cell* cell, const Matrix* base, Matrix* base_full) {
  if (cell->NCHILD == 0) {
    base_full->M = base->M;
    base_full->N = base->N;
    base_full->A.resize(base->M * base->N);
    std::copy(base->A.begin(), base->A.end(), base_full->A.begin());
    return;
  }

  for (auto c = cell->CHILD; c != cell->CHILD + cell->NCHILD; c++) {
    auto i = c - cell;
    convertFullBase(c, base + i, base_full + i);
  }

  if(base->N > 0) {
    base_full->M = cell->NBODY;
    base_full->N = base->N;
    base_full->A.resize(cell->NBODY * base->N);
    int y_off = 0, k_off = 0;
    for (auto c = cell->CHILD; c != cell->CHILD + cell->NCHILD; c++) {
      auto i = c - cell;
      for (int jj = 0; jj < base->N; jj++)
        for (int ii = 0; ii < c->NBODY; ii++) {
          double e = 0.;
          for (int k = 0; k < base_full[i].N; k++)
            e += base_full[i].A[ii + (size_t)k * base_full[i].M] * base->A[k_off + k + (size_t)jj * base->M];
          double* bf = &(base_full->A)[y_off];
          bf[ii + (size_t)jj * base_full->M] = e;
        }
      k_off += base_full[i].N;
      y_off += c->NBODY;
    }
  }

}

void mulUSV(const Matrix& u, const Matrix& v, const Matrix& s, double* a, int lda) {
  Matrix us;
  us.M = u.M;
  us.N = s.N;
  us.A.resize(us.M * us.N);

  for (int jj = 0; jj < s.N; jj++)
    for (int ii = 0; ii < u.M; ii++) {
      double e = 0.;
      for (int k = 0; k < u.N; k++)
        e += u.A[ii + (size_t)k * u.M] * s.A[k + (size_t)jj * s.M];
      us.A[ii + (size_t)jj * us.M] = e;
    }

  for (int jj = 0; jj < v.M; jj++)
    for (int ii = 0; ii < u.M; ii++) {
      double e = 0.;
      for (int k = 0; k < s.N; k++)
        e += us.A[ii + (size_t)k * us.M] * v.A[jj + (size_t)k * v.M];
      a[ii + (size_t)jj * lda] = e;
    }
}


void convertH2mat2Dense(EvalFunc ef, int dim, const Cells& icells, const Cells& jcells, const Matrices& ibase, const Matrices& jbase, const Matrices& d, double* a, int lda) {
  auto j_begin = jcells[0].BODY;
  auto i_begin = icells[0].BODY;
  int ld = (int)icells.size();

  Matrices ibase_full(icells.size()), jbase_full(jcells.size());
  convertFullBase(&icells[0], &ibase[0], &ibase_full[0]);
  convertFullBase(&jcells[0], &jbase[0], &jbase_full[0]);

#pragma omp parallel for
  for (int y = 0; y < icells.size(); y++) {
    auto i = icells[y];
    auto yi = i.BODY - i_begin;
    for (auto& j : i.listNear) {
      auto _x = j - &jcells[0];
      auto xi = j->BODY - j_begin;
      Matrix m;
      P2Pnear(ef, &icells[y], &jcells[_x], dim, m);
      for (int jj = 0; jj < m.N; jj++)
        for (int ii = 0; ii < m.M; ii++)
          a[ii + yi + (jj + xi) * lda] = m.A[ii + (size_t)jj * m.M];
    }
    for (auto& j : i.listFar) {
      auto _x = j - &jcells[0];
      auto xi = j->BODY - j_begin;
      const Matrix& m = d[y + _x * ld];
      mulUSV(ibase_full[y], jbase_full[_x], m, a + yi + xi * lda, lda);
    }
  }
}


double rel2err(const double* a, const double* ref, int m, int n, int lda, int ldref) {
  double err = 0., nrm = 0.;
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      double e = ref[i + j * ldref] - a[i + j * lda];
      err += e * e;
      nrm += a[i + j * lda] * a[i + j * lda];
    }
  }

  return std::sqrt(err / nrm);
}

