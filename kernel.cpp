
#include "kernel.h"
#include "build_tree.h"
#include "lra.h"

#include <cmath>
#include <cstdio>

using namespace nbd;

EvalFunc nbd::l2d() {
  EvalFunc ef;
  ef.r2f = [](real_t& r2, real_t singularity, real_t alpha) -> void {
    r2 = r2 == 0 ? singularity : std::log(std::sqrt(r2));
  };
  ef.singularity = 1.e5;
  ef.alpha = 1.;
  return ef;
}

EvalFunc nbd::l3d() {
  EvalFunc ef;
  ef.r2f = [](real_t& r2, real_t singularity, real_t alpha) -> void {
    r2 = r2 == 0 ? singularity : 1. / std::sqrt(r2);
  };
  ef.singularity = 1.e5;
  ef.alpha = 1.;
  return ef;
}


void nbd::eval(EvalFunc ef, const Body* bi, const Body* bj, int dim, real_t* out) {
  real_t& r2 = *out;
  r2 = 0.;
  for (int i = 0; i < dim; i++) {
    real_t dX = bi->X[i] - bj->X[i];
    r2 += dX * dX;
  }
  ef.r2f(r2, ef.singularity, ef.alpha);
}


void nbd::mvec_kernel(EvalFunc ef, const Cell* ci, const Cell* cj, int dim, real_t alpha, const real_t* x_vec, int incx, real_t beta, real_t* b_vec, int incb) {
  int m = ci->NBODY, n = cj->NBODY;

  for (int y = 0; y < m; y++) {
    real_t sum = 0.;
    for (int x = 0; x < n; x++) {
      real_t r2;
      eval(ef, ci->BODY + y, cj->BODY + x, dim, &r2);
      sum += r2 * x_vec[x * incx];
    }
    b_vec[y * incb] = alpha * sum + beta * b_vec[y * incb];
  }
}


void nbd::P2Pnear(EvalFunc ef, const Cell* ci, const Cell* cj, int dim, Matrix& a) {
  int m = ci->NBODY, n = cj->NBODY;
  a.A.resize((size_t)m * n);
  a.M = m;
  a.N = n;

  for (int i = 0; i < (size_t)m * n; i++) {
    int x = i / m, y = i - x * m;
    real_t r2;
    eval(ef, ci->BODY + y, cj->BODY + x, dim, &r2);
    a.A[(size_t)x * a.M + y] = r2;
  }
}

void nbd::P2Pfar(EvalFunc ef, const Cell* ci, const Cell* cj, int dim, Matrix& a, int rank) {
  int m = ci->NBODY, n = cj->NBODY;
  a.M = m;
  a.N = n;
  a.A.reserve(((size_t)m + n) * rank);
  a.A.resize((size_t)m * rank);

  std::vector<real_t> v((size_t)n * rank);

  int iters;
  daca_cells(ef, ci, cj, dim, rank, a.A.data(), m, v.data(), n, &iters);
  a.A.resize(((size_t)m + n) * iters);
  std::copy(v.begin(), v.begin() + ((size_t)n * iters), a.A.begin() + ((size_t)m * iters));
}

void nbd::SampleP2Pi(Matrix& s, const Matrix& a) {
  if (a.A.size() < a.M * a.N) {
    int r = (int)a.A.size() / (a.M + a.N);
    drspl(s.M, a.N, r, a.A.data(), a.M, a.A.data() + ((size_t)a.M * r), a.N, s.N, s.A.data(), s.M);
  }
}

void nbd::SampleP2Pj(Matrix& s, const Matrix& a) {
  if (a.A.size() < a.M * a.N) {
    int r = (int)a.A.size() / (a.M + a.N);
    drspl(s.M, a.M, r, a.A.data() + ((size_t)a.M * r), a.N, a.A.data(), a.M, s.N, s.A.data(), s.M);
  }
}

void nbd::SampleParent(Matrix& sc, const Matrix& sp, int c_off) {
  ddspl(sc.M, sp.N, sp.A.data() + c_off, sp.M, sc.N, sc.A.data(), sc.M);
}

void nbd::BasisOrth(Matrix& s, Matrix& r) {
  if (s.M * s.N > 0) {
    r.M = std::min(s.M, s.N);
    r.N = s.N;
    r.A.resize((size_t)r.M * r.N, 0);
    dorth(s.M, s.N, s.A.data(), s.M, r.A.data(), r.M);
  }
}

void atbc(int m, int n, int k, const real_t* a, int lda, const real_t* b, int ldb, real_t* c, int ldc) {
  for (int i = 0; i < (size_t)m * n; i++) {
    int x = i / m, y = i - x * m;
    real_t e = 0.;
    for (int j = 0; j < k; j++)
      e = e + a[(size_t)y * lda + j] * b[(size_t)x * ldb + j];
    c[(size_t)x * ldc + y] = e;
  }
}

void nbd::BasisInvRightAndMerge(const Matrix& s, Matrix& a) {
  if (a.A.size() < a.M * a.N && s.M == a.N) {
    int r = (int)a.A.size() / (a.M + a.N);

    // vxs = v * s;
    std::vector<real_t> vxs((size_t)r * s.N);
    real_t* vt = a.A.data() + ((size_t)a.M * r);
    atbc(r, s.N, a.N, vt, a.N, s.A.data(), s.M, vxs.data(), r);

    // b = u * vxs;
    std::vector<real_t> b((size_t)a.M * s.N);
    for (int i = 0; i < (size_t)a.M * s.N; i++) {
      int x = i / a.M, y = i - x * a.M;
      real_t e = 0.;
      for (int k = 0; k < r; k++)
        e = e + a.A[(size_t)k * a.M + y] * vxs[(size_t)x * r + k];
      b[(size_t)x * a.M + y] = e;
    }

    a.N = s.N;
    a.A.resize((size_t)a.M * a.N);
    std::copy(b.begin(), b.end(), a.A.begin());
  }
}

void nbd::BasisInvLeft(const Matrix& s, Matrix& a) {
  if (s.M == a.M && a.M * a.N > 0) {
    // b = st * a;
    std::vector<real_t> b((size_t)s.N * a.N);
    atbc(s.N, a.N, s.M, s.A.data(), s.M, a.A.data(), a.M, b.data(), s.N);

    a.M = s.N;
    a.A.resize((size_t)a.M * a.N);
    std::copy(b.begin(), b.end(), a.A.begin());
  }
}


void nbd::BasisInvMultipleLeft(const Matrix* s, int ls, Matrix& a) {
  if (a.M * a.N > 0) {
    int m = 0;
    for (const Matrix* si = s; si != s + ls; si++)
      m += si->N;
    std::vector<real_t> b((size_t)m * a.N);

    real_t* ai = a.A.data();
    real_t* bi = b.data();
    for (const Matrix* si = s; si != s + ls; si++) {
      int mi = si->N, ki = si->M;
      atbc(mi, a.N, ki, si->A.data(), si->M, ai, a.M, bi, m);
      ai = &ai[ki];
      bi = &bi[mi];
    }

    a.M = m;
    a.A.resize((size_t)a.M * a.N);
    std::copy(b.begin(), b.end(), a.A.begin());
  }

}
