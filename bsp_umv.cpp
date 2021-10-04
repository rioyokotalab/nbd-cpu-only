
#include "bsp_umv.h"
#include "kernel.h"

#include <lapacke.h>
#include <cblas.h>
#include <random>
#include <cstdio>

using namespace nbd;

Node nbd::node(EvalFunc ef, int dim, const Cell* i, const Cell* j) {
  Node n;
  n.M = i->NCHILD;
  n.N = j->NCHILD;
  n.A.resize((size_t)n.M * n.N);

  if(n.M * n.N > 0)
#pragma omp parallel for
    for (int y = 0; y < n.M; y++) {
      auto& cy = i->CHILD[y];
      for (auto& cx : cy.listNear) {
        int x = cx - j->CHILD;
        P2Pnear(ef, &cy, cx, dim, n.A[y + (size_t)x * n.M]);
      }
    }
  
  return n;
}

int nbd::a_inv_b(real_t repi, const Matrix& A, const Matrix& B, Matrix& C) {

  C.M = A.M;
  C.N = A.N;
  C.A.resize((size_t)C.M * C.N);
  std::copy(A.A.begin(), A.A.end(), C.A.begin());

  const real_t* tau = &B.A[(size_t)B.M * B.N];
  LAPACKE_dormqr(LAPACK_COL_MAJOR, 'R', 'T', C.M, C.N, B.M, B.A.data(), B.M, tau, C.A.data(), C.M);
  cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, C.M, C.N, 1., B.A.data(), B.M, C.A.data(), C.M);

  int rank = 0;
  if (repi < 1.) {
    std::vector<real_t> C_((size_t)C.M * C.N);
    std::vector<real_t> S(std::max(C.M, C.N));
    std::vector<real_t> superb(std::max(C.M, C.N) - 1);
    std::copy(C.A.begin(), C.A.end(), C_.begin());

    LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'N', 'N', C.M, C.N, C_.data(), C.M, S.data(), nullptr, C.M, nullptr, C.N, superb.data());
    while(S[rank] / S[0] > repi)
      rank += 1;
  }
  else
    rank = (int)repi;
  return rank;
}

inline real_t rand(real_t min, real_t max) {
  return min + (max - min) * ((real_t)std::rand() / RAND_MAX);
}

void sample(const Matrix& A, Matrix& S) {
  std::vector<real_t> rnd((size_t)A.N * S.N);

  for (auto& i : rnd)
    i = rand(-1, 1);
    
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A.M, S.N, A.N, 1., A.A.data(), A.M, rnd.data(), A.N, 1., S.A.data(), S.M);
}

void nbd::orth_base(Matrix& Us, Matrix& Uc) {
  Uc.M = Us.M;
  Uc.N = Us.M - Us.N;
  Uc.A.resize((size_t)Uc.M * Uc.N);

  std::vector<real_t> work((size_t)Us.M * Us.M);
  std::vector<real_t> tau(Us.N);
  LAPACKE_dgeqrf(LAPACK_COL_MAJOR, Us.M, Us.N, Us.A.data(), Us.M, tau.data());
  std::copy(Us.A.begin(), Us.A.end(), work.begin());

  LAPACKE_dorgqr(LAPACK_COL_MAJOR, Us.M, Us.M, Us.N, work.data(), Us.M, tau.data());
  std::copy(work.begin(), work.begin() + Us.A.size(), Us.A.begin());
  std::copy(work.begin() + Us.A.size(), work.end(), Uc.A.begin());
}

Base nbd::base_i(real_t repi, real_t p, const Node& H) {
  Base base;
  base.Uo.resize(H.M);
  base.Uc.resize(H.M);

  Matrices diag_inv;
  diag_inv.resize(H.M);

#pragma omp parallel for
  for (int i = 0; i < H.M; i++) {
    Matrix& di = diag_inv[i];
    const Matrix& A_ii = H.A[i + (size_t)i * H.M];
    di.M = di.N = A_ii.M;
    di.A.resize(di.M + (size_t)di.M * di.N);
    std::copy(A_ii.A.begin(), A_ii.A.end(), di.A.begin());

    real_t* tau = &di.A[(size_t)di.M * di.N];
    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, di.M, di.N, di.A.data(), di.M, tau);
  }

#pragma omp parallel for
  for (int y = 0; y < H.M; y++) {
    int rank = 0;
    std::vector<Matrix> C;
    for (int x = 0; x < H.N; x++) {
      const Matrix& Axy = H.A[y + (size_t)x * H.M];
      if(x != y && Axy.M * Axy.N > 0) {
        C.emplace_back();
        int r = a_inv_b(repi, Axy, diag_inv[x], C.back());
        rank = std::max(r, rank);
      }
    }

    int ym = H.A[y + (size_t)y * H.M].M;
    rank = std::min((int)(rank * (1. + p)), ym);
    Matrix& uy = base.Uo[y];
    uy.M = ym;
    uy.N = rank;
    uy.A.resize((size_t)uy.M * uy.N);
    std::fill(uy.A.begin(), uy.A.end(), 0.);

    for (auto& Ci : C)
      sample(Ci, uy);

    orth_base(uy, base.Uc[y]);
  }

  return base;
}

void nbd::utav(const Matrix& U, const Matrix& A, const Matrix& VT, Matrix& C) {
  C.M = U.N;
  C.N = VT.N;
  C.A.resize((size_t)C.M * C.N);

  std::vector<real_t> work((size_t)U.N * A.N);
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, U.N, A.N, A.M, 1., U.A.data(), U.M, A.A.data(), A.M, 0., work.data(), U.N);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, C.M, C.N, A.N, 1., work.data(), U.N, VT.A.data(), VT.M, 0., C.A.data(), C.M);
}

void nbd::split_A(Node& H, const Base& U, const Base& V) {
  H.A_cc.resize((size_t)H.M * H.N);
  H.A_co.resize((size_t)H.M * H.N);
  H.A_oc.resize((size_t)H.M * H.N);
  H.A_oo.resize((size_t)H.M * H.N);

#pragma omp parallel for
  for (int i = 0; i < H.M * H.N; i++) {
    int x = i / H.M, y = i - x * H.M;
    const Matrix& Axy = H.A[i];
    if(Axy.M * Axy.N > 0) {
      utav(U.Uc[y], Axy, V.Uc[x], H.A_cc[i]);
      utav(U.Uc[y], Axy, V.Uo[x], H.A_co[i]);
      utav(U.Uo[y], Axy, V.Uc[x], H.A_oc[i]);
      utav(U.Uo[y], Axy, V.Uo[x], H.A_oo[i]);
    }
  }
}

void dgetrfnp(int m, int n, real_t* a, int lda) {
  int k = std::min(m, n);
  for (int i = 0; i < k; i++) {
    real_t p = 1. / a[i + (size_t)i * lda];
    int mi = m - i - 1;
    int ni = n - i - 1;

    real_t* ax = a + i + (size_t)i * lda + 1;
    real_t* ay = a + i + (size_t)i * lda + lda;
    real_t* an = ay + 1;

    cblas_dscal(mi, p, ax, 1);
    cblas_dger(CblasColMajor, mi, ni, -1., ax, 1, ay, lda, an, lda);
  }
}

void schur_complement(const Matrix& A, const Matrix& B, Matrix& C) {
  if (A.M * A.N * B.M * B.N > 0) {
    if (C.M * C.N == 0) {
      C.M = A.M;
      C.N = B.N;
      C.A.resize((size_t)C.M * C.N);
      std::fill(C.A.begin(), C.A.end(), 0.);
    }

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, C.M, C.N, A.N, -1., A.A.data(), A.M, B.A.data(), B.M, 1., C.A.data(), C.M);
  }
}

void nbd::factor_A(Node& H) {
  
#pragma omp parallel for
  for (int i = 0; i < H.M; i++) {
    Matrix& A_ii = H.A_cc[i + (size_t)i * H.M];
    dgetrfnp(A_ii.M, A_ii.N, A_ii.A.data(), A_ii.M);

    for (int j = i + 1; j < H.M; j++) {
      Matrix& A_ji = H.A_cc[j + (size_t)i * H.M];
      if (A_ji.M * A_ji.N > 0)
        cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, A_ji.M, A_ji.N, 1., A_ii.A.data(), A_ii.M, A_ji.A.data(), A_ji.M);
    }

    for (int j = i + 1; j < H.N; j++) {
      Matrix& A_ij = H.A_cc[i + (size_t)j * H.M];
      if (A_ij.M * A_ij.N > 0)
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, A_ij.M, A_ij.N, 1., A_ii.A.data(), A_ii.M, A_ij.A.data(), A_ij.M);
    }

    for (int j = 0; j < H.M; j++) {
      Matrix& A_ji = H.A_oc[j + (size_t)i * H.M];
      if (A_ji.M * A_ji.N > 0)
        cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, A_ji.M, A_ji.N, 1., A_ii.A.data(), A_ii.M, A_ji.A.data(), A_ji.M);
    }

    for (int j = 0; j < H.N; j++) {
      Matrix& A_ij = H.A_co[i + (size_t)j * H.M];
      if (A_ij.M * A_ij.N > 0)
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, A_ij.M, A_ij.N, 1., A_ii.A.data(), A_ii.M, A_ij.A.data(), A_ij.M);
    }
  }

#pragma omp parallel for
  for (int i = 0; i < H.M * H.N; i++) {
    int x = i / H.M, y = i - x * H.M;
    Matrix& Axy = H.A_oo[i];
    for (int k = 0; k < H.M; k++)
      schur_complement(H.A_oc[y + (size_t)k * H.M], H.A_co[k + (size_t)x * H.M], Axy);
  }
}

std::vector<real_t*> nbd::base_fw(const Base& U, real_t* x) {
  int n = U.Uc.size(), lx = 0, lc = 0, lo = 0;
  std::vector<int> off_x(n + 1);
  std::vector<int> off_c(n + 1);
  std::vector<int> off_o(n + 1);

  off_x[0] = off_c[0] = off_o[0] = 0;

  for (int i = 0; i < n; i++) {
    lx += U.Uc[i].M;
    lc += U.Uc[i].N;
    lo += U.Uo[i].N;
    off_x[i + 1] = lx;
    off_c[i + 1] = lc;
    off_o[i + 1] = lo;
  }

  std::vector<real_t> x_out(lx);
  std::vector<real_t*> i_out(n + n);

#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    const real_t* xi = x + off_x[i];
    real_t* ci = x_out.data() + off_c[i];
    real_t* oi = x_out.data() + lc + off_o[i];

    const Matrix& Uc = U.Uc[i];
    const Matrix& Uo = U.Uo[i];

    cblas_dgemv(CblasColMajor, CblasTrans, Uc.M, Uc.N, 1., Uc.A.data(), Uc.M, xi, 1, 0., ci, 1);
    cblas_dgemv(CblasColMajor, CblasTrans, Uo.M, Uo.N, 1., Uo.A.data(), Uo.M, xi, 1, 0., oi, 1);

    i_out[i] = x + off_c[i];
    i_out[i + n] = x + lc + off_o[i];
  }

  std::copy(x_out.begin(), x_out.end(), x);
  return i_out;
}

void nbd::base_bk(const Base& U, real_t* x) {
  int n = U.Uc.size(), lx = 0, lc = 0, lo = 0;
  std::vector<int> off_x(n + 1);
  std::vector<int> off_c(n + 1);
  std::vector<int> off_o(n + 1);

  off_x[0] = off_c[0] = off_o[0] = 0;

  for (int i = 0; i < n; i++) {
    lx += U.Uc[i].M;
    lc += U.Uc[i].N;
    lo += U.Uo[i].N;
    off_x[i + 1] = lx;
    off_c[i + 1] = lc;
    off_o[i + 1] = lo;
  }

  std::vector<real_t> x_out(lx);

#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    const real_t* ci = x + off_c[i];
    const real_t* oi = x + lc + off_o[i];
    real_t* xi = x_out.data() + off_x[i];

    const Matrix& Uc = U.Uc[i];
    const Matrix& Uo = U.Uo[i];

    cblas_dgemv(CblasColMajor, CblasNoTrans, Uc.M, Uc.N, 1., Uc.A.data(), Uc.M, ci, 1, 0., xi, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, Uo.M, Uo.N, 1., Uo.A.data(), Uo.M, oi, 1, 1., xi, 1);
  }

  std::copy(x_out.begin(), x_out.end(), x);
}

void nbd::A_fw(const Node& H, std::vector<real_t*>& x) {
  int n = H.M;
  for (int i = 0; i < n; i++) {
    const Matrix& A_ii = H.A_cc[i + (size_t)i * n];
    cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit, A_ii.M, A_ii.A.data(), A_ii.M, x[i], 1);

    for (int j = i + 1; j < n; j++) {
      const Matrix& A_ji = H.A_cc[j + (size_t)i * n];
      if (A_ji.M * A_ji.N > 0)
        cblas_dgemv(CblasColMajor, CblasNoTrans, A_ji.M, A_ji.N, -1., A_ji.A.data(), A_ji.M, x[i], 1, 1., x[j], 1);
    }

    for (int j = 0; j < n; j++) {
      const Matrix& A_ji = H.A_oc[j + (size_t)i * n];
      if (A_ji.M * A_ji.N > 0)
        cblas_dgemv(CblasColMajor, CblasNoTrans, A_ji.M, A_ji.N, -1., A_ji.A.data(), A_ji.M, x[i], 1, 1., x[j + n], 1);
    }
  }
}


void nbd::A_bk(const Node& H, std::vector<real_t*>& x) {
  int n = H.M;
  for (int i = n - 1; i >= 0; i--) {
    for (int j = i + 1; j < n; j++) {
      const Matrix& A_ij = H.A_cc[i + (size_t)j * n];
      if (A_ij.M * A_ij.N > 0)
        cblas_dgemv(CblasColMajor, CblasNoTrans, A_ij.M, A_ij.N, -1., A_ij.A.data(), A_ij.M, x[j], 1, 1., x[i], 1);
    }

    for (int j = 0; j < n; j++) {
      const Matrix& A_ij = H.A_co[i + (size_t)j * n];
      if (A_ij.M * A_ij.N > 0)
        cblas_dgemv(CblasColMajor, CblasNoTrans, A_ij.M, A_ij.N, -1., A_ij.A.data(), A_ij.M, x[j + n], 1, 1., x[i], 1);
    }

    const Matrix& A_ii = H.A_cc[i + (size_t)i * n];
    cblas_dtrsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, A_ii.M, A_ii.A.data(), A_ii.M, x[i], 1);
  }
}

void merge4(int m0, int m1, int n0, int n1, const Matrix& A00, const Matrix& A10, const Matrix& A01, const Matrix& A11, Matrix& C) {
  if (A00.M * A00.N > 0 || A10.M * A10.N > 0 || A01.M * A01.N > 0 || A11.M * A11.N > 0) {
    C.M = m0 + m1;
    C.N = n0 + n1;
    C.A.resize((size_t)C.M * C.N);
    std::fill(C.A.begin(), C.A.end(), 0);

    if (A00.M * A00.N > 0)
      for (int i = 0; i < A00.M * A00.N; i++) {
        int x = i / A00.M, y = i - x * A00.M;
        C.A[y + (size_t)x * C.M] = A00.A[y + (size_t)x * A00.M];
      }

    if (A10.M * A10.N > 0)
      for (int i = 0; i < A10.M * A10.N; i++) {
        int x = i / A10.M, y = i - x * A10.M;
        C.A[m0 + y + (size_t)x * C.M] = A10.A[y + (size_t)x * A10.M];
      }

    if (A01.M * A01.N > 0)
      for (int i = 0; i < A01.M * A01.N; i++) {
        int x = i / A01.M, y = i - x * A01.M;
        C.A[y + ((size_t)x + n0) * C.M] = A01.A[y + (size_t)x * A01.M];
      }
    
    if (A11.M * A11.N > 0)
      for (int i = 0; i < A11.M * A11.N; i++) {
        int x = i / A11.M, y = i - x * A11.M;
        C.A[m0 + y + ((size_t)x + n0) * C.M] = A11.A[y + (size_t)x * A11.M];
      }
  }
}

Node nbd::merge_H(const Node& H) {
  Node n;
  n.M = H.M / 2;
  n.N = H.N / 2;
  n.A.resize((size_t)n.M * n.N);

  std::vector<int> len(H.M);
  for (int i = 0; i < H.M; i++)
    len[i] = H.A_oo[i + (size_t)i * H.M].M;

  if(n.M * n.N > 0)
#pragma omp parallel for
    for (int i = 0; i < n.M * n.N; i++) {
      int x = i / n.M, y = i - x * n.M;
      merge4(len[y*2], len[y*2+1], len[x*2], len[x*2+1], H.A_oo[y*2 + (size_t)x*2 * H.M], 
        H.A_oo[y*2+1 + (size_t)(x*2) * H.M], 
        H.A_oo[y*2 + (size_t)(x*2+1) * H.M], 
        H.A_oo[y*2+1 + (size_t)(x*2+1) * H.M], n.A[y + (size_t)x * n.M]);
    }
  
  return n;
}


Matrix nbd::merge_D(const Node& H) {
  int n = H.M, l = 0;
  std::vector<int> off(n + 1);
  off[0] = 0;

  for (int i = 0; i < n; i++) {
    l += H.A_oo[i + (size_t)i * n].M;
    off[i + 1] = l;
  }

  Matrix D;
  D.M = D.N = l;
  D.A.resize((size_t)l * l);
  std::fill(D.A.begin(), D.A.end(), 0.);

#pragma omp parallel for
  for (int i = 0; i < n * n; i++) {
    int x = i / n, y = i - x * n;
    const Matrix& Axy = H.A_oo[i];
    real_t* dxy = D.A.data() + off[y] + (size_t)off[x] * l;
    for (int xx = 0; xx < Axy.N; xx++)
      for (int yy = 0; yy < Axy.M; yy++)
        dxy[yy + (size_t)xx * l] = Axy.A[yy + (size_t)xx * Axy.M];
  }

  return D;
}


void nbd::solve_D(Matrix& D, real_t* x) {
  std::vector<int> ipiv(D.M);
  LAPACKE_dgesv(LAPACK_COL_MAJOR, D.M, 1, D.A.data(), D.M, ipiv.data(), x, D.M);
}

void nbd::h2_solve_complete(real_t repi, real_t p, Node& H, real_t* x) {
  Base bi = base_i(repi, p, H);
  split_A(H, bi, bi);
  factor_A(H);

  auto xi = base_fw(bi, x);
  A_fw(H, xi);

  if (H.M > 2) {
    Node H2 = merge_H(H);
    h2_solve_complete(repi, p, H2, xi[H.M]);
  }
  else {
    Matrix last = merge_D(H);
    solve_D(last, xi[H.M]);
  }

  A_bk(H, xi);
  base_bk(bi, x);
}
