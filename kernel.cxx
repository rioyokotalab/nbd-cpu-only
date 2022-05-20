
#include "kernel.hxx"
#include "build_tree.hxx"

#include <cmath>
#include <random>

using namespace nbd;

EvalFunc nbd::l3d() {
  EvalFunc ef;
  ef.r2f = [](double& r2, double singularity, double alpha) -> void {
    r2 = 1. / (std::sqrt(r2) + singularity);
  };
  ef.singularity = 1.e-6;
  ef.alpha = 1.;
  return ef;
}

EvalFunc nbd::yukawa3d() {
  EvalFunc ef;
  ef.r2f = [](double& r2, double singularity, double alpha) -> void {
    double r = std::sqrt(r2);
    r2 = std::exp(-alpha * r) / (r + singularity);
  };
  ef.singularity = 1.e-6;
  ef.alpha = 1.;
  return ef;
}

void nbd::eval(EvalFunc ef, const Body* bi, const Body* bj, int64_t dim, double* out) {
  double& r2 = *out;
  r2 = 0.;
  for (int64_t i = 0; i < dim; i++) {
    double dX = bi->X[i] - bj->X[i];
    r2 += dX * dX;
  }
  ef.r2f(r2, ef.singularity, ef.alpha);
}


void nbd::P2P(EvalFunc ef, const Cell* ci, const Cell* cj, int64_t dim, const Vector& X, Vector& B) {
  int64_t m = ci->NBODY;
  int64_t n = cj->NBODY;
  const double* x = &X.X[0];
  double* b = &B.X[0];

  for (int64_t i = 0; i < m; i++) {
    double sum = b[i];
    for (int64_t j = 0; j < n; j++) {
      double r2;
      eval(ef, ci->BODY + i, cj->BODY + j, dim, &r2);
      sum += r2 * x[j];
    }
    b[i] = sum;
  }
}


void nbd::P2Pmat(EvalFunc ef, const Cell* ci, const Cell* cj, int64_t dim, Matrix& a) {
  int64_t m = ci->NBODY, n = cj->NBODY;
  a.A.resize(m * n);
  a.M = m;
  a.N = n;

  for (int64_t i = 0; i < m * n; i++) {
    int64_t x = i / m;
    int64_t y = i - x * m;
    double r2;
    eval(ef, ci->BODY + y, cj->BODY + x, dim, &r2);
    a.A[x * a.M + y] = r2;
  }
}

void nbd::M2L(EvalFunc ef, const Cell* ci, const Cell* cj, int64_t dim, const double m[], double l[]) {
  const std::vector<int64_t>& mi = ci->Multipole;
  const std::vector<int64_t>& mj = cj->Multipole;
  int64_t y = mi.size();
  int64_t x = mj.size();

  for (int64_t i = 0; i < y; i++) {
    int64_t bi = mi[i];
    double sum = l[i];
    for (int64_t j = 0; j < x; j++) {
      double r2;
      int64_t bj = mj[j];
      eval(ef, ci->BODY + bi, cj->BODY + bj, dim, &r2);
      sum += r2 * m[j];
    }
    l[i] = sum;
  }
}

void nbd::M2Lc(EvalFunc ef, const Cell* ci, const Cell* cj, int64_t dim, const Vector& M, Vector& L) {
  int64_t off_m = 0;
  for (int64_t j = 0; j < cj->NCHILD; j++) {
    const Cell* ccj = cj->CHILD + j;
    int64_t off_l = 0;
    for (int64_t i = 0; i < ci->NCHILD; i++) {
      const Cell* cci = ci->CHILD + i;
      int64_t len_far = cci->listFar.size();
      for (int64_t k = 0; k < len_far; k++)
        if (cci->listFar[k] == ccj)
          M2L(ef, cci, ccj, dim, &M.X[off_m], &L.X[off_l]);

      off_l = off_l + cci->Multipole.size();
    }
    off_m = off_m + ccj->Multipole.size();
  }
}


void nbd::M2Lmat_bodies(EvalFunc ef, int64_t m, int64_t n, const int64_t mi[], const int64_t mj[], const Body* bi, const Body* bj, int64_t dim, Matrix& a) {
  for (int64_t i = 0; i < m * n; i++) {
    int64_t x = i / m;
    int64_t bx = mj == nullptr ? x : mj[x];
    int64_t y = i - x * m;
    int64_t by = mi == nullptr ? y : mi[y];

    double r2;
    eval(ef, bi + by, bj + bx, dim, &r2);
    a.A[x * a.M + y] = r2;
  }
}

void nbd::M2Lmat(EvalFunc ef, const Cell* ci, const Cell* cj, int64_t dim, Matrix& a) {
  M2Lmat_bodies(ef, ci->Multipole.size(), cj->Multipole.size(), &ci->Multipole[0], &cj->Multipole[0], ci->BODY, cj->BODY, dim, a);
}


