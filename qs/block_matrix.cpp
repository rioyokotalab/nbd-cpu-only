
#include "block_matrix.h"

#include "../kernel.h"
#include "../build_tree.h"
#include "factorize.h"
#include "solve.h"
#include <cstddef>
#include <algorithm>
#include <numeric>

using namespace qs;

void qs::matcpy(const nbd::Matrix& m1, Matrix& m2) {
  if(m1.M > 0 && m1.N > 0) {
    m2.M = m1.M;
    m2.N = m1.N;
    m2.A = m1.A;
    m2.LDA = m1.LDA;
  }
}

Matrices qs::convert(const nbd::Matrices& d) {
  Matrices ms(d.size());

  for (int i = 0; i < d.size(); i++) {
    matcpy(d[i], ms[i]);
    ms[i].LMO = ms[i].LNO = 0;
  }

  return ms;
}

H2Matrix qs::build(nbd::eval_func_t r2f, int dim, const nbd::Cells& cells, const nbd::Matrices& d) {
  H2Matrix b { (int)cells.size() };
  b.D = convert(d);

#pragma omp parallel for
  for (int y = 0; y < cells.size(); y++) {
    auto i = cells[y];
    for (auto& j : i.listNear) {
      auto _x = j - &cells[0];
      nbd::Matrix m;
      nbd::P2Pnear(r2f, &cells[y], &cells[_x], dim, m);
      matcpy(m, b.D[y + _x * b.N]);
    }
  }

  return b;

}

ElimOrder qs::order(const nbd::Cells& cells, const nbd::Matrices& base) {
  
  std::vector<int> levels(cells.size());
  int max_l = nbd::lvls(&cells[0], &levels[0]);
  
  ElimOrder eo(max_l + 1);
  for (int i = 0; i < cells.size(); i++) {
    int li = levels[i];
    eo[li].IND.emplace_back(i);
    eo[li].LI.emplace_back(base[i].M);
    eo[li].LIO.emplace_back(base[i].N);

    if(cells[i].NCHILD > 0) {
      eo[li].CHILD_IND.emplace_back(cells[i].CHILD - &cells[0]);
      eo[li].NCHILD.emplace_back(cells[i].NCHILD);
    }
    else {
      eo[li].CHILD_IND.emplace_back(0);
      eo[li].NCHILD.emplace_back(0);
    }
  }

  return eo;

}

void qs::elim(const Level& lvl, H2Matrix& h2, Matrices& base) {
  
  for (int i : lvl.IND) {
    Matrix R;
    eux(base[i], R);
    plu(base[i], h2.D[i + (size_t)i * h2.N]);

    for (int j : lvl.IND)
      if (j != i) {
        ptrsmr(base[i], h2.D[i + (size_t)i * h2.N], h2.D[i + (size_t)j * h2.N]);
        mulrleft(R, h2.D[i + (size_t)j * h2.N]);
      }

    for (int k : lvl.IND)
      if (k != i) {
        ptrsmc(base[i], h2.D[i + (size_t)i * h2.N], h2.D[k + (size_t)i * h2.N]);
        mulrright(R, h2.D[k + (size_t)i * h2.N]);
      }

    for (int k : lvl.IND)
      if (k != i)
        for (int j : lvl.IND)
          if (j != i)
            pgemm(h2.D[k + (size_t)i * h2.N], h2.D[i + (size_t)j * h2.N], h2.D[k + (size_t)j * h2.N]);
  }

}

void matcpy2d(int m, int n, const double* A, int lda, double* B, int ldb) {
  for (int j = 0; j < n; j++)
    for (int i = 0; i < m; i++) {
      B[i + (size_t)j * ldb] = A[i + (size_t)j * lda];
    }
}

void qs::merge(const Matrix* child, int ldc, int m, int n, Matrix& d) {

  for (int j = 0; j < n; j++)
    for (int i = 0; i < n; i++)
      if (child[i + (size_t)j * ldc].M == 0 && child[i + (size_t)j * ldc].N == 0)
        return;
  
  std::vector<int> offsets_x(n + 1, 0);
  for (int i = 1; i <= n; i++)
    offsets_x[i] = offsets_x[i - 1] + child[(size_t)(i - 1) * ldc].LNO;

  std::vector<int> offsets_y(m + 1, 0);
  for (int i = 1; i <= m; i++)
    offsets_y[i] = offsets_y[i - 1] + child[i - 1].LMO;

  d.M = offsets_y.back();
  d.N = offsets_x.back();
  d.A.resize((size_t)d.M * d.N);
  d.LDA = d.M;

  for (int j = 0; j < n; j++)
    for (int i = 0; i < n; i++) {
      double* d_ij = d.A.data() + offsets_y[i] + (size_t)offsets_x[j] * d.LDA;
      int LMO = child[i + (size_t)j * ldc].LMO;
      int LNO = child[i + (size_t)j * ldc].LNO;
      const double* c = child[i + (size_t)j * ldc].A.data();
      
      matcpy2d(LMO, LNO, c, child[i + (size_t)j * ldc].LDA, d_ij, d.LDA);
    }
}

void qs::pnm(const Level& lvl, H2Matrix& h2) {
  for (int i = 0; i < lvl.IND.size(); i++)
    for (int j = 0; j < lvl.IND.size(); j++) {
      int ci = lvl.CHILD_IND[i];
      int cj = lvl.CHILD_IND[j];
      int li = lvl.IND[i];
      int lj = lvl.IND[j];
      merge(&h2.D[ci + (size_t)cj * h2.N], h2.N, lvl.NCHILD[i], lvl.NCHILD[j], h2.D[li + (size_t)lj * h2.N]);
    }
}


void qs::fwd_solution(const Level& lvl, const H2Matrix& h2, const Matrices& base, double* X) {

  std::vector<int> offsets(lvl.IND.size() + 1, 0);
  for (int i = 1; i <= offsets.size(); i++)
    offsets[i] = offsets[i - 1] + lvl.LI[i - 1];

  for (int i = 0; i < lvl.IND.size(); i++) {
    int li = lvl.IND[i];
    uxmv('T', base[li], X + offsets[i]);
    fwsvcc(h2.D[li + (size_t)li * h2.N], X + offsets[i]);
  }

  std::vector<double> p(offsets.back());
  auto iter = p.begin();
  for (int i = 0; i < lvl.IND.size(); i++) {
    std::copy(X + offsets[i], X + offsets[i] + lvl.LIO[i], iter);
    iter = iter + lvl.LIO[i];
  }
  for (int i = 0; i < lvl.IND.size(); i++) {
    std::copy(X + offsets[i] + lvl.LIO[i], X + offsets[i + 1], iter);
    iter = iter + lvl.LI[i] - lvl.LIO[i];
  }

  std::copy(p.begin(), p.end(), X);

}


void qs::bkwd_solution(const Level& lvl, const H2Matrix& h2, const Matrices& base, double* X) {

  std::vector<int> offsets(lvl.IND.size() + 1, 0);
  for (int i = 1; i <= lvl.IND.size(); i++)
    offsets[i] = offsets[i - 1] + lvl.LI[i - 1];

  std::vector<double> p(offsets.back());
  const double* iter = X;
  for (int i = 0; i < lvl.IND.size(); i++) {
    std::copy(iter, iter + lvl.LIO[i], p.begin() + offsets[i]);
    iter = iter + lvl.LIO[i];
  }
  for (int i = 0; i < lvl.IND.size(); i++) {
    std::copy(iter, iter + (lvl.LI[i] - lvl.LIO[i]), p.begin() + offsets[i] + lvl.LIO[i]);
    iter = iter + (lvl.LI[i] - lvl.LIO[i]);
  }

  std::copy(p.begin(), p.end(), X);

  for (int i = lvl.IND.size() - 1; i >= 0; i--) {
    int li = lvl.IND[i];
    bksvcc(h2.D[li + (size_t)li * h2.N], X + offsets[i]);
    uxmv('N', base[li], X + offsets[i]);
  }
}
