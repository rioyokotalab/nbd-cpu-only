
#include "block_matrix.h"

#include "../kernel.h"
#include "../build_tree.h"
#include "factorize.h"
#include <cstddef>

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

ElimOrder qs::order(const nbd::Cells& cells) {
  
  std::vector<int> levels(cells.size());
  int max_l = nbd::lvls(&cells[0], &levels[0]);
  
  ElimOrder eo(max_l + 1);
  for (int i = 0; i < cells.size(); i++) {
    int li = levels[i];
    eo[li].IND.emplace_back(i);
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
    eux(base[i]);
    plu(base[i], h2.D[i + (size_t)i * h2.N]);

    for (int j : lvl.IND)
      if (j != i)
        ptrsmr(base[i], h2.D[i + (size_t)i * h2.N], h2.D[i + (size_t)j * h2.N]);

    for (int k : lvl.IND)
      if (k != i)
        ptrsmc(base[i], h2.D[i + (size_t)i * h2.N], h2.D[k + (size_t)i * h2.N]);

    for (int k : lvl.IND)
      if (k != i)
        for (int j : lvl.IND)
          if (j != i)
            pgemm(h2.D[k + (size_t)i * h2.N], h2.D[i + (size_t)j * h2.N], h2.D[k + (size_t)j * h2.N]);
  }

}