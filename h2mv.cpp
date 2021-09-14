
#include "h2mv.h"
#include "kernel.h"

#include <cblas.h>

using namespace nbd;

std::vector<int> nbd::dfs(const Cell* cell, bool pre_order, int i) {
  if (cell->CHILD == nullptr)
    return std::vector<int> (1, i);
  else {
    std::vector<int> order;
    if (pre_order)
      order.emplace_back(i);
    for (Cell* c = cell->CHILD; c != cell->CHILD + cell->NCHILD; c++) {
      auto ci = (c - cell) + i;
      auto oi = dfs(c, pre_order, ci);
      order.insert(order.end(), oi.begin(), oi.end());
    }
    if (!pre_order)
      order.emplace_back(i);

    return order;
  }
}

struct UpwardMap {
  const Matrix* base;
  const real_t* x;
  real_t* m;
};


void nbd::upwardPass(const Cells& jcells, const Matrices& jbase, const real_t* x, real_t* m) {
  std::vector<UpwardMap> map(jcells.size());
  int m_off = 0;
  for (int i = 0; i < jcells.size(); i++) {
    map[i].base = &jbase[i];
    map[i].m = m + m_off;
    m_off += jbase[i].N;
  }

  auto j_begin = jcells[0].BODY;
  for (int i = 0; i < jcells.size(); i++) {
    if (jcells[i].NCHILD == 0) {
      auto x_off = jcells[i].BODY - j_begin;
      map[i].x = x + x_off;
    }
    else {
      auto c = jcells[i].CHILD - &jcells[0];
      map[i].x = m + (map[c].m - m);
    }
  }

  std::vector<int> order = dfs(&jcells[0], false);

  for (int i = 0; i < jcells.size(); i++) {
    int ii = order[i];
    const Matrix* base = map[ii].base;
    const real_t* x = map[ii].x;
    real_t* m = map[ii].m;
    if (base->N > 0)
      cblas_dgemv(CblasColMajor, CblasTrans, base->M, base->N, 1., &(base->A)[0], base->LDA, x, 1, 0., m, 1);
  }
}

void nbd::upwardPassOne(const Matrices& jbase, const real_t* x, real_t* m) {
  std::vector<UpwardMap> map(jbase.size());
  int m_off = 0;
  int x_off = 0;
  for (int i = 0; i < jbase.size(); i++) {
    map[i].base = &jbase[i];
    map[i].m = m + m_off;
    m_off += jbase[i].N;
    map[i].x = x + x_off;
    x_off += jbase[i].M;
  }

  for (int i = 0; i < jbase.size(); i++) {
    const Matrix* base = map[i].base;
    const real_t* x = map[i].x;
    real_t* m = map[i].m;
    if (base->N > 0)
      cblas_dgemv(CblasColMajor, CblasTrans, base->M, base->N, 1., &(base->A)[0], base->LDA, x, 1, 0., m, 1);
  }
}


void nbd::horizontalPass(const Cells& icells, const Cells& jcells, const Matrices& ibase, const Matrices& jbase, const Matrices& d, const real_t* m, real_t* l) {
  std::vector<const real_t*> mo(jcells.size());
  std::vector<real_t*> lo(icells.size());

  int m_off = 0;
  for (int x = 0; x < jcells.size(); x++) {
    mo[x] = m + m_off;
    m_off += jbase[x].N;
  }

  int l_off = 0;
  for (int y = 0; y < icells.size(); y++) {
    lo[y] = l + l_off;
    l_off += ibase[y].N;
  }

  int ld = (int)icells.size();
#pragma omp parallel for
  for (int y = 0; y < icells.size(); y++) {
    auto i = icells[y];
    for (auto& j : i.listFar) {
      auto x = j - &jcells[0];
      const Matrix& s = d[y + x * ld];
      cblas_dgemv(CblasColMajor, CblasNoTrans, s.M, s.N, 1., &s.A[0], s.LDA, mo[x], 1, 1., lo[y], 1);
    }
  }
}

struct DownwardMap {
  const Matrix* base;
  real_t* l;
  real_t* b;
};

void nbd::downwardPass(const Cells& icells, const Matrices& ibase, real_t* l, real_t* b) {

  std::vector<DownwardMap> map(icells.size());
  int l_off = 0;
  for (int i = 0; i < icells.size(); i++) {
    map[i].base = &ibase[i];
    map[i].l = l + l_off;
    l_off += ibase[i].N;
  }

  auto i_begin = icells[0].BODY;
  for (int i = 0; i < icells.size(); i++) {
    if (icells[i].NCHILD == 0) {
      auto b_off = icells[i].BODY - i_begin;
      map[i].b = b + b_off;
    }
    else {
      auto c = icells[i].CHILD - &icells[0];
      map[i].b = l + (map[c].l - l);
    }
  }

  std::vector<int> order = dfs(&icells[0], true);

  for (int i = 0; i < icells.size(); i++) {
    int ii = order[i];
    const Matrix* base = map[ii].base;
    real_t* l = map[ii].l;
    real_t* b = map[ii].b;
    if (base->N > 0)
      cblas_dgemv(CblasColMajor, CblasNoTrans, base->M, base->N, 1., &(base->A)[0], base->LDA, l, 1, 1., b, 1);
  }

}

void downwardPassOne(const Matrices& ibase, real_t* l, real_t* b) {
  std::vector<DownwardMap> map(ibase.size());
  int l_off = 0;
  int b_off = 0;
  for (int i = 0; i < ibase.size(); i++) {
    map[i].base = &ibase[i];
    map[i].l = l + l_off;
    l_off += ibase[i].N;
    map[i].b = b + b_off;
    b_off += ibase[i].M;
  }

  for (int i = 0; i < ibase.size(); i++) {
    const Matrix* base = map[i].base;
    real_t* l = map[i].l;
    real_t* b = map[i].b;
    if (base->N > 0)
      cblas_dgemv(CblasColMajor, CblasNoTrans, base->M, base->N, 1., &(base->A)[0], base->LDA, l, 1, 1., b, 1);
  }
}

void nbd::closeQuarter(EvalFunc ef, const Cells& icells, const Cells& jcells, int dim, const real_t* x, real_t* b) {
  auto j_begin = jcells[0].BODY;
  auto i_begin = icells[0].BODY;

#pragma omp parallel for
  for (int y = 0; y < icells.size(); y++) {
    auto i = icells[y];
    auto yi = i.BODY - i_begin;
    for (auto& j : i.listNear) {
      auto _x = j - &jcells[0];
      auto xi = j->BODY - j_begin;
      mvec_kernel(ef, &icells[y], &jcells[_x], dim, 1., x + xi, 1, 1, b + yi, 1);
    }
  }
}


void nbd::h2mv_complete(EvalFunc ef, const Cells& icells, const Cells& jcells, int dim, const Matrices& ibase, const Matrices& jbase, const Matrices& d, const real_t* x, real_t* b) {

  int lm = 0;
  for (auto& c : jbase)
    lm += c.N;
  
  int ll = 0;
  for (auto& c : ibase)
    ll += c.N;
  
  std::vector<real_t> m(lm), l(ll);
  std::fill(b, b + icells[0].NBODY, 0.);
  std::fill(m.begin(), m.end(), 0.);
  std::fill(l.begin(), l.end(), 0.);

  upwardPass(jcells, jbase, x, &m[0]);
  horizontalPass(icells, jcells, ibase, jbase, d, &m[0], &l[0]);
  downwardPass(icells, ibase, &l[0], b);
  closeQuarter(ef, icells, jcells, dim, x, b);
}



