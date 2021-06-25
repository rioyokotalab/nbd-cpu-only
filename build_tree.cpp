
#include "build_tree.h"
#include "kernel.h"

#include <cmath>
#include <iterator>
#include <random>
#include <cstdio>

using namespace nbd;

Bodies::iterator spart(Bodies::iterator first, Bodies::iterator last, int sdim, real_t pivot) {

  auto l = [pivot, sdim](const Body& element) { return element.X[sdim] < pivot; };
  return std::partition(first, last, l);
}

Bodies::iterator spart_size_k(Bodies::iterator first, Bodies::iterator last, int sdim, int k) {

  if (last == first)
    return first;
  real_t pivot = std::next(first, std::distance(first, last) / 2)->X[sdim];

  Bodies::iterator i1 = std::partition(first, last,
    [pivot, sdim](const Body& element) { return element.X[sdim] < pivot; });
  Bodies::iterator i2 = std::partition(i1, last,
    [pivot, sdim](const Body& element) { return !(pivot < element.X[sdim]); });
  auto d1 = i1 - first, d2 = i2 - first;
  
  if (d1 > k)
    return spart_size_k(first, i1, sdim, k);
  else if (d2 < k)
    return spart_size_k(i2, last, sdim, (int)(k - d2));
  else
    return first + k;
}

Cells nbd::buildTree(Bodies& bodies, int ncrit, int dim) {

  Cells cells(1);
  cells.reserve(bodies.size());

  cells[0].BODY = bodies.data();
  cells[0].NBODY = (int)bodies.size();
  cells[0].NCHILD = 0;
  for (int d = 0; d < dim; d++) 
    cells[0].Xmin[d] = cells[0].Xmax[d] = bodies[0].X[d];
  for (int b = 0; b < (int)bodies.size(); b++) {
    for (int d = 0; d < dim; d++) 
      cells[0].Xmin[d] = std::fmin(bodies[b].X[d], cells[0].Xmin[d]);
    for (int d = 0; d < dim; d++) 
      cells[0].Xmax[d] = std::fmax(bodies[b].X[d], cells[0].Xmax[d]);
  }

  int nlis = ((int)bodies.size() + ncrit - 1) / ncrit, iters = 0;
  int sdim = 0;
  while (nlis >>= 1) ++iters;
  int last_off = 0, last_len = 1;

  for (int i = 1; i <= iters; i++) {

    int len = 0;

    for (int j = last_off; j < last_off + last_len; j++) {
      Cell& cell = cells[j];
      Bodies::iterator cell_b = bodies.begin() + std::distance(bodies.data(), cell.BODY);

#ifdef PART_EQ_SIZE
      auto p = spart_size_k(cell_b, cell_b + cell.NBODY, sdim, cell.NBODY / 2);
      real_t med = p->X[sdim];
#else
      real_t med = (cell.Xmin[sdim] + cell.Xmax[sdim]) / 2;
      auto p = spart(cell_b, cell_b + cell.NBODY, sdim, med);
#endif
      int size[2];
      size[0] = (int)(p - cell_b);
      size[1] = cell.NBODY - size[0];

      cell.NCHILD = (int)(size[0] > 0) + (int)(size[1] > 0);
      cells.resize(cells.size() + cell.NCHILD);
      Cell* child = &cells.back() - cell.NCHILD + 1;
      cell.CHILD = child;

      int offset = 0;
      for (int k = 0; k < 2; k++) {
        if (size[k]) {
          child->BODY = cell.BODY + offset;
          child->NBODY = size[k];
          child->NCHILD = 0;
          for (int d = 0; d < dim; d++) {
            child->Xmin[d] = cell.Xmin[d];
            child->Xmax[d] = cell.Xmax[d];
          }
          if ((k & 1) > 0)
            child->Xmin[sdim] = med;
          else
            child->Xmax[sdim] = med;
          child = child + 1;
          offset += size[k];
        }
      }

      len += cell.NCHILD;
    }

    last_off += last_len;
    last_len = len;
    sdim = sdim == dim - 1 ? 0 : sdim + 1;
  }
  return cells;
}


void nbd::getList(Cell * Ci, Cell * Cj, int dim, real_t theta) {
  real_t dX = 0., CiR = 0., CjR = 0.;
  for (int d = 0; d < dim; d++) {
    real_t CiC = (Ci->Xmin[d] + Ci->Xmax[d]) / 2;
    real_t CjC = (Cj->Xmin[d] + Cj->Xmax[d]) / 2;
    real_t diff = CiC - CjC;
    dX += diff * diff;

    CiR = std::max(CiR, CiC - Ci->Xmin[d]);
    CiR = std::max(CiR, - CiC + Ci->Xmax[d]);
    CjR = std::max(CjR, CjC - Cj->Xmin[d]);
    CjR = std::max(CjR, - CjC + Cj->Xmax[d]);
  }
  real_t R2 = dX * theta * theta;

  if (R2 > (CiR + CjR) * (CiR + CjR)) {
    Ci->listFar.push_back(Cj);
  } else if (Ci->NCHILD == 0 && Cj->NCHILD == 0) {
    Ci->listNear.push_back(Cj);
  } else if (Cj->NCHILD == 0 || (CiR >= CjR && Ci->NCHILD != 0)) {
    for (Cell * ci=Ci->CHILD; ci!=Ci->CHILD+Ci->NCHILD; ci++) {
      getList(ci, Cj, dim, theta);
    }
  } else {
    for (Cell * cj=Cj->CHILD; cj!=Cj->CHILD+Cj->NCHILD; cj++) {
      getList(Ci, cj, dim, theta);
    }
  }
}

void nbd::evaluate(eval_func_t r2f, Cells& cells, const Cell* jcell_start, int dim, Matrices& d, int rank) {
  for (auto& i : cells) {
    auto y = &i - cells.data();
    for (auto& j : i.listFar) {
      auto x = j - jcell_start;
      P2Pfar(r2f, &i, j, dim, d[y + x * cells.size()], rank);
    }
    for (auto& j : i.listNear) {
      auto x = j - jcell_start;
      P2Pnear(r2f, &i, j, dim, d[y + x * cells.size()]);
    }
  }
}

void nbd::traverse(eval_func_t r2f, Cells& icells, Cells& jcells, int dim, Matrices& d, real_t theta, int rank) {
  getList(&icells[0], &jcells[0], dim, theta);
  d.resize(icells.size() * jcells.size());
  evaluate(r2f, icells, &jcells[0], dim, d, rank);
}


void nbd::sample_base_i(Cells& icells, Cells& jcells, Matrices& d, int ld, Matrices& base, int p) {
  for (auto& i : icells) {
    auto y = &i - icells.data();
    int r = 0;
    for (auto& j : i.listFar) {
      auto x = j - jcells.data();
      r = std::max(r, d[y + x * ld].R);
    }

    if (r > 0) {
      int osp = std::min(i.NBODY, r + p);
      base[y] = Matrix(i.NBODY, osp, i.NBODY);
      std::fill(base[y].A.begin(), base[y].A.end(), 0);
    }

    for (auto& j : i.listFar) {
      auto x = j - jcells.data();
      Matrix& m = d[y + x * ld];
      SampleP2Pi(base[y], m);
    }
  }
}

void nbd::sample_base_j(Cells& icells, Cells& jcells, Matrices& d, int ld, Matrices& base, int p) {
  for (auto& j : jcells) {
    auto x = &j - jcells.data();
    int r = 0;
    for (auto& i : j.listFar) {
      auto y = i - icells.data();
      r = std::max(r, d[y + x * ld].R);
    }

    if (r > 0) {
      int osp = std::min(j.NBODY, r + p);
      base[x] = Matrix(j.NBODY, osp, j.NBODY);
      std::fill(base[x].A.begin(), base[x].A.end(), 0);
    }

    for (auto& i : j.listFar) {
      auto y = i - icells.data();
      Matrix& m = d[y + x * ld];
      SampleP2Pi(base[x], m);
    }
  }
}


void nbd::sample_base_recur(Cell* cell, Matrix* base) {

  int c_off = 0;
  for (Cell* c = cell->CHILD; c != cell->CHILD + cell->NCHILD; c++) {
    auto i = c - cell;
    if (base[i].N == 0 && base->N > 0) {
      base[i] = Matrix(c->NBODY, base->N, c->NBODY);
      std::fill(base[i].A.begin(), base[i].A.end(), 0);
    }
    if (base->N > 0)
      SampleParent(base[i], *base, c_off);
    sample_base_recur(c, base + i);
    c_off += c->NBODY;
  }

}

void nbd::shared_base_i(Cells& icells, Cells& jcells, Matrices& d, int ld, Matrices& base, bool symm) {
  for (auto& i : icells) {
    auto y = &i - icells.data();
    BasisOrth(base[y]);
    for (auto& j : i.listFar) {
      auto x = j - jcells.data();
      Matrix& m = d[y + x * ld];
      BasisInvLeft(base[y], m);
    }
  }

  if (symm)
    for (auto& i : icells) {
      auto y = &i - icells.data();
      for (auto& j : i.listFar) {
        auto x = j - jcells.data();
        Matrix& m = d[y + x * ld];
        BasisInvRight(base[x], m);
      }
    }
}

void nbd::shared_base_j(Cells& icells, Cells& jcells, Matrices& d, int ld, Matrices& base) {
  for (auto& j : jcells) {
    auto x = &j - jcells.data();
    BasisOrth(base[x]);
    for (auto& i : j.listFar) {
      auto y = i - icells.data();
      Matrix& m = d[y + x * ld];
      BasisInvRight(base[x], m);
    }
  }
}

void nbd::nest_base(Cell* icell, Matrix* base) {
  if (icell->NCHILD == 0)
    return;

  if (base->N > 0) {
    Matrix* m = base + (icell->CHILD - icell);
    BasisInvMultipleLeft(m, icell->NCHILD, *base);
  }

  for (Cell* c = icell->CHILD; c != icell->CHILD + icell->NCHILD; c++) {
    Matrix* m = base + (c - icell);
    nest_base(c, m);
  }
}

void nbd::traverse_i(Cells& icells, Cells& jcells, Matrices& d, Matrices& base, int p) {
  int ld = (int)icells.size();
  base.resize(icells.size());
  sample_base_i(icells, jcells, d, ld, base, p);
  sample_base_recur(&icells[0], &base[0]);

  shared_base_i(icells, jcells, d, ld, base, &icells == &jcells);
  nest_base(&icells[0], &base[0]);
}

void nbd::traverse_j(Cells& icells, Cells& jcells, Matrices& d, Matrices& base, int p) {
  int ld = (int)icells.size();
  base.resize(jcells.size());
  sample_base_j(icells, jcells, d, ld, base, p);
  sample_base_recur(&jcells[0], &base[0]);
  shared_base_j(icells, jcells, d, ld, base);
  nest_base(&jcells[0], &base[0]);
}

void nbd::shared_epilogue(Matrices& d) {
  for (auto& m : d)
    if (m.R > 0)
      MergeS(m);
}

