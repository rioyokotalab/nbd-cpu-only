
#include "build_tree.h"
#include "kernel.h"

#include <cmath>
#include <algorithm>
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

void nbd::evaluate(eval_func_t r2f, Cells& cells, int dim, Matrices& d, Matrices& lr, int rank) {
  for (auto& i : cells) {
    auto y = &i - cells.data();
    for (auto& j : i.listFar) {
      auto x = j - cells.data();
      M2L(r2f, &i, j, dim, d[y + x * cells.size()], lr[y + x * cells.size()], rank);
    }
    for (auto& j : i.listNear) {
      auto x = j - cells.data();
      P2P(r2f, &i, j, dim, d[y + x * cells.size()]);
    }
  }
}

void nbd::traverse(eval_func_t r2f, Cells& icells, Cells& jcells, int dim, Matrices& d, Matrices& lr, real_t theta, int rank) {
  getList(&icells[0], &jcells[0], dim, theta);
  d.resize(icells.size() * jcells.size());
  lr.resize(icells.size() * jcells.size());

  evaluate(r2f, icells, dim, d, lr, rank);
}

inline real_t rand(real_t min, real_t max) {
  return min + (max - min) * ((double)std::rand() / RAND_MAX);
}

void scalBox(real_t a, int dim, const real_t Xmin[], const real_t Xmax[], real_t Xmin_out[], real_t Xmax_out[]) {
  for (int d = 0; d < dim; d++) {
    real_t cc = (Xmin[d] + Xmax[d]) / 2;
    real_t rc = cc - Xmin[d];

    real_t arc = a * rc;
    Xmin_out[d] = cc - arc;
    Xmax_out[d] = cc + arc;
  }
}

void nbd::getBoundBox(int m, Cell* cell, Bodies& box, int dim, real_t s) {
  box.resize(m);

  real_t Xmin_b[nbd::dim], Xmax_b[nbd::dim];
  scalBox(s, dim, cell->Xmin, cell->Xmax, Xmin_b, Xmax_b);

  int i = 0;
  for (int d = 0; d < dim; d++) {
    int end = m * (2 * d + 1) / dim / 2;
    while (i < end) {
      for (int db = 0; db < dim; db++)
        box[i].X[db] = db == d ? Xmin_b[db] : rand(Xmin_b[db], Xmax_b[db]);
      i++;
    }

    end = m * (2 * d + 2) / dim / 2;
    while (i < end) {
      for (int db = 0; db < dim; db++)
        box[i].X[db] = db == d ? Xmax_b[db] : rand(Xmin_b[db], Xmax_b[db]);
      i++;
    }
  }

}


void nbd::printTree(const Cell* cell, int level, int offset) {
  for (int i = 0; i < level; i++)
    printf("  ");
  printf("<%d, %d>", offset, offset + cell->NBODY);
  printf("\n");
  for (auto c = cell->CHILD; c != cell->CHILD + cell->NCHILD; c++)
    printTree(c, level + 1, offset + (int)(c->BODY - cell->BODY));
}
