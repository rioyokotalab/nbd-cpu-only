
#include "build_tree.h"
#include <cmath>
#include <algorithm>
#include <iterator>

using namespace nbd;

Bodies::iterator spart(Bodies::iterator first, Bodies::iterator last, int sdim, real_t pivot) {

  auto l = [pivot, sdim](const Body& element) { return element.X[sdim] < pivot; };
  return std::partition(first, last, l);
}

Bodies::iterator spart_size_k(Bodies::iterator first, Bodies::iterator last, int sdim, int64_t k) {

  if (last == first)
    return first;
  real_t pivot = std::next(first, std::distance(first, last) / 2)->X[sdim];

  Bodies::iterator i1 = std::partition(first, last,
    [pivot, sdim](const Body& element) { return element.X[sdim] < pivot; });
  Bodies::iterator i2 = std::partition(i1, last,
    [pivot, sdim](const Body& element) { return !(pivot < element.X[sdim]); });
  int64_t d1 = i1 - first, d2 = i2 - first;
  
  if (d1 > k)
    return spart_size_k(first, i1, sdim, k);
  else if (d2 < k)
    return spart_size_k(i2, last, sdim, k - d2);
  else
    return first + k;
}

Cells nbd::buildTree(Bodies& bodies, int ncrit, int dim) {

  Cells cells(1);
  cells.reserve(bodies.size());

  cells[0].BODY = bodies.data();
  cells[0].NBODY = bodies.size();
  cells[0].NCHILD = 0;
  for (int d = 0; d < dim; d++) 
    cells[0].Xmin[d] = cells[0].Xmax[d] = bodies[0].X[d];
  for (int64_t b = 0; b < (int64_t)bodies.size(); b++) {
    for (int d = 0; d < dim; d++) 
      cells[0].Xmin[d] = std::fmin(bodies[b].X[d], cells[0].Xmin[d]);
    for (int d = 0; d < dim; d++) 
      cells[0].Xmax[d] = std::fmax(bodies[b].X[d], cells[0].Xmax[d]);
  }

  int64_t nlis = (bodies.size() + ncrit - 1) / ncrit, iters = 0;
  int sdim = 0;
  while (nlis >>= 1) ++iters;
  int64_t last_off = 0, last_len = 1;

  for (int64_t i = 1; i <= iters; i++) {

    int64_t len = 0;

    for (int64_t j = last_off; j < last_off + last_len; j++) {
      Cell& cell = cells[j];
      Bodies::iterator cell_b = bodies.begin() + std::distance(bodies.data(), cell.BODY);

#ifdef PART_EQ_SIZE
      auto p = spart_size_k(cell_b, cell_b + cell.NBODY, sdim, cell.NBODY / 2);
      real_t med = p->X[sdim];
#else
      real_t med = (cell.Xmin[sdim] + cell.Xmax[sdim]) / 2;
      auto p = spart(cell_b, cell_b + cell.NBODY, sdim, med);
#endif
      int64_t size[2];
      size[0] = p - cell_b;
      size[1] = cell.NBODY - size[0];

      cell.NCHILD = (int64_t)(size[0] > 0) + (int64_t)(size[1] > 0);
      cells.resize(cells.size() + cell.NCHILD);
      Cell* child = &cells.back() - cell.NCHILD + 1;
      cell.CHILD = child;

      int64_t offset = 0;
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

