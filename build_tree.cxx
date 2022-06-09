
#include "build_tree.hxx"
#include "basis.hxx"
#include "dist.hxx"

#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <set>
#include <iostream>
#include <fstream>

using namespace nbd;

int64_t nbd::partition(Body* bodies, int64_t nbodies, int64_t sdim) {
  auto comp = [sdim](const Body& b1, const Body& b2) -> bool {
    return b1.X[sdim] < b2.X[sdim];
  };
  int64_t loc = nbodies / 2;
  std::nth_element(bodies, bodies + loc, bodies + nbodies, comp);
  return loc;
}

int64_t nbd::buildTree(Cell* cells, Body* bodies, int64_t nbodies, int64_t levels) {
  int64_t nleaves = (int64_t)1 << levels;
  int64_t ncells = nleaves + nleaves - 1;

  Cell* root = &cells[0];
  root->BODY = &bodies[0];
  root->NBODY = nbodies;
  root->ZID = 0;
  root->LEVEL = 0;
  get_bounds(root->BODY, root->NBODY, root->R, root->C);

  for (int64_t i = 0; i < ncells; i++) {
    Cell* ci = &cells[i];

    if (ci->LEVEL < levels) {
      int64_t sdim = 0;
      double maxR = ci->R[0];
      if (ci->R[1] > maxR)
      { sdim = 1; maxR = ci->R[1]; }
      if (ci->R[2] > maxR)
      { sdim = 2; maxR = ci->R[2]; }

      int64_t loc = partition(ci->BODY, ci->NBODY, sdim);

      Cell* c0 = &cells[(i << 1) + 1];
      Cell* c1 = &cells[(i << 1) + 2];
      ci->CHILD = c0;
      ci->NCHILD = 2;

      c0->SIBL = c1;
      c0->BODY = ci->BODY;
      c0->NBODY = loc;
      c0->ZID = (ci->ZID) << 1;
      c0->LEVEL = ci->LEVEL + 1;

      c1->SIBL = c0;
      c1->BODY = ci->BODY + loc;
      c1->NBODY = ci->NBODY - loc;
      c1->ZID = ((ci->ZID) << 1) + 1;
      c1->LEVEL = ci->LEVEL + 1;

      get_bounds(c0->BODY, c0->NBODY, c0->R, c0->C);
      get_bounds(c1->BODY, c1->NBODY, c1->R, c1->C);
    }
    else {
      ci->CHILD = NULL;
      ci->NCHILD = 0;
    }
  }

  return levels;
}


void nbd::readPartitionedBodies(const char fname[], Body* bodies, int64_t nbodies, int64_t buckets[], int64_t dim) {
  std::ifstream file;
  file.open(fname, std::ios_base::in);

  int64_t count = 0;
  int64_t bucket_i = 1;
  for (int64_t i = 0; i < nbodies; i++) {
    double* arr = bodies[i].X;
    for (int64_t d = 0; d < dim; d++)
      file >> arr[d];
    
    int64_t loc;
    file >> loc;
    if (loc > bucket_i) {
      buckets[bucket_i - 1] = count;
      count = 0;
      bucket_i = loc;
    }
    count = count + 1;
  }
  file.close();
  if (count > 0)
    buckets[bucket_i - 1] = count;
}

void nbd::buildTreeBuckets(Cell* cells, Body* bodies, const int64_t buckets[], int64_t levels) {
  int64_t nleaves = (int64_t)1 << levels;
  int64_t ileaf = nleaves - 1;

  int64_t offset = 0;
  for (int64_t i = 0; i < nleaves; i++) {
    Cell* ci = &cells[ileaf + i];
    ci->CHILD = NULL;
    ci->NCHILD = 0;
    ci->BODY = bodies + offset;
    ci->NBODY = buckets[i];
    ci->SIBL = NULL;
    ci->ZID = i;
    ci->LEVEL = levels;
    ci->listNear.clear();
    ci->listFar.clear();
    ci->Multipole.clear();
    get_bounds(ci->BODY, ci->NBODY, ci->R, ci->C);
    offset = offset + ci->NBODY;
  }

  for (int64_t i = ileaf - 1; i >= 0; i--) {
    Cell* ci = &cells[i];
    Cell* c0 = &cells[(i << 1) + 1];
    Cell* c1 = &cells[(i << 1) + 2];
    ci->CHILD = c0;
    ci->NCHILD = 2;
    ci->BODY = c0->BODY;
    ci->NBODY = c0->NBODY + c1->NBODY;
    ci->SIBL = NULL;
    ci->ZID = (c0->ZID) >> 1;
    ci->LEVEL = c0->LEVEL - 1;
    ci->listNear.clear();
    ci->listFar.clear();
    ci->Multipole.clear();
    get_bounds(ci->BODY, ci->NBODY, ci->R, ci->C);
    c0->SIBL = c1;
    c1->SIBL = c0;
  }
}

void nbd::getList(Cell* Ci, Cell* Cj, double theta) {
  if (Ci->LEVEL < Cj->LEVEL)
    for (Cell* ci = Ci->CHILD; ci != Ci->CHILD + Ci->NCHILD; ci++)
      getList(ci, Cj, theta);
  else if (Cj->LEVEL < Ci->LEVEL)
    for (Cell* cj = Cj->CHILD; cj != Cj->CHILD + Cj->NCHILD; cj++)
      getList(Ci, cj, theta);
  else {
    double dC = 0.;
    double dRi = 0.;
    double dRj = 0.;
    for (int64_t d = 0; d < 3; d++) {
      double diff = Ci->C[d] - Cj->C[d];
      dC = dC + diff * diff;
      dRi = dRi + Ci->R[d] * Ci->R[d];
      dRj = dRj + Cj->R[d] * Cj->R[d];
    }
    double dR = (dRi + dRj) * theta;

    if (dC > dR)
      Ci->listFar.push_back(Cj);
    else {
      Ci->listNear.push_back(Cj);

      if (Ci->NCHILD > 0)
        for (Cell* ci = Ci->CHILD; ci != Ci->CHILD + Ci->NCHILD; ci++)
          getList(ci, Cj, theta);
    }
  }
}

void nbd::findCellsAtLevel(const Cell* cells[], int64_t* len, const Cell* cell, int64_t level) {
  if (level == cell->LEVEL) {
    int64_t i = *len;
    cells[i] = cell;
    *len = i + 1;
  }
  else if (level > cell->LEVEL && cell->NCHILD > 0)
    for (int64_t i = 0; i < cell->NCHILD; i++)
      findCellsAtLevel(cells, len, cell->CHILD + i, level);
}

void nbd::findCellsAtLevelModify(Cell* cells[], int64_t* len, Cell* cell, int64_t level) {
  if (level == cell->LEVEL) {
    int64_t i = *len;
    cells[i] = cell;
    *len = i + 1;
  }
  else if (level > cell->LEVEL && cell->NCHILD > 0)
    for (int64_t i = 0; i < cell->NCHILD; i++)
      findCellsAtLevelModify(cells, len, cell->CHILD + i, level);
}

const Cell* nbd::findLocalAtLevel(const Cell* cell, int64_t level) {
  const Cell* iter = cell;
  int64_t mpi_rank;
  int64_t mpi_levels;
  commRank(&mpi_rank, &mpi_levels);
  int64_t iters = level < mpi_levels ? level : mpi_levels;

  for (int64_t i = iter->LEVEL + 1; i <= iters; i++) {
    int64_t lvl_diff = mpi_levels - i;
    int64_t my_rank = mpi_rank >> lvl_diff;
    int64_t nchild = iter->NCHILD;
    Cell* child = iter->CHILD;
    for (int64_t n = 0; n < nchild; n++)
      if (child[n].ZID == my_rank)
        iter = child + n;
  }

  int64_t my_rank = mpi_rank >> (mpi_levels - iters);
  if (iter->ZID == my_rank)
    return iter;
  else
    return nullptr;
}

Cell* nbd::findLocalAtLevelModify(Cell* cell, int64_t level) {
  Cell* iter = cell;
  int64_t mpi_rank;
  int64_t mpi_levels;
  commRank(&mpi_rank, &mpi_levels);
  int64_t iters = level < mpi_levels ? level : mpi_levels;

  for (int64_t i = iter->LEVEL + 1; i <= iters; i++) {
    int64_t lvl_diff = mpi_levels - i;
    int64_t my_rank = mpi_rank >> lvl_diff;
    int64_t nchild = iter->NCHILD;
    Cell* child = iter->CHILD;
    for (int64_t n = 0; n < nchild; n++)
      if (child[n].ZID == my_rank)
        iter = child + n;
  }

  int64_t my_rank = mpi_rank >> (mpi_levels - iters);
  if (iter->ZID == my_rank)
    return iter;
  else
    return nullptr;
}


void nbd::traverse(Cell* cells, int64_t levels, int64_t theta) {
  getList(&cells[0], &cells[0], theta);
  int64_t mpi_levels;
  commRank(NULL, &mpi_levels);

  configureComm(levels, NULL, 0);
  const Cell* local = &cells[0];
  for (int64_t i = 0; i <= levels; i++) {
    local = findLocalAtLevel(local, i);
    int64_t nodes = i > mpi_levels ? (int64_t)1 << (i - mpi_levels) : 1;

    int64_t len = 0;
    std::vector<const Cell*> leaves(nodes);
    findCellsAtLevel(&leaves[0], &len, local, i);
    std::set<int64_t> ngbs;

    for (int64_t n = 0; n < len; n++) {
      const Cell* c = leaves[n];
      int64_t nlen = c->listNear.size();
      for (int64_t j = 0; j < nlen; j++) {
        const Cell* cj = (c->listNear)[j];
        int64_t ngb = (cj->ZID) / nodes;
        ngbs.emplace(ngb);
      }
      int64_t flen = c->listFar.size();
      for (int64_t j = 0; j < flen; j++) {
        const Cell* cj = (c->listFar)[j];
        int64_t ngb = (cj->ZID) / nodes;
        ngbs.emplace(ngb);
      }
    }

    int64_t size = ngbs.size();
    std::vector<int64_t> ngbs_v(size);
    std::set<int64_t>::iterator iter = ngbs.begin();
    for (int64_t n = 0; n < size; n++) {
      ngbs_v[n] = *iter;
      iter = std::next(iter);
    }

    configureComm(i, &ngbs_v[0], size);
  }
}

int64_t nbd::remoteBodies(Body* remote, int64_t size, const Cell& cell, const Body* bodies, int64_t nbodies) {
  int64_t avail = nbodies;
  int64_t len = cell.listNear.size();
  std::vector<int64_t> offsets(len);
  std::vector<int64_t> lens(len);

  const Body* begin = &bodies[0];
  for (int64_t i = 0; i < len; i++) {
    const Cell* c = cell.listNear[i];
    avail = avail - c->NBODY;
    offsets[i] = c->BODY - begin;
    lens[i] = c->NBODY;
  }

  size = size > avail ? avail : size;

  for (int64_t i = 0; i < size; i++) {
    int64_t loc = (int64_t)((double)(avail * i) / size);
    for (int64_t j = 0; j < len; j++)
      if (loc >= offsets[j])
        loc = loc + lens[j];

    remote[i].X[0] = bodies[loc].X[0];
    remote[i].X[1] = bodies[loc].X[1];
    remote[i].X[2] = bodies[loc].X[2];
    remote[i].B = bodies[loc].B;
  }
  return size;
}

int64_t nbd::closeBodies(Body* remote, int64_t size, const Cell& cell) {
  int64_t avail = 0;
  int64_t len = cell.listNear.size();
  std::vector<int64_t> offsets(len);
  std::vector<int64_t> lens(len);

  int64_t cpos = -1;
  const Body* begin = cell.BODY;
  for (int64_t i = 0; i < len; i++) {
    const Cell* c = cell.listNear[i];
    offsets[i] = c->BODY - begin;
    lens[i] = c->NBODY;
    if (c != &cell)
      avail = avail + c->NBODY;
    else
      cpos = i;
  }

  size = size > avail ? avail : size;

  for (int64_t i = 0; i < size; i++) {
    int64_t loc = (int64_t)((double)(avail * i) / size);
    int64_t region = -1;
    for (int64_t j = 0; j < len; j++)
      if (j != cpos && region == -1) {
        if (loc < lens[j]) {
          region = j;
          loc = loc + offsets[region];
        }
        else
          loc = loc - lens[j];
      }
    remote[i].X[0] = begin[loc].X[0];
    remote[i].X[1] = begin[loc].X[1];
    remote[i].X[2] = begin[loc].X[2];
    remote[i].B = begin[loc].B;
  }
  return size;
}

void nbd::collectChildMultipoles(const Cell& cell, int64_t multipoles[]) {
  if (cell.NCHILD > 0) {
    int64_t count = 0;
    for (int64_t i = 0; i < cell.NCHILD; i++) {
      const Cell& c = cell.CHILD[i];
      int64_t loc = c.BODY - cell.BODY;
      int64_t len = c.Multipole.size();
      for (int64_t n = 0; n < len; n++) {
        int64_t nloc = loc + c.Multipole[n];
        multipoles[count] = nloc;
        count += 1;
      }
    }
  }
  else {
    int64_t len = cell.NBODY;
    std::iota(multipoles, multipoles + len, 0);
  }
}

void nbd::childMultipoleSize(int64_t* size, const Cell& cell) {
  if (cell.NCHILD > 0) {
    int64_t s = 0;
    for (int64_t i = 0; i < cell.NCHILD; i++)
      s += cell.CHILD[i].Multipole.size();
    *size = s;
  }
  else
    *size = cell.NBODY;
}


void nbd::relationsNear(CSC rels[], const Cell* cells, int64_t levels) {
  int64_t mpi_rank;
  int64_t mpi_levels;
  commRank(&mpi_rank, &mpi_levels);

  for (int64_t i = 0; i <= levels; i++) {
    int64_t mpi_boxes = i > mpi_levels ? (int64_t)1 << (i - mpi_levels) : 1;
    int64_t mpi_dups = i < mpi_levels ? (mpi_levels - i) : 0;
    CSC& csc = rels[i];

    csc.M = (int64_t)1 << i;
    csc.N = mpi_boxes;
    csc.COLS_NEAR.resize(mpi_boxes + 1);
    csc.COLS_FAR.resize(mpi_boxes + 1);
    std::fill(csc.COLS_NEAR.begin(), csc.COLS_NEAR.end(), 0);
    std::fill(csc.COLS_FAR.begin(), csc.COLS_FAR.end(), 0);
    csc.ROWS_NEAR.clear();
    csc.ROWS_FAR.clear();
    csc.NNZ_NEAR = 0;
    csc.NNZ_FAR = 0;
    csc.CBGN = (mpi_rank >> mpi_dups) * mpi_boxes;
  }

  int64_t nleaves = (int64_t)1 << levels;
  int64_t ncells = nleaves + nleaves - 1;

  for (int64_t i = 0; i < ncells; i++) {
    const Cell& c = cells[i];
    int64_t l = c.LEVEL;
    CSC& csc = rels[l];
    int64_t n = c.ZID - csc.CBGN;
    if (n >= 0 && n < csc.N) {
      int64_t ent = c.listNear.size();
      csc.COLS_NEAR[n] = ent;
      for (int64_t j = 0; j < ent; j++)
        csc.ROWS_NEAR.emplace_back((c.listNear[j])->ZID);
      csc.NNZ_NEAR = csc.NNZ_NEAR + ent;

      ent = c.listFar.size();
      csc.COLS_FAR[n] = ent;
      for (int64_t j = 0; j < ent; j++)
        csc.ROWS_FAR.emplace_back((c.listFar[j])->ZID);
      csc.NNZ_FAR = csc.NNZ_FAR + ent;
    }
  }

  for (int64_t i = 0; i <= levels; i++) {
    CSC& csc = rels[i];
    int64_t count_n = 0;
    int64_t count_f = 0;
    for (int64_t j = 0; j <= csc.N; j++) {
      int64_t ent_n = csc.COLS_NEAR[j];
      int64_t ent_f = csc.COLS_FAR[j];
      csc.COLS_NEAR[j] = count_n;
      csc.COLS_FAR[j] = count_f;
      count_n = count_n + ent_n;
      count_f = count_f + ent_f;
    }
  }
}

void nbd::evaluateLeafNear(Matrix* d, KerFunc_t ef, const Cell* cell, const CSC& csc) {
  if (cell->NCHILD > 0)
    for (int64_t i = 0; i < cell->NCHILD; i++)
      evaluateLeafNear(d, ef, cell->CHILD + i, csc);
  else {
    int64_t n = cell->ZID - csc.CBGN;
    if (n >= 0 && n < csc.N) {
      int64_t len = cell->listNear.size();
      int64_t off = csc.COLS_NEAR[n];
      for (int64_t i = 0; i < len; i++) {
        int64_t m = cell->listNear[i]->NBODY;
        int64_t n = cell->NBODY;
        cMatrix(d[off + i], m, n);
        gen_matrix(ef, m, n, cell->listNear[i]->BODY, cell->BODY, d[off + i].A.data(), NULL, NULL);
      }
    }
  }
}

void nbd::evaluateFar(Matrix* s, KerFunc_t ef, const Cell* cell, const CSC& csc, int64_t level) {
  if (cell->LEVEL < level)
    for (int64_t i = 0; i < cell->NCHILD; i++)
      evaluateFar(s, ef, cell->CHILD + i, csc, level);
  else {
    int64_t n = cell->ZID - csc.CBGN;
    if (n >= 0 && n < csc.N) {
      int64_t len = cell->listFar.size();
      int64_t off = csc.COLS_FAR[n];
      for (int64_t i = 0; i < len; i++) {
        int64_t m = cell->listFar[i]->Multipole.size();
        int64_t n = cell->Multipole.size();
        cMatrix(s[off + i], m, n);
        gen_matrix(ef, m, n, cell->listFar[i]->BODY, cell->BODY, s[off + i].A.data(), cell->listFar[i]->Multipole.data(), cell->Multipole.data());
      }
    }
  }
}

void nbd::lookupIJ(char NoF, int64_t& ij, const CSC& rels, int64_t i, int64_t j) {
  int64_t lj = j - rels.CBGN;
  if (lj < 0 || lj >= rels.N)
  { ij = -1; return; }
  if (NoF == 'N' || NoF == 'n') {
    int64_t k = std::distance(rels.ROWS_NEAR.data(), 
      std::find(rels.ROWS_NEAR.data() + rels.COLS_NEAR[lj], rels.ROWS_NEAR.data() + rels.COLS_NEAR[lj + 1], i));
    ij = (k < rels.COLS_NEAR[lj + 1]) ? k : -1;
  }
  else if (NoF == 'F' || NoF == 'f') {
    int64_t k = std::distance(rels.ROWS_FAR.data(), 
      std::find(rels.ROWS_FAR.data() + rels.COLS_FAR[lj], rels.ROWS_FAR.data() + rels.COLS_FAR[lj + 1], i));
    ij = (k < rels.COLS_FAR[lj + 1]) ? k : -1;
  }
}


void nbd::loadX(Vector* X, const Cell* cell, int64_t level) {
  int64_t xlen = (int64_t)1 << level;
  contentLength(&xlen, level);

  int64_t ibegin = 0;
  int64_t iend = (int64_t)1 << level;
  selfLocalRange(&ibegin, &iend, level);
  int64_t nodes = iend - ibegin;

  int64_t len = 0;
  std::vector<const Cell*> cells(nodes);
  std::vector<int64_t> dims(xlen);
  findCellsAtLevel(&cells[0], &len, cell, level);

  for (int64_t i = 0; i < len; i++) {
    const Cell* ci = cells[i];
    int64_t li = ci->ZID;
    iLocal(&li, ci->ZID, level);
    Vector& Xi = X[li];
    cVector(Xi, ci->NBODY);
    dims[li] = ci->NBODY;

    for (int64_t n = 0; n < ci->NBODY; n++)
      Xi.X[n] = ci->BODY[n].B;
  }

  DistributeDims(&dims[0], level);

  for (int64_t i = 0; i < xlen; i++)
    if (X[i].N != dims[i])
      cVector(X[i], dims[i]);
  DistributeVectorsList(X, level);
}

void nbd::h2MatVecReference(Vector* B, KerFunc_t ef, const Cell* root, int64_t levels) {
  int64_t len = 0, lenj = 0;
  std::vector<const Cell*> cells((int64_t)1 << levels);
  std::vector<const Cell*> cells_leaf((int64_t)1 << levels);

  const Cell* local = findLocalAtLevel(root, levels);
  findCellsAtLevel(&cells[0], &len, local, levels);
  findCellsAtLevel(&cells_leaf[0], &lenj, root, levels);

#pragma omp parallel for
  for (int64_t i = 0; i < len; i++) {
    const Cell* ci = cells[i];
    int64_t li = ci->ZID;
    iLocal(&li, ci->ZID, levels);
    Vector& Bi = B[li];
    cVector(Bi, ci->NBODY);
    zeroVector(Bi);

    for (int64_t j = 0; j < lenj; j++) {
      Vector X;
      int64_t m = ci->NBODY;
      int64_t n = cells_leaf[j]->NBODY;
      cVector(X, n);
      for (int64_t k = 0; k < n; k++)
        X.X[k] = cells_leaf[j]->BODY[k].B;
      
      Matrix Aij;
      cMatrix(Aij, m, n);
      gen_matrix(ef, m, n, ci->BODY, cells_leaf[j]->BODY, Aij.A.data(), NULL, NULL);
      mvec('N', Aij, X, Bi, 1., 1.);
      cMatrix(Aij, 0, 0);
      cVector(X, 0);
    }
  }
}
