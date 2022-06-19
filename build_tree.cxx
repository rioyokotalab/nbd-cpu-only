
#include "build_tree.h"
#include "basis.h"
#include "dist.h"

#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <set>
#include <iostream>
#include <fstream>

void buildTree(Cell* cells, Body* bodies, int64_t nbodies, int64_t levels) {
  int64_t nleaves = (int64_t)1 << levels;
  int64_t ncells = nleaves + nleaves - 1;

  Cell* root = &cells[0];
  root->BODY[0] = 0;
  root->BODY[1] = nbodies;
  root->ZID = 0;
  root->LEVEL = 0;
  get_bounds(bodies, nbodies, root->R, root->C);

  for (int64_t i = 0; i < ncells; i++) {
    Cell* ci = &cells[i];

    if (ci->LEVEL < levels) {
      int64_t sdim = 0;
      double maxR = ci->R[0];
      if (ci->R[1] > maxR)
      { sdim = 1; maxR = ci->R[1]; }
      if (ci->R[2] > maxR)
      { sdim = 2; maxR = ci->R[2]; }

      int64_t i_begin = ci->BODY[0];
      int64_t i_end = ci->BODY[1];
      int64_t nbody_i = i_end - i_begin;
      sort_bodies(&bodies[i_begin], nbody_i, sdim);
      int64_t loc = i_begin + nbody_i / 2;

      Cell* c0 = &cells[(i << 1) + 1];
      Cell* c1 = &cells[(i << 1) + 2];
      ci->CHILD = c0;
      ci->NCHILD = 2;

      c0->SIBL = c1;
      c0->BODY[0] = i_begin;
      c0->BODY[1] = loc;
      c0->ZID = (ci->ZID) << 1;
      c0->LEVEL = ci->LEVEL + 1;

      c1->SIBL = c0;
      c1->BODY[0] = loc;
      c1->BODY[1] = i_end;
      c1->ZID = ((ci->ZID) << 1) + 1;
      c1->LEVEL = ci->LEVEL + 1;

      get_bounds(&bodies[i_begin], loc - i_begin, c0->R, c0->C);
      get_bounds(&bodies[loc], i_end - loc, c1->R, c1->C);
    }
    else {
      ci->CHILD = NULL;
      ci->NCHILD = 0;
    }
  }
}


void readPartitionedBodies(const char fname[], Body* bodies, int64_t nbodies, int64_t buckets[], int64_t dim) {
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

void buildTreeBuckets(Cell* cells, Body* bodies, const int64_t buckets[], int64_t levels) {
  int64_t nleaves = (int64_t)1 << levels;
  int64_t ileaf = nleaves - 1;

  int64_t offset = 0;
  for (int64_t i = 0; i < nleaves; i++) {
    Cell* ci = &cells[ileaf + i];
    ci->CHILD = NULL;
    ci->NCHILD = 0;
    ci->BODY[0] = offset;
    ci->BODY[1] = offset + buckets[i];
    ci->SIBL = NULL;
    ci->ZID = i;
    ci->LEVEL = levels;
    ci->listNear.clear();
    ci->listFar.clear();
    ci->Multipole.clear();
    get_bounds(&bodies[offset], buckets[i], ci->R, ci->C);
    offset = ci->BODY[1];
  }

  for (int64_t i = ileaf - 1; i >= 0; i--) {
    Cell* ci = &cells[i];
    Cell* c0 = &cells[(i << 1) + 1];
    Cell* c1 = &cells[(i << 1) + 2];
    ci->CHILD = c0;
    ci->NCHILD = 2;
    ci->BODY[0] = c0->BODY[0];
    ci->BODY[1] = c1->BODY[1];
    ci->SIBL = NULL;
    ci->ZID = (c0->ZID) >> 1;
    ci->LEVEL = c0->LEVEL - 1;
    ci->listNear.clear();
    ci->listFar.clear();
    ci->Multipole.clear();
    get_bounds(&bodies[ci->BODY[0]], ci->BODY[1] - ci->BODY[0], ci->R, ci->C);
    c0->SIBL = c1;
    c1->SIBL = c0;
  }
}

void getList(Cell* Ci, Cell* Cj, double theta) {
  if (Ci->LEVEL < Cj->LEVEL)
    for (Cell* ci = Ci->CHILD; ci != Ci->CHILD + Ci->NCHILD; ci++)
      getList(ci, Cj, theta);
  else if (Cj->LEVEL < Ci->LEVEL)
    for (Cell* cj = Cj->CHILD; cj != Cj->CHILD + Cj->NCHILD; cj++)
      getList(Ci, cj, theta);
  else {
    int admis;
    admis_check(&admis, theta, Ci->C, Cj->C, Ci->R, Cj->R);
    if (admis)
      Ci->listFar.push_back(Cj);
    else {
      Ci->listNear.push_back(Cj);

      if (Ci->NCHILD > 0)
        for (Cell* ci = Ci->CHILD; ci != Ci->CHILD + Ci->NCHILD; ci++)
          getList(ci, Cj, theta);
    }
  }
}


void traverse(Cell* cells, int64_t levels, int64_t theta) {
  getList(&cells[0], &cells[0], theta);
  int64_t mpi_rank, mpi_levels;
  commRank(&mpi_rank, &mpi_levels);

  configureComm(levels, NULL, 0);
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nodes = i > mpi_levels ? (int64_t)1 << (i - mpi_levels) : 1;
    int64_t lvl_diff = i < mpi_levels ? mpi_levels - i : 0;
    int64_t my_rank = mpi_rank >> lvl_diff;
    int64_t gbegin = my_rank * nodes;

    int64_t len = (int64_t)1 << i;
    Cell* leaves = &cells[len - 1];
    std::set<int64_t> ngbs;

    for (int64_t n = 0; n < nodes; n++) {
      const Cell* c = &leaves[n + gbegin];
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

int64_t remoteBodies(int64_t* remote, int64_t size, const Cell& cell, int64_t nbodies) {
  int64_t avail = nbodies;
  int64_t len = cell.listNear.size();
  std::vector<int64_t> offsets(len);
  std::vector<int64_t> lens(len);

  for (int64_t i = 0; i < len; i++) {
    const Cell* c = cell.listNear[i];
    offsets[i] = c->BODY[0];
    lens[i] = c->BODY[1] - c->BODY[0];
    avail = avail - lens[i];
  }

  size = size > avail ? avail : size;

  for (int64_t i = 0; i < size; i++) {
    int64_t loc = (int64_t)((double)(avail * i) / size);
    for (int64_t j = 0; j < len; j++)
      if (loc >= offsets[j])
        loc = loc + lens[j];
    remote[i] = loc;
  }
  return size;
}

int64_t closeBodies(int64_t* remote, int64_t size, const Cell& cell) {
  int64_t avail = 0;
  int64_t len = cell.listNear.size();
  std::vector<int64_t> offsets(len);
  std::vector<int64_t> lens(len);

  int64_t cpos = -1;
  for (int64_t i = 0; i < len; i++) {
    const Cell* c = cell.listNear[i];
    offsets[i] = c->BODY[0];
    lens[i] = c->BODY[1] - c->BODY[0];
    if (c != &cell)
      avail = avail + lens[i];
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
    remote[i] = loc;
  }
  return size;
}

void collectChildMultipoles(const Cell& cell, int64_t multipoles[]) {
  if (cell.NCHILD > 0) {
    int64_t count = 0;
    for (int64_t i = 0; i < cell.NCHILD; i++) {
      const Cell& c = cell.CHILD[i];
      int64_t len = c.Multipole.size();
      for (int64_t n = 0; n < len; n++) {
        int64_t nloc = c.Multipole[n];
        multipoles[count] = nloc;
        count += 1;
      }
    }
  }
  else {
    int64_t len = cell.BODY[1] - cell.BODY[0];
    std::iota(multipoles, multipoles + len, cell.BODY[0]);
  }
}

void childMultipoleSize(int64_t* size, const Cell& cell) {
  if (cell.NCHILD > 0) {
    int64_t s = 0;
    for (int64_t i = 0; i < cell.NCHILD; i++)
      s += cell.CHILD[i].Multipole.size();
    *size = s;
  }
  else
    *size = cell.BODY[1] - cell.BODY[0];
}


void relations(char NoF, CSC rels[], const Cell* cells, int64_t levels) {
  int64_t mpi_rank;
  int64_t mpi_levels;
  commRank(&mpi_rank, &mpi_levels);

  for (int64_t i = 0; i <= levels; i++) {
    int64_t mpi_boxes = i > mpi_levels ? (int64_t)1 << (i - mpi_levels) : 1;
    CSC& csc = rels[i];

    csc.M = (int64_t)1 << i;
    csc.N = mpi_boxes;
    csc.COL_INDEX.resize(mpi_boxes + 1);
    std::fill(csc.COL_INDEX.begin(), csc.COL_INDEX.end(), 0);
    csc.ROW_INDEX.clear();
  }

  int64_t nleaves = (int64_t)1 << levels;
  int64_t ncells = nleaves + nleaves - 1;

  for (int64_t i = 0; i < ncells; i++) {
    const Cell& c = cells[i];
    int64_t l = c.LEVEL;
    CSC& csc = rels[l];

    int64_t ibegin = 0, iend = 0, lbegin = 0;
    selfLocalRange(&ibegin, &iend, l);
    iGlobal(&lbegin, ibegin, l);

    int64_t n = c.ZID - lbegin;
    if ((NoF == 'N' || NoF == 'n') && n >= 0 && n < csc.N) {
      int64_t ent = c.listNear.size();
      csc.COL_INDEX[n] = ent;
      for (int64_t j = 0; j < ent; j++) {
        int64_t zi = c.listNear[j]->ZID;
        csc.ROW_INDEX.emplace_back(zi);
      }
    }
    else if ((NoF == 'F' || NoF == 'f') && n >= 0 && n < csc.N) {
      int64_t ent = c.listFar.size();
      csc.COL_INDEX[n] = ent;
      for (int64_t j = 0; j < ent; j++) {
        int64_t zi = c.listFar[j]->ZID;
        csc.ROW_INDEX.emplace_back(zi);
      }
    }
  }

  for (int64_t i = 0; i <= levels; i++) {
    CSC& csc = rels[i];
    int64_t count = 0;
    for (int64_t j = 0; j <= csc.N; j++) {
      int64_t ent = csc.COL_INDEX[j];
      csc.COL_INDEX[j] = count;
      count = count + ent;
    }
  }
}

void evaluate(char NoF, Matrix* d, KerFunc_t ef, const Cell* cell, const Body* bodies, const CSC& csc, int64_t level) {
  int64_t len = (int64_t)1 << level;
  const Cell* leaves = &cell[len - 1];

  int64_t ibegin = 0, iend = 0;
  selfLocalRange(&ibegin, &iend, level);
  int64_t nodes = iend - ibegin;

#pragma omp parallel for
  for (int64_t i = 0; i < nodes; i++) {
    int64_t gi = i + ibegin;
    iGlobal(&gi, i + ibegin, level);
    const Cell* ci = &leaves[gi];
    int64_t off = csc.COL_INDEX[i];

    if (NoF == 'N' || NoF == 'n') {
      int64_t len = ci->listNear.size();
      for (int64_t j = 0; j < len; j++) {
        int64_t i_begin = ci->listNear[j]->BODY[0];
        int64_t j_begin = ci->BODY[0];
        int64_t m = ci->listNear[j]->BODY[1] - i_begin;
        int64_t n = ci->BODY[1] - j_begin;
        matrixCreate(&d[off + j], m, n);
        gen_matrix(ef, m, n, &bodies[i_begin], &bodies[j_begin], d[off + j].A, NULL, NULL);
      }
    }
    else if (NoF == 'F' || NoF == 'f') {
      int64_t len = ci->listFar.size();
      for (int64_t j = 0; j < len; j++) {
        int64_t m = ci->listFar[j]->Multipole.size();
        int64_t n = ci->Multipole.size();
        matrixCreate(&d[off + j], m, n);
        gen_matrix(ef, m, n, bodies, bodies, d[off + j].A, ci->listFar[j]->Multipole.data(), ci->Multipole.data());
      }
    }
  }
}

void lookupIJ(int64_t& ij, const CSC& rels, int64_t i, int64_t j) {
  if (j < 0 || j >= rels.N)
  { ij = -1; return; }
  int64_t k = std::distance(rels.ROW_INDEX.data(), 
    std::find(rels.ROW_INDEX.data() + rels.COL_INDEX[j], rels.ROW_INDEX.data() + rels.COL_INDEX[j + 1], i));
  ij = (k < rels.COL_INDEX[j + 1]) ? k : -1;
}


void loadX(Vector* X, const Cell* cell, const Body* bodies, int64_t level) {
  int64_t xlen = (int64_t)1 << level;
  contentLength(&xlen, level);
  int64_t len = (int64_t)1 << level;
  const Cell* leaves = &cell[len - 1];

#pragma omp parallel for
  for (int64_t i = 0; i < xlen; i++) {
    int64_t gi = i;
    iGlobal(&gi, i, level);
    const Cell* ci = &leaves[gi];

    Vector& Xi = X[i];
    int64_t nbegin = ci->BODY[0];
    int64_t ni = ci->BODY[1] - nbegin;
    vectorCreate(&Xi, ni);
    for (int64_t n = 0; n < ni; n++)
      Xi.X[n] = bodies[n + nbegin].B;
  }
}

void h2MatVecReference(Vector* B, KerFunc_t ef, const Cell* cell, const Body* bodies, int64_t level) {
  int64_t nbodies = cell->BODY[1];
  int64_t xlen = (int64_t)1 << level;
  contentLength(&xlen, level);
  int64_t len = (int64_t)1 << level;
  const Cell* leaves = &cell[len - 1];

#pragma omp parallel for
  for (int64_t i = 0; i < xlen; i++) {
    int64_t gi = i;
    iGlobal(&gi, i, level);
    const Cell* ci = &leaves[gi];

    Vector& Bi = B[i];
    int64_t ibegin = ci->BODY[0];
    int64_t m = ci->BODY[1] - ibegin;
    vectorCreate(&Bi, m);

    int64_t block = 500;
    int64_t last = nbodies % block;
    if (last > 0) {
      Vector X;
      vectorCreate(&X, last);
      for (int64_t k = 0; k < last; k++)
        X.X[k] = bodies[k].B;
      
      Matrix Aij;
      matrixCreate(&Aij, m, last);
      gen_matrix(ef, m, last, &bodies[ibegin], bodies, Aij.A, NULL, NULL);
      mvec('N', &Aij, &X, &Bi, 1., 0.);
      matrixDestroy(&Aij);
      vectorDestroy(&X);
    }
    else
      zeroVector(&Bi);

    for (int64_t j = last; j < nbodies; j += block) {
      Vector X;
      vectorCreate(&X, block);
      for (int64_t k = 0; k < block; k++)
        X.X[k] = bodies[k + j].B;
      
      Matrix Aij;
      matrixCreate(&Aij, m, block);
      gen_matrix(ef, m, block, &bodies[ibegin], &bodies[j], Aij.A, NULL, NULL);
      mvec('N', &Aij, &X, &Bi, 1., 1.);
      matrixDestroy(&Aij);
      vectorDestroy(&X);
    }
  }
}
