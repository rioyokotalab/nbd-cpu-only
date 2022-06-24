
#include "build_tree.h"
#include "basis.h"
#include "dist.h"

#include "stdlib.h"
#include "math.h"
#include <algorithm>

void buildTree(Cell* cells, Body* bodies, int64_t nbodies, int64_t levels) {
  int64_t nleaves = (int64_t)1 << levels;
  int64_t ncells = nleaves + nleaves - 1;

  Cell* root = &cells[0];
  root->BODY[0] = 0;
  root->BODY[1] = nbodies;
  get_bounds(bodies, nbodies, root->R, root->C);

  for (int64_t i = 0; i < ncells; i++) {
    Cell* ci = &cells[i];
    ci->CHILD = -1;

    if (i < nleaves - 1) {
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
      ci->CHILD = (i << 1) + 1;

      c0->BODY[0] = i_begin;
      c0->BODY[1] = loc;
      c1->BODY[0] = loc;
      c1->BODY[1] = i_end;

      get_bounds(&bodies[i_begin], loc - i_begin, c0->R, c0->C);
      get_bounds(&bodies[loc], i_end - loc, c1->R, c1->C);
    }
  }
}

void getList(char NoF, int64_t* len, int64_t rels[], int64_t ncells, const Cell cells[], int64_t i, int64_t j, int64_t ilevel, int64_t jlevel, double theta) {
  const Cell* Ci = &cells[i];
  const Cell* Cj = &cells[j];
  if (ilevel == jlevel) {
    int admis;
    admis_check(&admis, theta, Ci->C, Cj->C, Ci->R, Cj->R);
    int write_far = NoF == 'F' || NoF == 'f';
    int write_near = NoF == 'N' || NoF == 'n';
    if (admis ? write_far : write_near) {
      int64_t n = *len;
      rels[n] = i + j * ncells;
      *len = n + 1;
    }
  }
  if (ilevel <= jlevel && Ci->CHILD >= 0) {
    getList(NoF, len, rels, ncells, cells, Ci->CHILD, j, ilevel + 1, jlevel, theta);
    getList(NoF, len, rels, ncells, cells, Ci->CHILD + 1, j, ilevel + 1, jlevel, theta);
  }
  else if (jlevel <= ilevel && Cj->CHILD >= 0) {
    getList(NoF, len, rels, ncells, cells, i, Cj->CHILD, ilevel, jlevel + 1, theta);
    getList(NoF, len, rels, ncells, cells, i, Cj->CHILD + 1, ilevel, jlevel + 1, theta);
  }
}

void traverse(Cell* cells, int64_t levels, double theta) {
  int64_t nleaves = (int64_t)1 << levels;
  int64_t ncells = nleaves + nleaves - 1;

  std::vector<int64_t> rels(ncells * ncells);
  int64_t len = 0;
  getList('N', &len, &rels[0], ncells, &cells[0], 0, 0, 0, 0, theta);
  std::sort(&rels[0], &rels[len]);

  for (int64_t i = 0; i < len; i++) {
    int64_t r = rels[i];
    int64_t x = r / ncells;
    int64_t y = r - x * ncells;
    cells[y].listNear.emplace_back(x);
  }

  len = 0;
  getList('F', &len, &rels[0], ncells, &cells[0], 0, 0, 0, 0, theta);
  std::sort(&rels[0], &rels[len]);

  for (int64_t i = 0; i < len; i++) {
    int64_t r = rels[i];
    int64_t x = r / ncells;
    int64_t y = r - x * ncells;
    cells[y].listFar.emplace_back(x);
  }

  int64_t mpi_rank, mpi_levels;
  commRank(&mpi_rank, &mpi_levels);

  configureComm(levels, NULL, 0);
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nodes = i > mpi_levels ? (int64_t)1 << (i - mpi_levels) : 1;
    int64_t lvl_diff = i < mpi_levels ? mpi_levels - i : 0;
    int64_t my_rank = mpi_rank >> lvl_diff;
    int64_t gbegin = my_rank * nodes;

    int64_t offc = (int64_t)(1 << i) - 1;
    Cell* leaves = &cells[offc];
    std::vector<int64_t> ngbs;

    for (int64_t n = 0; n < nodes; n++) {
      const Cell* c = &leaves[n + gbegin];
      int64_t nlen = c->listNear.size();
      for (int64_t j = 0; j < nlen; j++) {
        int64_t ngb = c->listNear[j] - offc;
        ngb /= nodes;
        ngbs.emplace_back(ngb);
      }
      int64_t flen = c->listFar.size();
      for (int64_t j = 0; j < flen; j++) {
        int64_t ngb = c->listFar[j] - offc;
        ngb /= nodes;
        ngbs.emplace_back(ngb);
      }
    }

    std::sort(ngbs.begin(), ngbs.end());
    std::vector<int64_t>::iterator iter = std::unique(ngbs.begin(), ngbs.end());
    int64_t size = std::distance(ngbs.begin(), iter);
    configureComm(i, &ngbs[0], size);
  }
}


void relations(char NoF, CSC rels[], const Cell* cells, int64_t levels) {
  for (int64_t i = 0; i <= levels; i++) {
    int64_t ibegin = 0, iend = 0, lbegin = 0;
    selfLocalRange(&ibegin, &iend, i);
    iGlobal(&lbegin, ibegin, i);
    int64_t nodes = iend - ibegin;
    CSC* csc = &rels[i];

    csc->M = (int64_t)1 << i;
    csc->N = nodes;
    int64_t ent_max = nodes * csc->M;
    int64_t* cols = (int64_t*)malloc(sizeof(int64_t) * (nodes + 1 + ent_max));
    int64_t* rows = &cols[nodes + 1];

    int64_t offc = (int64_t)(1 << i) - 1;
    const Cell* leaves = &cells[offc];

    int64_t count = 0;
    for (int64_t j = 0; j < nodes; j++) {
      const Cell* c = &leaves[j + lbegin];
      int64_t ent = 0;
      cols[j] = count;
      if (NoF == 'N' || NoF == 'n') {
        ent = c->listNear.size();
        for (int64_t k = 0; k < ent; k++) {
          int64_t zi = c->listNear[k] - offc;
          rows[count + k] = zi;
        }
      }
      else if (NoF == 'F' || NoF == 'f') {
        ent = c->listFar.size();
        for (int64_t k = 0; k < ent; k++) {
          int64_t zi = c->listFar[k] - offc;
          rows[count + k] = zi;
        }
      }
      count = count + ent;
    }

    if (count < ent_max)
      cols = (int64_t*)realloc(cols, sizeof(int64_t) * (nodes + 1 + count));
    cols[nodes] = count;
    csc->COL_INDEX = cols;
    csc->ROW_INDEX = &cols[nodes + 1];
  }
}

void evaluate(char NoF, Matrix* d, KerFunc_t ef, const Cell* cell, const Body* bodies, const CSC* csc, int64_t level) {
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
    int64_t off = csc->COL_INDEX[i];

    if (NoF == 'N' || NoF == 'n') {
      int64_t len = ci->listNear.size();
      for (int64_t j = 0; j < len; j++) {
        int64_t jj = ci->listNear[j];
        const Cell* cj = &cell[jj];
        int64_t i_begin = cj->BODY[0];
        int64_t j_begin = ci->BODY[0];
        int64_t m = cj->BODY[1] - i_begin;
        int64_t n = ci->BODY[1] - j_begin;
        matrixCreate(&d[off + j], m, n);
        gen_matrix(ef, m, n, &bodies[i_begin], &bodies[j_begin], d[off + j].A, NULL, NULL);
      }
    }
    else if (NoF == 'F' || NoF == 'f') {
      int64_t len = ci->listFar.size();
      for (int64_t j = 0; j < len; j++) {
        int64_t jj = ci->listFar[j];
        const Cell* cj = &cell[jj];
        int64_t m = cj->Multipole.size();
        int64_t n = ci->Multipole.size();
        matrixCreate(&d[off + j], m, n);
        gen_matrix(ef, m, n, bodies, bodies, d[off + j].A, cj->Multipole.data(), ci->Multipole.data());
      }
    }
  }
}

void lookupIJ(int64_t* ij, const CSC* rels, int64_t i, int64_t j) {
  if (j < 0 || j >= rels->N)
  { *ij = -1; return; }
  const int64_t* row = rels->ROW_INDEX;
  int64_t jbegin = rels->COL_INDEX[j];
  int64_t jend = rels->COL_INDEX[j + 1];
  int64_t k = std::distance(row, std::find(&row[jbegin], &row[jend], i));
  *ij = (k < jend) ? k : -1;
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
    Vector X;
    Matrix Aij;
    vectorCreate(&X, block);
    matrixCreate(&Aij, m, block);
    zeroVector(&X);
    zeroMatrix(&Aij);

    if (last > 0) {
      for (int64_t k = 0; k < last; k++)
        X.X[k] = bodies[k].B;
      gen_matrix(ef, m, last, &bodies[ibegin], bodies, Aij.A, NULL, NULL);
      mvec('N', &Aij, &X, &Bi, 1., 0.);
    }
    else
      zeroVector(&Bi);

    for (int64_t j = last; j < nbodies; j += block) {
      for (int64_t k = 0; k < block; k++)
        X.X[k] = bodies[k + j].B;
      gen_matrix(ef, m, block, &bodies[ibegin], &bodies[j], Aij.A, NULL, NULL);
      mvec('N', &Aij, &X, &Bi, 1., 1.);
    }

    matrixDestroy(&Aij);
    vectorDestroy(&X);
  }
}
