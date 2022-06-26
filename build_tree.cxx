
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

int comp_int_64(const void *a, const void *b) {
  return *(int64_t*)a - *(int64_t*)b;
}

void traverse(char NoF, CSC* rels, int64_t ncells, const Cell* cells, double theta) {
  rels->M = ncells;
  rels->N = ncells;
  int64_t* rel_arr = (int64_t*)malloc(sizeof(int64_t) * (ncells * ncells + ncells + 1));
  int64_t len = 0;
  getList(NoF, &len, &rel_arr[ncells + 1], ncells, cells, 0, 0, 0, 0, theta);

  if (len < ncells * ncells)
    rel_arr = (int64_t*)realloc(rel_arr, sizeof(int64_t) * (len + ncells + 1));
  int64_t* rel_rows = &rel_arr[ncells + 1];
  qsort(rel_rows, len, sizeof(int64_t), comp_int_64);
  rels->COL_INDEX = rel_arr;
  rels->ROW_INDEX = rel_rows;

  int64_t loc = -1;
  for (int64_t i = 0; i < len; i++) {
    int64_t r = rel_rows[i];
    int64_t x = r / ncells;
    int64_t y = r - x * ncells;
    rel_rows[i] = y;
    while (x > loc) {
      loc = loc + 1;
      rel_arr[loc] = i;
    }
  }
  for (int64_t i = loc + 1; i <= ncells; i++)
    rel_arr[i] = len;
}

void traverse_dist(const CSC* cellFar, const CSC* cellNear, int64_t levels) {
  int64_t mpi_rank, mpi_levels;
  commRank(&mpi_rank, &mpi_levels);

  configureComm(levels, NULL, 0);
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nodes = i > mpi_levels ? (int64_t)1 << (i - mpi_levels) : 1;
    int64_t lvl_diff = i < mpi_levels ? mpi_levels - i : 0;
    int64_t my_rank = mpi_rank >> lvl_diff;
    int64_t gbegin = my_rank * nodes;

    int64_t offc = (int64_t)(1 << i) - 1;
    std::vector<int64_t> ngbs;

    for (int64_t n = 0; n < nodes; n++) {
      int64_t nc = offc + n + gbegin;
      int64_t nbegin = cellNear->COL_INDEX[nc];
      int64_t nlen = cellNear->COL_INDEX[nc + 1] - nbegin;
      for (int64_t j = 0; j < nlen; j++) {
        int64_t ngb = cellNear->ROW_INDEX[nbegin + j] - offc;
        ngb /= nodes;
        ngbs.emplace_back(ngb);
      }
      int64_t fbegin = cellFar->COL_INDEX[nc];
      int64_t flen = cellFar->COL_INDEX[nc + 1] - fbegin;
      for (int64_t j = 0; j < flen; j++) {
        int64_t ngb = cellFar->ROW_INDEX[fbegin + j] - offc;
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


void relations(CSC rels[], const CSC* cellRel, int64_t levels) {
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
    int64_t count = 0;
    for (int64_t j = 0; j < nodes; j++) {
      int64_t lc = offc + lbegin + j;
      cols[j] = count;
      int64_t cbegin = cellRel->COL_INDEX[lc];
      int64_t ent = cellRel->COL_INDEX[lc + 1] - cbegin;
      for (int64_t k = 0; k < ent; k++) {
        int64_t zi = cellRel->ROW_INDEX[cbegin + k] - offc;
        rows[count + k] = zi;
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
  int64_t ibegin = 0, iend = 0, lbegin = 0;;
  selfLocalRange(&ibegin, &iend, level);
  iGlobal(&lbegin, ibegin, level);
  int64_t nodes = iend - ibegin;
  int64_t offc = (int64_t)(1 << level) - 1;

#pragma omp parallel for
  for (int64_t i = 0; i < nodes; i++) {
    int64_t lc = offc + lbegin + i;
    const Cell* ci = &cell[lc];
    int64_t off = csc->COL_INDEX[i];
    int64_t len = csc->COL_INDEX[i + 1] - off;

    if (NoF == 'N' || NoF == 'n') {
      for (int64_t j = 0; j < len; j++) {
        int64_t jj = csc->ROW_INDEX[j + off] + offc;
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
      for (int64_t j = 0; j < len; j++) {
        int64_t jj = csc->ROW_INDEX[j + off] + offc;
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


void loadX(Matrix* X, const Cell* cell, const Body* bodies, int64_t level) {
  int64_t xlen = (int64_t)1 << level;
  contentLength(&xlen, level);
  int64_t len = (int64_t)1 << level;
  const Cell* leaves = &cell[len - 1];

#pragma omp parallel for
  for (int64_t i = 0; i < xlen; i++) {
    int64_t gi = i;
    iGlobal(&gi, i, level);
    const Cell* ci = &leaves[gi];

    Matrix& Xi = X[i];
    int64_t nbegin = ci->BODY[0];
    int64_t ni = ci->BODY[1] - nbegin;
    matrixCreate(&Xi, ni, 1);
    for (int64_t n = 0; n < ni; n++)
      Xi.A[n] = bodies[n + nbegin].B;
  }
}

void h2MatVecReference(Matrix* B, KerFunc_t ef, const Cell* cell, const Body* bodies, int64_t level) {
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

    Matrix& Bi = B[i];
    int64_t ibegin = ci->BODY[0];
    int64_t m = ci->BODY[1] - ibegin;
    matrixCreate(&Bi, m, 1);

    int64_t block = 500;
    int64_t last = nbodies % block;
    Matrix X;
    Matrix Aij;
    matrixCreate(&X, block, 1);
    matrixCreate(&Aij, m, block);
    zeroMatrix(&X);
    zeroMatrix(&Aij);

    if (last > 0) {
      for (int64_t k = 0; k < last; k++)
        X.A[k] = bodies[k].B;
      gen_matrix(ef, m, last, &bodies[ibegin], bodies, Aij.A, NULL, NULL);
      mmult('N', 'N', &Aij, &X, &Bi, 1., 0.);
    }
    else
      zeroMatrix(&Bi);

    for (int64_t j = last; j < nbodies; j += block) {
      for (int64_t k = 0; k < block; k++)
        X.A[k] = bodies[k + j].B;
      gen_matrix(ef, m, block, &bodies[ibegin], &bodies[j], Aij.A, NULL, NULL);
      mmult('N', 'N', &Aij, &X, &Bi, 1., 1.);
    }

    matrixDestroy(&Aij);
    matrixDestroy(&X);
  }
}
