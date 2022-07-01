
#include "build_tree.h"
#include "dist.h"

#include "stdlib.h"
#include "math.h"

void buildTree(int64_t* ncells, struct Cell* cells, struct Body* bodies, int64_t nbodies, int64_t levels, int64_t mpi_size) {
  struct Cell* root = &cells[0];
  root->BODY[0] = 0;
  root->BODY[1] = nbodies;
  root->LEVEL = 0;
  root->Procs[0] = 0;
  root->Procs[1] = mpi_size;
  get_bounds(bodies, nbodies, root->R, root->C);

  int64_t len = 1;
  int64_t i = 0;
  while (i < len) {
    struct Cell* ci = &cells[i];
    ci->CHILD = -1;

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

      struct Cell* c0 = &cells[len];
      struct Cell* c1 = &cells[len + 1];
      ci->CHILD = len;
      len = len + 2;

      c0->BODY[0] = i_begin;
      c0->BODY[1] = loc;
      c1->BODY[0] = loc;
      c1->BODY[1] = i_end;
      
      c0->LEVEL = ci->LEVEL + 1;
      c1->LEVEL = ci->LEVEL + 1;
      c0->Procs[0] = ci->Procs[0];
      c1->Procs[1] = ci->Procs[1];

      int64_t divp = (ci->Procs[1] - ci->Procs[0]) / 2;
      if (divp >= 1) {
        int64_t p = divp + ci->Procs[0];
        c0->Procs[1] = p;
        c1->Procs[0] = p;
      }
      else {
        c0->Procs[1] = ci->Procs[1];
        c1->Procs[0] = ci->Procs[0];
      }

      get_bounds(&bodies[i_begin], loc - i_begin, c0->R, c0->C);
      get_bounds(&bodies[loc], i_end - loc, c1->R, c1->C);
    }
    i++;
  }
  *ncells = len;
}

void getList(char NoF, int64_t* len, int64_t rels[], int64_t ncells, const struct Cell cells[], int64_t i, int64_t j, int64_t ilevel, int64_t jlevel, double theta) {
  const struct Cell* Ci = &cells[i];
  const struct Cell* Cj = &cells[j];
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

void traverse(char NoF, struct CSC* rels, int64_t ncells, const struct Cell* cells, double theta) {
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

void get_level(int64_t i[], int64_t ncells, const struct Cell* cells, int64_t level) {

}

int64_t* unique_int_64(int64_t* arr, int64_t len) {
  int64_t* last = &arr[len];
  if (arr == last)
    return last;
  int64_t* result = arr;
  while (++arr != last)
    if (!(*result == *arr) && ++result != arr)
      *result = *arr;
  return ++result;
}

void traverse_dist(const struct CSC* cellFar, const struct CSC* cellNear, int64_t levels) {
  int64_t mpi_rank, mpi_levels;
  commRank(&mpi_rank, &mpi_levels);

  configureComm(levels, NULL, 0);
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nodes = i > mpi_levels ? (int64_t)1 << (i - mpi_levels) : 1;
    int64_t lvl_diff = i < mpi_levels ? mpi_levels - i : 0;
    int64_t my_rank = mpi_rank >> lvl_diff;
    int64_t gbegin = my_rank * nodes;

    int64_t offc = (int64_t)(1 << i) - 1;
    int64_t nc = offc + gbegin;
    int64_t nbegin = cellNear->COL_INDEX[nc];
    int64_t nlen = cellNear->COL_INDEX[nc + nodes] - nbegin;
    int64_t fbegin = cellFar->COL_INDEX[nc];
    int64_t flen = cellFar->COL_INDEX[nc + nodes] - fbegin;
    int64_t* ngbs = (int64_t*)malloc(sizeof(int64_t) * (nlen + flen));

    for (int64_t j = 0; j < nlen; j++) {
      int64_t ngb = cellNear->ROW_INDEX[nbegin + j] - offc;
      ngb /= nodes;
      ngbs[j] = ngb;
    }
    for (int64_t j = 0; j < flen; j++) {
      int64_t ngb = cellFar->ROW_INDEX[fbegin + j] - offc;
      ngb /= nodes;
      ngbs[j + nlen] = ngb;
    }

    qsort(ngbs, nlen + flen, sizeof(int64_t), comp_int_64);
    const int64_t* iter = unique_int_64(&ngbs[0], nlen + flen);
    int64_t size = iter - ngbs;
    configureComm(i, &ngbs[0], size);
  }
}


void relations(struct CSC rels[], const struct CSC* cellRel, int64_t levels) {
  for (int64_t i = 0; i <= levels; i++) {
    int64_t ibegin = 0, iend = 0, lbegin = 0;
    selfLocalRange(&ibegin, &iend, i);
    iGlobal(&lbegin, ibegin, i);
    int64_t nodes = iend - ibegin;
    struct CSC* csc = &rels[i];

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

void evaluate(char NoF, struct Matrix* d, KerFunc_t ef, const struct Cell* cell, const struct Body* bodies, const struct CSC* csc, int64_t level) {
  int64_t ibegin = 0, iend = 0, lbegin = 0;;
  selfLocalRange(&ibegin, &iend, level);
  iGlobal(&lbegin, ibegin, level);
  int64_t nodes = iend - ibegin;
  int64_t offc = (int64_t)(1 << level) - 1;

#pragma omp parallel for
  for (int64_t i = 0; i < nodes; i++) {
    int64_t lc = offc + lbegin + i;
    const struct Cell* ci = &cell[lc];
    int64_t off = csc->COL_INDEX[i];
    int64_t len = csc->COL_INDEX[i + 1] - off;

    if (NoF == 'N' || NoF == 'n') {
      for (int64_t j = 0; j < len; j++) {
        int64_t jj = csc->ROW_INDEX[j + off] + offc;
        const struct Cell* cj = &cell[jj];
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
        const struct Cell* cj = &cell[jj];
        int64_t m = cj->lenMultipole;
        int64_t n = ci->lenMultipole;
        matrixCreate(&d[off + j], m, n);
        gen_matrix(ef, m, n, bodies, bodies, d[off + j].A, cj->Multipole, ci->Multipole);
      }
    }
  }
}

void lookupIJ(int64_t* ij, const struct CSC* rels, int64_t i, int64_t j) {
  if (j < 0 || j >= rels->N)
  { *ij = -1; return; }
  const int64_t* row = rels->ROW_INDEX;
  int64_t jbegin = rels->COL_INDEX[j];
  int64_t jend = rels->COL_INDEX[j + 1];
  const int64_t* row_iter = &row[jbegin];
  while (row_iter != &row[jend] && *row_iter != i)
    row_iter = row_iter + 1;
  int64_t k = row_iter - row;
  *ij = (k < jend) ? k : -1;
}

void remoteBodies(int64_t* remote, int64_t size[], int64_t nlen, const int64_t ngbs[], const struct Cell* cells, int64_t ci) {
  int64_t rmsize = size[0];
  int64_t clsize = size[1];
  int64_t nbodies = size[2];

  int64_t sum_len = 0;
  int64_t c_len = 0;
  int64_t cpos = -1;
  for (int64_t j = 0; j < nlen; j++) {
    int64_t jc = ngbs[j];
    const struct Cell* c = &cells[jc];
    int64_t len = c->BODY[1] - c->BODY[0];
    sum_len = sum_len + len;
    if (jc == ci) {
      c_len = len;
      cpos = j;
    }
  }

  int64_t rm_len = nbodies - sum_len;
  rmsize = rmsize > rm_len ? rm_len : rmsize;

  int64_t box_i = 0;
  int64_t s_lens = 0;
  int64_t ic = ngbs[box_i];
  int64_t offset_i = cells[ic].BODY[0];
  int64_t len_i = cells[ic].BODY[1] - offset_i;

  for (int64_t i = 0; i < rmsize; i++) {
    int64_t loc = (int64_t)((double)(rm_len * i) / rmsize);
    while (box_i < nlen && loc + s_lens >= offset_i) {
      s_lens = s_lens + len_i;
      box_i = box_i + 1;
      ic = box_i < nlen ? ngbs[box_i] : ic;
      offset_i = cells[ic].BODY[0];
      len_i = cells[ic].BODY[1] - offset_i;
    }
    remote[i] = loc + s_lens;
  }

  int64_t* close = &remote[rmsize];
  int64_t cl_len = sum_len - c_len;
  clsize = clsize > cl_len ? cl_len : clsize;

  box_i = (int64_t)(cpos == 0);
  s_lens = 0;
  ic = box_i < nlen ? ngbs[box_i] : ic;
  offset_i = cells[ic].BODY[0];
  len_i = cells[ic].BODY[1] - offset_i;

  for (int64_t i = 0; i < clsize; i++) {
    int64_t loc = (int64_t)((double)(cl_len * i) / clsize);
    while (loc - s_lens >= len_i) {
      s_lens = s_lens + len_i;
      box_i = box_i + 1;
      box_i = box_i + (int64_t)(box_i == cpos);
      ic = ngbs[box_i];
      offset_i = cells[ic].BODY[0];
      len_i = cells[ic].BODY[1] - offset_i;
    }
    close[i] = loc + offset_i - s_lens;
  }

  size[0] = rmsize;
  size[1] = clsize;
}

void evaluateBasis(KerFunc_t ef, double epi, int64_t* rank, struct Matrix* Base, int64_t m, int64_t n[], int64_t cellm[], const int64_t remote[], const struct Body* bodies) {
  int64_t n1 = n[0];
  int64_t n2 = n[1];
  int64_t len_s = n1 + (n2 > 0 ? m : 0);
  if (len_s > 0) {
    const int64_t* close = &remote[n1];
    struct Matrix S, S_lr;
    matrixCreate(&S, m, len_s);

    if (n1 > 0) {
      S_lr = (struct Matrix){ S.A, m, n1 };
      gen_matrix(ef, m, n1, bodies, bodies, S_lr.A, cellm, remote);
    }

    if (n2 > 0) {
      struct Matrix S_dn = (struct Matrix){ &S.A[m * n1], m, m };
      struct Matrix S_dn_work;
      matrixCreate(&S_dn_work, m, n2);
      gen_matrix(ef, m, n2, bodies, bodies, S_dn_work.A, cellm, close);
      mmult('N', 'T', &S_dn_work, &S_dn_work, &S_dn, 1., 0.);
      if (n1 > 0)
        normalizeA(&S_dn, &S_lr);
      matrixDestroy(&S_dn_work);
    }

    int64_t mrank = *rank;
    mrank = mrank > m ? m : mrank;
    int64_t un = mrank > 0 ? mrank : m;
    struct Matrix work_u;
    int64_t* pa = (int64_t*)malloc(sizeof(int64_t) * un);
    matrixCreate(&work_u, m, un);

    int64_t iters = un;
    lraID(epi, &S, &work_u, pa, &iters);

    matrixCreate(Base, m, iters);
    cpyMatToMat(m, iters, &work_u, Base, 0, 0, 0, 0);
    matrixDestroy(&S);
    matrixDestroy(&work_u);

    for (int64_t i = 0; i < iters; i++) {
      int64_t piv_i = pa[i] - 1;
      if (piv_i != i) {
        int64_t row_piv = cellm[piv_i];
        cellm[piv_i] = cellm[i];
        cellm[i] = row_piv;
      }
    }
    *rank = iters;
    free(pa);
  }
}


void loadX(struct Matrix* X, const struct Cell* cell, const struct Body* bodies, int64_t level) {
  int64_t xlen = (int64_t)1 << level;
  contentLength(&xlen, level);
  int64_t len = (int64_t)1 << level;
  const struct Cell* leaves = &cell[len - 1];

#pragma omp parallel for
  for (int64_t i = 0; i < xlen; i++) {
    int64_t gi = i;
    iGlobal(&gi, i, level);
    const struct Cell* ci = &leaves[gi];

    struct Matrix* Xi = &X[i];
    int64_t nbegin = ci->BODY[0];
    int64_t ni = ci->BODY[1] - nbegin;
    matrixCreate(Xi, ni, 1);
    for (int64_t n = 0; n < ni; n++)
      Xi->A[n] = bodies[n + nbegin].B;
  }
}

void h2MatVecReference(struct Matrix* B, KerFunc_t ef, const struct Cell* cell, const struct Body* bodies, int64_t level) {
  int64_t nbodies = cell->BODY[1];
  int64_t xlen = (int64_t)1 << level;
  contentLength(&xlen, level);
  int64_t len = (int64_t)1 << level;
  const struct Cell* leaves = &cell[len - 1];

#pragma omp parallel for
  for (int64_t i = 0; i < xlen; i++) {
    int64_t gi = i;
    iGlobal(&gi, i, level);
    const struct Cell* ci = &leaves[gi];

    struct Matrix* Bi = &B[i];
    int64_t ibegin = ci->BODY[0];
    int64_t m = ci->BODY[1] - ibegin;
    matrixCreate(Bi, m, 1);

    int64_t block = 500;
    int64_t last = nbodies % block;
    struct Matrix X;
    struct Matrix Aij;
    matrixCreate(&X, block, 1);
    matrixCreate(&Aij, m, block);
    zeroMatrix(&X);
    zeroMatrix(&Aij);

    if (last > 0) {
      for (int64_t k = 0; k < last; k++)
        X.A[k] = bodies[k].B;
      gen_matrix(ef, m, last, &bodies[ibegin], bodies, Aij.A, NULL, NULL);
      mmult('N', 'N', &Aij, &X, Bi, 1., 0.);
    }
    else
      zeroMatrix(Bi);

    for (int64_t j = last; j < nbodies; j += block) {
      for (int64_t k = 0; k < block; k++)
        X.A[k] = bodies[k + j].B;
      gen_matrix(ef, m, block, &bodies[ibegin], &bodies[j], Aij.A, NULL, NULL);
      mmult('N', 'N', &Aij, &X, Bi, 1., 1.);
    }

    matrixDestroy(&Aij);
    matrixDestroy(&X);
  }
}
