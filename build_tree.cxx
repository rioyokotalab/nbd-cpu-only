#include "nbd.hxx"
#include "profile.hxx"

#include <cassert>
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "string.h"

void buildTree(int64_t* ncells, struct Cell* cells, double* bodies, int64_t nbodies, int64_t levels) {
  int __mpi_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &__mpi_size);
  int64_t mpi_size = __mpi_size;

  struct Cell* root = &cells[0];
  root->Body[0] = 0;
  root->Body[1] = nbodies;
  root->Level = 0;
  root->Procs[0] = 0;
  root->Procs[1] = mpi_size;
  get_bounds(bodies, nbodies, root->R, root->C);

  int64_t len = 1;
  int64_t i = 0;
  while (i < len) {
    struct Cell* ci = &cells[i];
    ci->Child[0] = -1;
    ci->Child[1] = -1;

    if (ci->Level < levels) {
      int64_t sdim = 0;
      double maxR = ci->R[0];
      if (ci->R[1] > maxR)
      { sdim = 1; maxR = ci->R[1]; }
      if (ci->R[2] > maxR)
      { sdim = 2; maxR = ci->R[2]; }

      int64_t i_begin = ci->Body[0];
      int64_t i_end = ci->Body[1];
      int64_t nbody_i = i_end - i_begin;
      sort_bodies(&bodies[i_begin * 3], nbody_i, sdim);
      int64_t loc = i_begin + nbody_i / 2;

      struct Cell* c0 = &cells[len];
      struct Cell* c1 = &cells[len + 1];
      ci->Child[0] = len;
      ci->Child[1] = len + 2;
      len = len + 2;

      c0->Body[0] = i_begin;
      c0->Body[1] = loc;
      c1->Body[0] = loc;
      c1->Body[1] = i_end;
      
      c0->Level = ci->Level + 1;
      c1->Level = ci->Level + 1;
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

      get_bounds(&bodies[i_begin * 3], loc - i_begin, c0->R, c0->C);
      get_bounds(&bodies[loc * 3], i_end - loc, c1->R, c1->C);
    }
    i++;
  }
  *ncells = len;
}

void buildTreeBuckets(struct Cell* cells, const double* bodies, const int64_t buckets[], int64_t levels) {
  int64_t nleaf = (int64_t)1 << levels;
  int64_t count = 0;
  for (int64_t i = 0; i < nleaf; i++) {
    int64_t ci = i + nleaf - 1;
    cells[ci].Child[0] = -1;
    cells[ci].Child[1] = -1;
    cells[ci].Body[0] = count;
    cells[ci].Body[1] = count + buckets[i];
    cells[ci].Level = levels;
    get_bounds(&bodies[count * 3], buckets[i], cells[ci].R, cells[ci].C);
    count = count + buckets[i];
  }

  for (int64_t i = nleaf - 2; i >= 0; i--) {
    int64_t c0 = (i << 1) + 1;
    int64_t c1 = (i << 1) + 2;
    int64_t begin = cells[c0].Body[0];
    int64_t len = cells[c1].Body[1] - begin;
    cells[i].Child[0] = c0;
    cells[i].Child[1] = c0 + 2;
    cells[i].Body[0] = begin;
    cells[i].Body[1] = begin + len;
    cells[i].Level = cells[c0].Level - 1;
    get_bounds(&bodies[begin * 3], len, cells[i].R, cells[i].C);
  }

  int __mpi_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &__mpi_size);
  int64_t mpi_size = __mpi_size;
  cells[0].Procs[0] = 0;
  cells[0].Procs[1] = mpi_size;

  for (int64_t i = 0; i < nleaf - 1; i++) {
    struct Cell* ci = &cells[i];
    struct Cell* c0 = &cells[ci->Child[0]];
    struct Cell* c1 = c0 + 1;
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
    c0->Procs[0] = ci->Procs[0];
    c1->Procs[1] = ci->Procs[1];
  }
}

int admis_check(double theta, const double C1[], const double C2[], const double R1[], const double R2[]) {
  double dCi[3];
  dCi[0] = C1[0] - C2[0];
  dCi[1] = C1[1] - C2[1];
  dCi[2] = C1[2] - C2[2];

  dCi[0] = dCi[0] * dCi[0];
  dCi[1] = dCi[1] * dCi[1];
  dCi[2] = dCi[2] * dCi[2];

  double dRi[3];
  dRi[0] = R1[0] * R1[0];
  dRi[1] = R1[1] * R1[1];
  dRi[2] = R1[2] * R1[2];

  double dRj[3];
  dRj[0] = R2[0] * R2[0];
  dRj[1] = R2[1] * R2[1];
  dRj[2] = R2[2] * R2[2];

  double dC = dCi[0] + dCi[1] + dCi[2];
  double dR = (dRi[0] + dRi[1] + dRi[2] + dRj[0] + dRj[1] + dRj[2]) * theta;
  return (int)(dC > dR);
}

void getList(char NoF, int64_t* len, int64_t rels[], int64_t ncells, const struct Cell cells[], int64_t i, int64_t j, double theta) {
  const struct Cell* Ci = &cells[i];
  const struct Cell* Cj = &cells[j];
  int64_t ilevel = Ci->Level;
  int64_t jlevel = Cj->Level; 
  if (ilevel == jlevel) {
    int admis = admis_check(theta, Ci->C, Cj->C, Ci->R, Cj->R);
    int write_far = NoF == 'F' || NoF == 'f';
    int write_near = NoF == 'N' || NoF == 'n';
    if (admis ? write_far : write_near) {
      int64_t n = *len;
      rels[n] = i + j * ncells;
      *len = n + 1;
    }
    if (admis)
      return;
  }
  if (ilevel <= jlevel && Ci->Child[0] >= 0)
    for (int64_t k = Ci->Child[0]; k < Ci->Child[1]; k++)
      getList(NoF, len, rels, ncells, cells, k, j, theta);
  else if (jlevel <= ilevel && Cj->Child[0] >= 0)
    for (int64_t k = Cj->Child[0]; k < Cj->Child[1]; k++)
      getList(NoF, len, rels, ncells, cells, i, k, theta);
}

int comp_int_64(const void *a, const void *b) {
  int64_t c = *(int64_t*)a - *(int64_t*)b;
  return c < 0 ? -1 : (int)(c > 0);
}

void traverse(char NoF, struct CSC* rels, int64_t ncells, const struct Cell* cells, double theta) {
  rels->M = ncells;
  rels->N = ncells;
  int64_t* rel_arr = (int64_t*)malloc(sizeof(int64_t) * (ncells * ncells + ncells + 1));
  int64_t len = 0;
  getList(NoF, &len, &rel_arr[ncells + 1], ncells, cells, 0, 0, theta);

  if (len < ncells * ncells)
    rel_arr = (int64_t*)realloc(rel_arr, sizeof(int64_t) * (len + ncells + 1));
  int64_t* rel_rows = &rel_arr[ncells + 1];
  qsort(rel_rows, len, sizeof(int64_t), comp_int_64);
  rels->ColIndex = rel_arr;
  rels->RowIndex = rel_rows;

  int64_t loc = -1;
  for (int64_t i = 0; i < len; i++) {
    int64_t r = rel_rows[i];
    int64_t x = r / ncells;
    int64_t y = r - x * ncells;
    rel_rows[i] = y;
    while (x > loc)
      rel_arr[++loc] = i;
  }
  for (int64_t i = loc + 1; i <= ncells; i++)
    rel_arr[i] = len;
}

void csc_free(struct CSC* csc) {
  free(csc->ColIndex);
}

void get_level(int64_t* begin, int64_t* end, const struct Cell* cells, int64_t level, int64_t mpi_rank) {
  int64_t low = *begin;
  int64_t high = *end;
  while (low < high) {
    int64_t mid = low + (high - low) / 2;
    const struct Cell* c = &cells[mid];
    int64_t l = c->Level - level;
    int ri = (int)(mpi_rank < c->Procs[0]) - (int)(mpi_rank >= c->Procs[1]);
    int cmp = l < 0 ? -1 : (l > 0 ? 1 : (mpi_rank == -1 ? 0 : ri));
    low = cmp < 0 ? mid + 1 : low;
    high = cmp < 0 ? high : mid;
  }
  *begin = high;

  low = high;
  high = *end;
  while (low < high) {
    int64_t mid = low + (high - low) / 2;
    const struct Cell* c = &cells[mid];
    int64_t l = c->Level - level;
    int ri = (int)(mpi_rank < c->Procs[0]) - (int)(mpi_rank >= c->Procs[1]);
    int cmp = l < 0 ? -1 : (l > 0 ? 1 : (mpi_rank == -1 ? 0 : ri));
    low = cmp <= 0 ? mid + 1 : low;
    high = cmp <= 0 ? high : mid;
  }
  *end = low;
}

void lookupIJ(int64_t* ij, const struct CSC* rels, int64_t i, int64_t j) {
  if (j < 0 || j >= rels->N)
  { *ij = -1; return; }
  const int64_t* row = rels->RowIndex;
  int64_t jbegin = rels->ColIndex[j];
  int64_t jend = rels->ColIndex[j + 1];
  const int64_t* row_iter = &row[jbegin];
  while (row_iter != &row[jend] && *row_iter != i)
    row_iter = row_iter + 1;
  int64_t k = row_iter - row;
  *ij = (k < jend) ? k : -1;
}

void local_bodies(int64_t body[], int64_t ncells, const struct Cell cells[], int64_t levels) {
  int __mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &__mpi_rank);
  int64_t mpi_rank = __mpi_rank;
  int64_t ibegin = 0, iend = ncells;
  get_level(&ibegin, &iend, cells, levels, mpi_rank);
  body[0] = cells[ibegin].Body[0];
  body[1] = cells[iend - 1].Body[1];
}

void loadX(double* X, int64_t seg, const double Xbodies[], int64_t ncells, const struct Cell cells[], int64_t levels) {
  int __mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &__mpi_rank);
  int64_t mpi_rank = __mpi_rank;
  int64_t ibegin = 0, iend = ncells;
  get_level(&ibegin, &iend, cells, levels, mpi_rank);
  memset(X, 0, sizeof(double) * (iend - ibegin) * seg);

  for (int64_t i = 0; i < (iend - ibegin); i++) {
    int64_t b0 = cells[i + ibegin].Body[0];
    int64_t b1 = cells[i + ibegin].Body[1];
    for (int64_t j = 0; j < (b1 - b0); j++)
      X[i * seg + j] = Xbodies[j + b0];
  }
}

void relations(struct CSC rels[], int64_t ncells, const struct Cell* cells, const struct CSC* cellRel, int64_t levels, const struct CellComm* comm) {
  int __mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &__mpi_rank);
  int64_t mpi_rank = __mpi_rank;
  
  for (int64_t i = 0; i <= levels; i++) {
    int64_t jbegin = 0, jend = ncells;
    get_level(&jbegin, &jend, cells, i, -1);
    int64_t ibegin = jbegin, iend = jend;
    get_level(&ibegin, &iend, cells, i, mpi_rank);
    int64_t nodes = iend - ibegin;
    struct CSC* csc = &rels[i];

    csc->M = jend - jbegin;
    csc->N = nodes;
    int64_t ent_max = nodes * csc->M;
    int64_t* cols = (int64_t*)malloc(sizeof(int64_t) * (nodes + 1 + ent_max));
    int64_t* rows = &cols[nodes + 1];

    int64_t count = 0;
    for (int64_t j = 0; j < nodes; j++) {
      int64_t lc = ibegin + j;
      cols[j] = count;
      int64_t cbegin = cellRel->ColIndex[lc];
      int64_t ent = cellRel->ColIndex[lc + 1] - cbegin;
      for (int64_t k = 0; k < ent; k++) {
        rows[count + k] = cellRel->RowIndex[cbegin + k] - jbegin;
        i_local(&rows[count + k], &comm[i]);
      }
      count = count + ent;
    }

    if (count < ent_max)
      cols = (int64_t*)realloc(cols, sizeof(int64_t) * (nodes + 1 + count));
    cols[nodes] = count;
    csc->ColIndex = cols;
    csc->RowIndex = &cols[nodes + 1];
  }
}

void evalD(double(*func)(double), struct Matrix* D, int64_t ncells, const struct Cell* cells, const double* bodies, const struct CSC* rels, int64_t level) {
  int __mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &__mpi_rank);
  int64_t mpi_rank = __mpi_rank;
  
  int64_t jbegin = 0, jend = ncells;
  get_level(&jbegin, &jend, cells, level, -1);
  int64_t ibegin = jbegin, iend = jend;
  get_level(&ibegin, &iend, cells, level, mpi_rank);
  int64_t nodes = iend - ibegin;

#pragma omp parallel for
  for (int64_t i = 0; i < nodes; i++) {
    int64_t lc = ibegin + i;
    const struct Cell* ci = &cells[lc];
    int64_t nbegin = rels->ColIndex[lc];
    int64_t nlen = rels->ColIndex[lc + 1] - nbegin;
    const int64_t* ngbs = &rels->RowIndex[nbegin];
    int64_t x_begin = ci->Body[0];
    int64_t n = ci->Body[1] - x_begin;
    int64_t offsetD = nbegin - rels->ColIndex[ibegin];

    for (int64_t j = 0; j < nlen; j++) {
      int64_t lj = ngbs[j];
      const struct Cell* cj = &cells[lj];
      int64_t y_begin = cj->Body[0];
      int64_t m = cj->Body[1] - y_begin;
      gen_matrix(func, m, n, &bodies[y_begin * 3], &bodies[x_begin * 3], D[offsetD + j].A, D[offsetD + j].LDA);
    }
  }
}

void evalS(double(*func)(double), struct Matrix* S, const struct Base* basis, const struct CSC* rels, const struct CellComm* comm) {
  int64_t ibegin = 0;
  content_length(NULL, NULL, &ibegin, comm);
  int64_t seg = basis->dimS * 3;

#pragma omp parallel for
  for (int64_t x = 0; x < rels->N; x++) {
    int64_t n = basis->DimsLr[x + ibegin];

    for (int64_t yx = rels->ColIndex[x]; yx < rels->ColIndex[x + 1]; yx++) {
      int64_t y = rels->RowIndex[yx];
      int64_t m = basis->DimsLr[y];
      gen_matrix(func, m, n, &basis->M_cpu[y * seg], &basis->M_cpu[(x + ibegin) * seg], S[yx].A, S[yx].LDA);
      upper_tri_reflec_mult('L', 1, &basis->R[y], &S[yx]);
      upper_tri_reflec_mult('R', 1, &basis->R[x + ibegin], &S[yx]);
    }
  }
}
