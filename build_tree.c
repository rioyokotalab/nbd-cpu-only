
#include "build_tree.h"
#include "linalg.h"
#include "kernel.h"

#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "string.h"

void buildTree(int64_t* ncells, struct Cell* cells, struct Body* bodies, int64_t nbodies, int64_t levels) {
  int __mpi_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &__mpi_size);
  int64_t mpi_size = __mpi_size;

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
    if (admis)
      return;
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
  int64_t c = *(int64_t*)a - *(int64_t*)b;
  return c > 0 ? 1 : (c == 0 ? 0 : -1);
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
    while (x > loc)
      rel_arr[++loc] = i;
  }
  for (int64_t i = loc + 1; i <= ncells; i++)
    rel_arr[i] = len;
}

int comp_cell_lvl(const struct Cell* cell, int64_t level, int64_t mpi_rank) {
  int64_t l = cell->LEVEL - level;
  int ri = (int)(mpi_rank < cell->Procs[0]) - (int)(mpi_rank >= cell->Procs[1]);
  return l > 0 ? 1 : (l == 0 ? (mpi_rank == -1 ? 0 : ri) : -1);
}

void get_level(int64_t* begin, int64_t* end, const struct Cell* cells, int64_t level, int64_t mpi_rank) {
  int64_t low = *begin;
  int64_t high = *end;
  while (low < high) {
    int64_t mid = low + (high - low) / 2;
    if (comp_cell_lvl(&cells[mid], level, mpi_rank) < 0)
      low = mid + 1;
    else
      high = mid;
  }
  *begin = high;

  low = high;
  high = *end;
  while (low < high) {
    int64_t mid = low + (high - low) / 2;
    if (comp_cell_lvl(&cells[mid], level, mpi_rank) <= 0)
      low = mid + 1;
    else
      high = mid;
  }
  *end = low;
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

void buildComm(struct CellComm* comms, int64_t ncells, const struct Cell* cells, const struct CSC* cellFar, const struct CSC* cellNear, int64_t levels) {
  int __mpi_rank = 0, __mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &__mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &__mpi_size);
  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  int64_t mpi_rank = __mpi_rank;
  int64_t mpi_size = __mpi_size;
  int* ranks = (int*)malloc(sizeof(int) * mpi_size);

  for (int64_t i = 0; i <= levels; i++) {
    struct CellComm* comm_i = &comms[i];
    int64_t ibegin = 0, iend = ncells;
    get_level(&ibegin, &iend, cells, i, -1);
    int64_t len_i = iend - ibegin;

    int64_t nbegin = cellNear->COL_INDEX[ibegin];
    int64_t nlen = cellNear->COL_INDEX[iend] - nbegin;
    int64_t fbegin = cellFar->COL_INDEX[ibegin];
    int64_t flen = cellFar->COL_INDEX[iend] - fbegin;
    int64_t len_arr = flen + nlen + mpi_size * 3 + 1;

    int64_t* rel_arr = (int64_t*)malloc(sizeof(int64_t) * len_arr);
    int64_t* rel_rows = &rel_arr[mpi_size + 1];

    for (int64_t j = 0; j < len_i; j++) {
      int64_t j_c = ibegin + j;
      int64_t src = cells[j_c].Procs[0];
      int64_t kj_hi = cellFar->COL_INDEX[j_c + 1];
      for (int64_t kj = cellFar->COL_INDEX[j_c]; kj < kj_hi; kj++) {
        int64_t k = cellFar->ROW_INDEX[kj];
        int64_t tgt = cells[k].Procs[0];
        int64_t row_i = kj - fbegin;
        rel_rows[row_i] = tgt + src * mpi_size;
      }
    }

    for (int64_t j = 0; j < len_i; j++) {
      int64_t j_c = ibegin + j;
      int64_t src = cells[j_c].Procs[0];
      int64_t kj_hi = cellNear->COL_INDEX[j_c + 1];
      for (int64_t kj = cellNear->COL_INDEX[j_c]; kj < kj_hi; kj++) {
        int64_t k = cellNear->ROW_INDEX[kj];
        int64_t tgt = cells[k].Procs[0];
        int64_t row_i = kj - nbegin + flen;
        rel_rows[row_i] = tgt + src * mpi_size;
      }
    }

    struct CSC* csc_i = &comm_i->Comms;
    csc_i->M = mpi_size;
    csc_i->N = mpi_size;
    qsort(rel_rows, nlen + flen, sizeof(int64_t), comp_int_64);
    const int64_t* iter = unique_int_64(rel_rows, nlen + flen);
    int64_t len = iter - rel_rows;
    if (len < flen + nlen) {
      len_arr = len + mpi_size * 3 + 1;
      rel_arr = (int64_t*)realloc(rel_arr, sizeof(int64_t) * len_arr);
      rel_rows = &rel_arr[mpi_size + 1];
    }
    csc_i->COL_INDEX = rel_arr;
    csc_i->ROW_INDEX = rel_rows;

    int64_t loc = -1;
    for (int64_t j = 0; j < len; j++) {
      int64_t r = rel_rows[j];
      int64_t x = r / mpi_size;
      int64_t y = r - x * mpi_size;
      rel_rows[j] = y;
      while (x > loc)
        rel_arr[++loc] = j;
    }
    for (int64_t j = loc + 1; j <= mpi_size; j++)
      rel_arr[j] = len;

    comm_i->ProcBoxes = &rel_arr[len + mpi_size + 1];
    comm_i->ProcBoxesEnd = &rel_arr[len + mpi_size * 2 + 1];
    for (int64_t j = 0; j < mpi_size; j++) {
      int64_t jbegin = ibegin, jend = iend;
      get_level(&jbegin, &jend, cells, i, j);
      comm_i->ProcBoxes[j] = jbegin - ibegin;
      comm_i->ProcBoxesEnd[j] = jend - ibegin;

      if (j == mpi_rank && jbegin < jend) {
        const struct Cell* c = &cells[jbegin];
        comm_i->Proc[0] = c->Procs[0];
        comm_i->Proc[1] = c->Procs[1];
      }
    }

    comm_i->Comm_box = (MPI_Comm*)malloc(sizeof(MPI_Comm) * mpi_size);
    for (int64_t j = 0; j < mpi_size; j++) {
      int64_t jbegin = rel_arr[j];
      int64_t jlen = rel_arr[j + 1] - jbegin;
      if (jlen > 0) {
        const int64_t* row = &rel_rows[jbegin];
        for (int64_t k = 0; k < jlen; k++)
          ranks[k] = row[k];

        MPI_Group group_j;
        MPI_Group_incl(world_group, jlen, ranks, &group_j);
        MPI_Comm_create_group(MPI_COMM_WORLD, group_j, j, &comm_i->Comm_box[j]);
        MPI_Group_free(&group_j);
      }
      else
        comm_i->Comm_box[j] = MPI_COMM_NULL;
    }

    int64_t p = comm_i->Proc[0];
    int64_t lenp = comm_i->Proc[1] - p;
    for (int64_t i = 0; i < lenp; i++)
      ranks[i] = i + p;

    MPI_Group group_merge;
    MPI_Group_incl(world_group, lenp, ranks, &group_merge);
    MPI_Comm_create_group(MPI_COMM_WORLD, group_merge, mpi_size, &comm_i->Comm_merge);
    MPI_Group_free(&group_merge);
  }

  MPI_Group_free(&world_group);
  free(ranks);
}

void cellComm_free(struct CellComm* comms, int64_t levels) {
  int64_t mpi_size = comms->Comms.M;
  for (int64_t i = 0; i <= levels; i++) {
    for (int64_t j = 0; j < mpi_size; j++)
      if (comms[i].Comm_box[j] != MPI_COMM_NULL)
        MPI_Comm_free(&comms[i].Comm_box[j]);
    MPI_Comm_free(&comms[i].Comm_merge);
    free(comms[i].Comms.COL_INDEX);
    free(comms[i].Comm_box);
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

void i_local(int64_t* ilocal, int64_t iglobal, const struct CellComm* comm) {
  int64_t p = comm->Proc[0];
  const int64_t* ngbs = comm->Comms.ROW_INDEX;
  int64_t nbegin = comm->Comms.COL_INDEX[p];
  int64_t nend = comm->Comms.COL_INDEX[p + 1];
  const int64_t* ngbs_iter = &ngbs[nbegin];
  int64_t slen = 0;
  while (ngbs_iter != &ngbs[nend] && 
  comm->ProcBoxesEnd[*ngbs_iter] <= iglobal) {
    slen = slen + comm->ProcBoxesEnd[*ngbs_iter] - comm->ProcBoxes[*ngbs_iter];
    ngbs_iter = ngbs_iter + 1;
  }
  int64_t k = ngbs_iter - ngbs;
  if (k < nend)
    *ilocal = slen + iglobal - comm->ProcBoxes[*ngbs_iter];
  else
    *ilocal = -1;
}

void i_global(int64_t* iglobal, int64_t ilocal, const struct CellComm* comm) {
  int64_t p = comm->Proc[0];
  const int64_t* ngbs = comm->Comms.ROW_INDEX;
  int64_t nbegin = comm->Comms.COL_INDEX[p];
  int64_t nend = comm->Comms.COL_INDEX[p + 1];
  const int64_t* ngbs_iter = &ngbs[nbegin];
  while (ngbs_iter != &ngbs[nend] && 
  comm->ProcBoxesEnd[*ngbs_iter] <= (comm->ProcBoxes[*ngbs_iter] + ilocal)) {
    ilocal = ilocal - comm->ProcBoxesEnd[*ngbs_iter] + comm->ProcBoxes[*ngbs_iter];
    ngbs_iter = ngbs_iter + 1;
  }
  int64_t k = ngbs_iter - ngbs;
  if (0 <= ilocal && k < nend)
    *iglobal = comm->ProcBoxes[*ngbs_iter] + ilocal;
  else
    *iglobal = -1;
}

void self_local_range(int64_t* ibegin, int64_t* iend, const struct CellComm* comm) {
  int64_t p = comm->Proc[0];
  const int64_t* ngbs = comm->Comms.ROW_INDEX;
  int64_t nbegin = comm->Comms.COL_INDEX[p];
  int64_t nend = comm->Comms.COL_INDEX[p + 1];
  const int64_t* ngbs_iter = &ngbs[nbegin];
  int64_t slen = 0;
  while (ngbs_iter != &ngbs[nend] && *ngbs_iter != p) {
    slen = slen + comm->ProcBoxesEnd[*ngbs_iter] - comm->ProcBoxes[*ngbs_iter];
    ngbs_iter = ngbs_iter + 1;
  }
  int64_t k = ngbs_iter - ngbs;
  if (k < nend) {
    *ibegin = slen;
    *iend = slen + comm->ProcBoxesEnd[*ngbs_iter] - comm->ProcBoxes[*ngbs_iter];
  }
  else {
    *ibegin = -1;
    *iend = -1;
  }
}

void self_global_range(int64_t* ibegin, int64_t* iend, const struct CellComm* comm) {
  int64_t p = comm->Proc[0];
  const int64_t* ngbs = comm->Comms.ROW_INDEX;
  int64_t nbegin = comm->Comms.COL_INDEX[p];
  int64_t nend = comm->Comms.COL_INDEX[p + 1];
  const int64_t* ngbs_iter = &ngbs[nbegin];
  while (ngbs_iter != &ngbs[nend] && *ngbs_iter != p)
    ngbs_iter = ngbs_iter + 1;
  int64_t k = ngbs_iter - ngbs;
  if (k < nend) {
    *ibegin = comm->ProcBoxes[*ngbs_iter];
    *iend = comm->ProcBoxesEnd[*ngbs_iter];
  }
  else {
    *ibegin = -1;
    *iend = -1;
  }
}

void content_length(int64_t* len, const struct CellComm* comm) {
  int64_t p = comm->Proc[0];
  const int64_t* ngbs = comm->Comms.ROW_INDEX;
  int64_t nbegin = comm->Comms.COL_INDEX[p];
  int64_t nend = comm->Comms.COL_INDEX[p + 1];
  const int64_t* ngbs_iter = &ngbs[nbegin];
  int64_t slen = 0;
  while (ngbs_iter != &ngbs[nend]) {
    slen = slen + comm->ProcBoxesEnd[*ngbs_iter] - comm->ProcBoxes[*ngbs_iter];
    ngbs_iter = ngbs_iter + 1;
  }
  *len = slen;
}

void relations(struct CSC rels[], int64_t ncells, const struct Cell* cells, const struct CSC* cellRel, int64_t mpi_rank, int64_t levels) {
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
      int64_t cbegin = cellRel->COL_INDEX[lc];
      int64_t ent = cellRel->COL_INDEX[lc + 1] - cbegin;
      for (int64_t k = 0; k < ent; k++)
        rows[count + k] = cellRel->ROW_INDEX[cbegin + k] - jbegin;
      count = count + ent;
    }

    if (count < ent_max)
      cols = (int64_t*)realloc(cols, sizeof(int64_t) * (nodes + 1 + count));
    cols[nodes] = count;
    csc->COL_INDEX = cols;
    csc->ROW_INDEX = &cols[nodes + 1];
  }
}

void evaluate(char NoF, struct Matrix* d, void(*ef)(double*), int64_t ncells, const struct Cell* cells, const struct Body* bodies, const struct CSC* csc, int64_t mpi_rank, int64_t level) {
  int64_t jbegin = 0, jend = ncells;
  get_level(&jbegin, &jend, cells, level, -1);
  int64_t ibegin = jbegin, iend = jend;
  get_level(&ibegin, &iend, cells, level, mpi_rank);
  int64_t nodes = iend - ibegin;

  for (int64_t i = 0; i < nodes; i++) {
    int64_t lc = ibegin + i;
    const struct Cell* ci = &cells[lc];
    int64_t off = csc->COL_INDEX[i];
    int64_t len = csc->COL_INDEX[i + 1] - off;

    if (NoF == 'N' || NoF == 'n')
      for (int64_t j = 0; j < len; j++) {
        int64_t jj = csc->ROW_INDEX[j + off] + jbegin;
        const struct Cell* cj = &cells[jj];
        int64_t i_begin = cj->BODY[0];
        int64_t j_begin = ci->BODY[0];
        int64_t m = cj->BODY[1] - i_begin;
        int64_t n = ci->BODY[1] - j_begin;
        matrixCreate(&d[off + j], m, n);
        gen_matrix(ef, m, n, &bodies[i_begin], &bodies[j_begin], d[off + j].A, NULL, NULL);
      }
    else if (NoF == 'F' || NoF == 'f')
      for (int64_t j = 0; j < len; j++) {
        int64_t jj = csc->ROW_INDEX[j + off] + jbegin;
        const struct Cell* cj = &cells[jj];
        int64_t m = cj->lenMultipole;
        int64_t n = ci->lenMultipole;
        matrixCreate(&d[off + j], m, n);
        gen_matrix(ef, m, n, bodies, bodies, d[off + j].A, cj->Multipole, ci->Multipole);
      }
  }
}

void remoteBodies(void(*ef)(double*), struct Matrix* S, const int64_t cellm[], const int64_t size[], int64_t nlen, const int64_t ngbs[], const struct Cell* cells, int64_t ci, const struct Body* bodies) {
  int64_t rmsize = size[0];
  int64_t clsize = size[1];
  int64_t cmsize = size[2];
  int64_t nbodies = size[3];
  int64_t* remote = (int64_t*)malloc(sizeof(int64_t) * (rmsize + clsize));
  int64_t* close = &remote[rmsize];

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

  int64_t len_s = rmsize + (clsize > 0 ? cmsize : 0);
  matrixCreate(S, cmsize, len_s);

  if (len_s > 0) {
    struct Matrix S_lr;
    if (rmsize > 0) {
      S_lr = (struct Matrix){ S->A, cmsize, rmsize };
      gen_matrix(ef, cmsize, rmsize, bodies, bodies, S_lr.A, cellm, remote);
    }

    if (clsize > 0) {
      struct Matrix S_dn = (struct Matrix){ &S->A[cmsize * rmsize], cmsize, cmsize };
      struct Matrix S_dn_work;
      matrixCreate(&S_dn_work, cmsize, clsize);
      gen_matrix(ef, cmsize, clsize, bodies, bodies, S_dn_work.A, cellm, close);
      mmult('N', 'T', &S_dn_work, &S_dn_work, &S_dn, 1., 0.);
      if (rmsize > 0)
        normalizeA(&S_dn, &S_lr);
      matrixDestroy(&S_dn_work);
    }
  }
  
  free(remote);
}

void allocBasis(struct Base* basis, int64_t levels, int64_t ncells, const struct Cell* cells, const struct CellComm* comm) {
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nodes = 0;
    content_length(&nodes, &comm[i]);

    basis[i].Ulen = nodes;
    int64_t* arr_i = (int64_t*)malloc(sizeof(int64_t) * (nodes * 4));
    basis[i].Lchild = arr_i;
    basis[i].DIMS = &arr_i[nodes];
    basis[i].DIML = &arr_i[nodes * 2];
    basis[i].Offsets = &arr_i[nodes * 3];
    basis[i].Multipoles = NULL;

    int64_t ibegin = 0, iend = ncells;
    get_level(&ibegin, &iend, cells, i, -1);
    for (int64_t j = 0; j < nodes; j++) {
      int64_t gj = j;
      i_global(&gj, j, &comm[i]);
      const struct Cell* c = &cells[ibegin + gj];
      int64_t coc = c->CHILD;
      if (coc >= 0)
        i_local(&arr_i[j], coc - iend, &comm[i + 1]);
      else
        arr_i[j] = -1;
    }

    struct Matrix* arr_m = (struct Matrix*)malloc(sizeof(struct Matrix) * (nodes * 3));
    basis[i].Uo = arr_m;
    basis[i].Uc = &arr_m[nodes];
    basis[i].R = &arr_m[nodes * 2];
    basis[i].Comm = &comm[i];
  }
}

void deallocBasis(struct Base* basis, int64_t levels) {
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nodes = basis[i].Ulen;
    for (int64_t n = 0; n < nodes; n++) {
      matrixDestroy(&basis[i].Uo[n]);
      matrixDestroy(&basis[i].Uc[n]);
      matrixDestroy(&basis[i].R[n]);
    }

    basis[i].Ulen = 0;
    free(basis[i].Lchild);
    if (basis[i].Multipoles)
      free(basis[i].Multipoles);
    free(basis[i].Uo);
  }
}

void basis_mem(int64_t* bytes, const struct Base* basis, int64_t levels) {
  int64_t count = 0;
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nodes = basis[i].Ulen;
    int64_t bytes_o, bytes_c, bytes_r;
    matrix_mem(&bytes_o, &basis[i].Uo[0], nodes);
    matrix_mem(&bytes_c, &basis[i].Uc[0], nodes);
    matrix_mem(&bytes_r, &basis[i].R[0], nodes);

    count = count + bytes_o + bytes_c + bytes_r;
  }
  *bytes = count;
}

#include "dist.h"

void evaluateBaseAll(void(*ef)(double*), struct Base basis[], int64_t ncells, struct Cell* cells, const struct CSC* cellsNear, int64_t levels, const struct Body* bodies, int64_t nbodies, double epi, int64_t mrank, int64_t sp_pts) {
  for (int64_t l = levels; l >= 0; l--) {
    struct Base* base_i = basis + l;
    int64_t xlen = base_i->Ulen;
    int64_t jbegin = 0, jend = ncells;
    get_level(&jbegin, &jend, cells, l, -1);
    int64_t ibegin = 0, iend = xlen, gbegin;
    self_local_range(&ibegin, &iend, base_i->Comm);
    i_global(&gbegin, ibegin, base_i->Comm);
    int64_t nodes = iend - ibegin;
    struct Cell* leaves = &cells[jbegin];
    int64_t** cms = (int64_t**)malloc(sizeof(int64_t*) * nodes);

    for (int64_t i = 0; i < nodes; i++) {
      struct Cell* ci = &leaves[i + gbegin];
      int64_t box_i = i + ibegin;
      int64_t lc = base_i->Lchild[box_i];
      int64_t ni = 0;
      int64_t* cellm;

      if (lc >= 0) {
        int64_t len0 = basis[l + 1].DIML[lc];
        int64_t len1 = basis[l + 1].DIML[lc + 1];
        ni = len0 + len1;
        cellm = (int64_t*)malloc(sizeof(int64_t) * ni);

        int64_t offset = basis[l + 1].Offsets[lc];
        memcpy(cellm, &basis[l + 1].Multipoles[offset], sizeof(int64_t) * ni);
      }
      else {
        int64_t nbegin = ci->BODY[0];
        ni = ci->BODY[1] - nbegin;
        cellm = (int64_t*)malloc(sizeof(int64_t) * ni);
        for (int64_t j = 0; j < ni; j++)
          cellm[j] = nbegin + j;
      }
      cms[i] = cellm;
      
      int64_t rank = mrank;
      int64_t n[4] = { sp_pts, sp_pts, ni, nbodies };

      int64_t ii = ci - cells;
      int64_t lbegin = cellsNear->COL_INDEX[ii];
      int64_t llen = cellsNear->COL_INDEX[ii + 1] - lbegin;
      struct Matrix S;
      remoteBodies(ef, &S, cellm, n, llen, &cellsNear->ROW_INDEX[lbegin], cells, ii, bodies);

      struct Matrix work_u;
      int64_t* pa = (int64_t*)malloc(sizeof(int64_t) * ni);
      matrixCreate(&work_u, ni, ni);
      if (S.N > 0)
        lraID(epi, &S, &work_u, pa, &rank);
      else
        rank = 0;

      matrixCreate(&(base_i->Uo)[box_i], ni, rank);
      matrixCreate(&(base_i->Uc)[box_i], ni, ni - rank);
      matrixCreate(&(base_i->R)[box_i], rank, rank);
      cpyMatToMat(ni, rank, &work_u, &(base_i->Uo)[box_i], 0, 0, 0, 0);
      if (lc >= 0)
        updateSubU(&(base_i->Uo)[box_i], &(basis[l + 1].R)[lc], &(basis[l + 1].R)[lc + 1]);
      qr_with_complements(&(base_i->Uo)[box_i], &(base_i->Uc)[box_i], &(base_i->R)[box_i]);

      matrixDestroy(&S);
      matrixDestroy(&work_u);

      for (int64_t j = 0; j < rank; j++) {
        int64_t piv_i = pa[j] - 1;
        if (piv_i != j) {
          int64_t row_piv = cellm[piv_i];
          cellm[piv_i] = cellm[j];
          cellm[j] = row_piv;
        }
      }
      free(pa);
      base_i->DIMS[box_i] = ni;
      base_i->DIML[box_i] = rank;
    }

    DistributeDims(base_i->DIMS, l);
    DistributeDims(base_i->DIML, l);

    int64_t count = 0;
    int64_t* offsets = base_i->Offsets;
    for (int64_t i = 0; i < xlen; i++) {
      int64_t m = base_i->DIMS[i];
      int64_t n = base_i->DIML[i];
      offsets[i] = count;
      count = count + n;

      int64_t msize = m * n;
      if (msize > 0 && (i < ibegin || i >= iend)) {
        matrixCreate(&(base_i->Uo)[i], m, n);
        matrixCreate(&(base_i->Uc)[i], m, m - n);
        matrixCreate(&(base_i->R)[i], n, n);
      }
    }

    DistributeMatricesList(base_i->Uc, l);
    DistributeMatricesList(base_i->Uo, l);
    DistributeMatricesList(base_i->R, l);

    if (count > 0)
      base_i->Multipoles = (int64_t*)malloc(sizeof(int64_t) * count);
    int64_t* mps_comm = base_i->Multipoles;
    for (int64_t i = 0; i < nodes; i++) {
      int64_t offset_i = offsets[i + ibegin];
      int64_t n = base_i->DIML[i + ibegin];
      if (n > 0)
        memcpy(&mps_comm[offset_i], cms[i], sizeof(int64_t) * n);
      free(cms[i]);
    }
    free(cms);
    DistributeMultipoles(mps_comm, base_i->DIML, l);

    for (int64_t i = 0; i < xlen; i++) {
      int64_t gi = i;
      i_global(&gi, i, base_i->Comm);
      struct Cell* ci = &leaves[gi];

      int64_t mi = base_i->DIML[i];
      int64_t offset_i = offsets[i];
      ci->Multipole = &mps_comm[offset_i];
      ci->lenMultipole = mi;
    }
  }
}


void loadX(struct Matrix* X, const struct Cell* cell, const struct Body* bodies, int64_t level) {
  int64_t xlen = (int64_t)1 << level;
  contentLength(&xlen, level);
  int64_t len = (int64_t)1 << level;
  const struct Cell* leaves = &cell[len - 1];

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

void h2MatVecReference(struct Matrix* B, void(*ef)(double*), const struct Cell* cell, const struct Body* bodies, int64_t level) {
  int64_t nbodies = cell->BODY[1];
  int64_t xlen = (int64_t)1 << level;
  contentLength(&xlen, level);
  int64_t len = (int64_t)1 << level;
  const struct Cell* leaves = &cell[len - 1];

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
    free(ngbs);
  }
}
