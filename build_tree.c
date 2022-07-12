
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
  int64_t ilevel = Ci->LEVEL;
  int64_t jlevel = Cj->LEVEL; 
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
  if (ilevel <= jlevel && Ci->CHILD >= 0) {
    getList(NoF, len, rels, ncells, cells, Ci->CHILD, j, theta);
    getList(NoF, len, rels, ncells, cells, Ci->CHILD + 1, j, theta);
  }
  else if (jlevel <= ilevel && Cj->CHILD >= 0) {
    getList(NoF, len, rels, ncells, cells, i, Cj->CHILD, theta);
    getList(NoF, len, rels, ncells, cells, i, Cj->CHILD + 1, theta);
  }
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
  return l < 0 ? -1 : (l > 0 ? 1 : (mpi_rank == -1 ? 0 : ri));
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
    int64_t len_arr = flen + nlen + mpi_size * 4 + 1;

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
      len_arr = len + mpi_size * 4 + 1;
      rel_arr = (int64_t*)realloc(rel_arr, sizeof(int64_t) * len_arr);
      rel_rows = &rel_arr[mpi_size + 1];
    }
    csc_i->COL_INDEX = rel_arr;
    csc_i->ROW_INDEX = rel_rows;
    int64_t* root_i = &rel_arr[len + mpi_size + 1];
    memset(root_i, 0xFF, sizeof(int64_t) * mpi_size);

    int64_t loc = -1;
    for (int64_t j = 0; j < len; j++) {
      int64_t r = rel_rows[j];
      int64_t x = r / mpi_size;
      int64_t y = r - x * mpi_size;
      rel_rows[j] = y;
      while (x > loc)
        rel_arr[++loc] = j;
      if (y == x)
        root_i[x] = j - rel_arr[x];
    }
    for (int64_t j = loc + 1; j <= mpi_size; j++)
      rel_arr[j] = len;

    comm_i->ProcRootI = root_i;
    comm_i->ProcBoxes = &root_i[mpi_size];
    comm_i->ProcBoxesEnd = &root_i[mpi_size * 2];
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
    if (lenp > 1) {
      for (int64_t i = 0; i < lenp; i++)
        ranks[i] = i + p;

      MPI_Group group_merge;
      MPI_Group_incl(world_group, lenp, ranks, &group_merge);
      MPI_Comm_create_group(MPI_COMM_WORLD, group_merge, mpi_size, &comm_i->Comm_merge);
      MPI_Group_free(&group_merge);
    }
    else
      comm_i->Comm_merge = MPI_COMM_NULL;
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
    if (comms[i].Comm_merge != MPI_COMM_NULL)
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

void i_local(int64_t* ilocal, const struct CellComm* comm) {
  int64_t iglobal = *ilocal;
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

void i_global(int64_t* iglobal, const struct CellComm* comm) {
  int64_t ilocal = *iglobal;
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

void relations(struct CSC rels[], int64_t ncells, const struct Cell* cells, const struct CSC* cellRel, int64_t levels) {
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

void allocBasis(struct Base* basis, int64_t levels, int64_t ncells, const struct Cell* cells, const struct CellComm* comm) {
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nodes = 0;
    content_length(&nodes, &comm[i]);

    basis[i].Ulen = nodes;
    int64_t* arr_i = (int64_t*)malloc(sizeof(int64_t) * (nodes * 4 + 1));
    basis[i].Lchild = arr_i;
    basis[i].DIMS = &arr_i[nodes];
    basis[i].DIML = &arr_i[nodes * 2];
    basis[i].Offsets = &arr_i[nodes * 3];
    basis[i].Multipoles = NULL;

    int64_t ibegin = 0, iend = ncells;
    get_level(&ibegin, &iend, cells, i, -1);
    for (int64_t j = 0; j < nodes; j++) {
      int64_t gj = j;
      i_global(&gj, &comm[i]);
      const struct Cell* c = &cells[ibegin + gj];
      int64_t coc = c->CHILD;
      if (coc >= 0) {
        int64_t cc = coc - iend;
        i_local(&cc, &comm[i + 1]);
        arr_i[j] = cc;
      }
      else
        arr_i[j] = -1;
    }

    struct Matrix* arr_m = (struct Matrix*)malloc(sizeof(struct Matrix) * (nodes * 3));
    basis[i].Uo = arr_m;
    basis[i].Uc = &arr_m[nodes];
    basis[i].R = &arr_m[nodes * 2];
  }
}

void deallocBasis(struct Base* basis, int64_t levels) {
  for (int64_t i = 0; i <= levels; i++) {
    double* data = basis[i].Uo[0].A;
    if (data)
      free(data);

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

void dist_int_64_xlen(int64_t arr_xlen[], const struct CellComm* comm) {
  int __mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &__mpi_rank);
  int64_t mpi_rank = __mpi_rank;
  int64_t pbegin = comm->Comms.COL_INDEX[mpi_rank];
  int64_t plen = comm->Comms.COL_INDEX[mpi_rank + 1] - pbegin;
  const int64_t* row = &comm->Comms.ROW_INDEX[pbegin];

  for (int64_t i = 0; i < plen; i++) {
    int64_t p = row[i];
    int64_t lbegin = comm->ProcBoxes[p];
    int64_t len = comm->ProcBoxesEnd[p] - lbegin;
    i_local(&lbegin, comm);
    MPI_Bcast(&arr_xlen[lbegin], len, MPI_INT64_T, comm->ProcRootI[p], comm->Comm_box[p]);
  }

  int64_t xlen = 0;
  content_length(&xlen, comm);
  if (comm->Proc[1] - comm->Proc[0] > 1)
    MPI_Bcast(arr_xlen, xlen, MPI_INT64_T, 0, comm->Comm_merge);
}

void dist_int_64(int64_t arr[], const int64_t offsets[], const struct CellComm* comm) {
  int __mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &__mpi_rank);
  int64_t mpi_rank = __mpi_rank;
  int64_t pbegin = comm->Comms.COL_INDEX[mpi_rank];
  int64_t plen = comm->Comms.COL_INDEX[mpi_rank + 1] - pbegin;
  const int64_t* row = &comm->Comms.ROW_INDEX[pbegin];

  for (int64_t i = 0; i < plen; i++) {
    int64_t p = row[i];
    int64_t lbegin = comm->ProcBoxes[p];
    int64_t llen = comm->ProcBoxesEnd[p] - lbegin;
    i_local(&lbegin, comm);
    int64_t offset = offsets[lbegin];
    int64_t len = offsets[lbegin + llen] - offset;
    MPI_Bcast(&arr[offset], len, MPI_INT64_T, comm->ProcRootI[p], comm->Comm_box[p]);
  }

  int64_t xlen = 0;
  content_length(&xlen, comm);
  int64_t alen = offsets[xlen];
  if (comm->Proc[1] - comm->Proc[0] > 1)
    MPI_Bcast(arr, alen, MPI_INT64_T, 0, comm->Comm_merge);
}

void dist_double(double* arr[], const struct CellComm* comm) {
  int __mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &__mpi_rank);
  int64_t mpi_rank = __mpi_rank;
  int64_t pbegin = comm->Comms.COL_INDEX[mpi_rank];
  int64_t plen = comm->Comms.COL_INDEX[mpi_rank + 1] - pbegin;
  const int64_t* row = &comm->Comms.ROW_INDEX[pbegin];
  double* data = arr[0];

  for (int64_t i = 0; i < plen; i++) {
    int64_t p = row[i];
    int64_t lbegin = comm->ProcBoxes[p];
    int64_t llen = comm->ProcBoxesEnd[p] - lbegin;
    i_local(&lbegin, comm);
    int64_t offset = arr[lbegin] - data;
    int64_t len = arr[lbegin + llen] - arr[lbegin];
    MPI_Bcast(&data[offset], len, MPI_DOUBLE, comm->ProcRootI[p], comm->Comm_box[p]);
  }

  int64_t xlen = 0;
  content_length(&xlen, comm);
  int64_t alen = arr[xlen] - data;
  if (comm->Proc[1] - comm->Proc[0] > 1)
    MPI_Bcast(data, alen, MPI_DOUBLE, 0, comm->Comm_merge);
}

void evaluateBaseAll(void(*ef)(double*), struct Base basis[], int64_t ncells, struct Cell* cells, const struct CSC* rel_near, int64_t levels, 
const struct CellComm* comm, const struct Body* bodies, int64_t nbodies, double epi, int64_t mrank, int64_t sp_pts) {

  for (int64_t l = levels; l >= 0; l--) {
    struct Base* base_i = basis + l;
    int64_t xlen = base_i->Ulen;
    int64_t jbegin = 0, jend = ncells;
    get_level(&jbegin, &jend, cells, l, -1);
    int64_t ibegin = 0, iend = xlen;
    self_local_range(&ibegin, &iend, &comm[l]);
    int64_t gbegin = ibegin;
    i_global(&gbegin, &comm[l]);
    int64_t nodes = iend - ibegin;
    struct Cell* leaves = &cells[jbegin];
    int64_t** cms = (int64_t**)malloc(sizeof(int64_t*) * nodes);
    double** mms = (double**)malloc(sizeof(double*) * (xlen + 1));

    for (int64_t i = 0; i < nodes; i++) {
      struct Cell* ci = &leaves[i + gbegin];
      int64_t lc = base_i->Lchild[i + ibegin];
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
      
      int64_t lbegin = rel_near[l].COL_INDEX[i];
      int64_t nlen = rel_near[l].COL_INDEX[i + 1] - lbegin;
      const int64_t* ngbs = &rel_near[l].ROW_INDEX[lbegin];

      int64_t rmsize = sp_pts;
      int64_t clsize = sp_pts;

      int64_t rm_len = nbodies;
      int64_t cl_len = 0;
      int64_t cpos = -1;
      for (int64_t j = 0; j < nlen; j++) {
        int64_t jc = ngbs[j];
        const struct Cell* c = &leaves[jc];
        int64_t len = c->BODY[1] - c->BODY[0];
        rm_len = rm_len - len;
        if (jc == i + gbegin)
          cpos = j;
        else
          cl_len = cl_len + len;
      }

      rmsize = rmsize > rm_len ? rm_len : rmsize;
      clsize = clsize > cl_len ? cl_len : clsize;
      int64_t* remote = (int64_t*)malloc(sizeof(int64_t) * (rmsize + clsize));
      int64_t* close = &remote[rmsize];

      int64_t box_i = 0;
      int64_t s_lens = 0;
      int64_t ic = ngbs[box_i];
      int64_t offset_i = leaves[ic].BODY[0];
      int64_t len_i = leaves[ic].BODY[1] - offset_i;

      for (int64_t j = 0; j < rmsize; j++) {
        int64_t loc = (int64_t)((double)(rm_len * j) / rmsize);
        while (box_i < nlen && loc + s_lens >= offset_i) {
          s_lens = s_lens + len_i;
          box_i = box_i + 1;
          ic = box_i < nlen ? ngbs[box_i] : ic;
          offset_i = leaves[ic].BODY[0];
          len_i = leaves[ic].BODY[1] - offset_i;
        }
        remote[j] = loc + s_lens;
      }

      box_i = (int64_t)(cpos == 0);
      s_lens = 0;
      ic = box_i < nlen ? ngbs[box_i] : ic;
      offset_i = leaves[ic].BODY[0];
      len_i = leaves[ic].BODY[1] - offset_i;

      for (int64_t j = 0; j < clsize; j++) {
        int64_t loc = (int64_t)((double)(cl_len * j) / clsize);
        while (loc - s_lens >= len_i) {
          s_lens = s_lens + len_i;
          box_i = box_i + 1;
          box_i = box_i + (int64_t)(box_i == cpos);
          ic = ngbs[box_i];
          offset_i = leaves[ic].BODY[0];
          len_i = leaves[ic].BODY[1] - offset_i;
        }
        close[j] = loc + offset_i - s_lens;
      }

      int64_t len_s = rmsize + (clsize > 0 ? ni : 0);
      struct Matrix S;
      matrixCreate(&S, ni, len_s);

      if (len_s > 0) {
        struct Matrix S_lr;
        if (rmsize > 0) {
          S_lr = (struct Matrix){ S.A, ni, rmsize };
          gen_matrix(ef, ni, rmsize, bodies, bodies, S_lr.A, cellm, remote);
        }

        if (clsize > 0) {
          struct Matrix S_dn = (struct Matrix){ &S.A[ni * rmsize], ni, ni };
          struct Matrix S_dn_work;
          matrixCreate(&S_dn_work, ni, clsize);
          gen_matrix(ef, ni, clsize, bodies, bodies, S_dn_work.A, cellm, close);
          mmult('N', 'T', &S_dn_work, &S_dn_work, &S_dn, 1., 0.);
          if (rmsize > 0)
            normalizeA(&S_dn, &S_lr);
          matrixDestroy(&S_dn_work);
        }
      }
      
      free(remote);

      int64_t rank = mrank > 0 ? (mrank < len_s ? mrank : len_s) : len_s;
      double* mat = (double*)malloc(sizeof(double) * (ni * ni + rank * rank));
      int64_t* pa = (int64_t*)malloc(sizeof(int64_t) * ni);
      struct Matrix U = { mat, ni, ni };
      if (rank > 0)
        lraID(epi, &S, &U, pa, &rank);

      struct Matrix Uo = { mat, ni, rank };
      struct Matrix Uc = { &mat[ni * rank], ni, ni - rank };
      struct Matrix R = { &mat[ni * ni], rank, rank };
      if (rank > 0) {
        if (lc >= 0)
          updateSubU(&Uo, &(basis[l + 1].R)[lc], &(basis[l + 1].R)[lc + 1]);
        qr_with_complements(&Uo, &Uc, &R);

        for (int64_t j = 0; j < rank; j++) {
          int64_t piv = pa[j] - 1;
          if (piv != j) 
          { int64_t c = cellm[piv]; cellm[piv] = cellm[j]; cellm[j] = c; }
        }
      }

      matrixDestroy(&S);
      mms[i + ibegin] = mat;
      free(pa);
      base_i->DIMS[i + ibegin] = ni;
      base_i->DIML[i + ibegin] = rank;
    }

    dist_int_64_xlen(base_i->DIMS, &comm[l]);
    dist_int_64_xlen(base_i->DIML, &comm[l]);

    int64_t count = 0;
    int64_t count_m = 0;
    int64_t* offsets = base_i->Offsets;
    for (int64_t i = 0; i < xlen; i++) {
      int64_t m = base_i->DIMS[i];
      int64_t n = base_i->DIML[i];
      offsets[i] = count;
      count = count + n;
      count_m = count_m + m * m + n * n;
    }
    offsets[xlen] = count;

    int64_t* mps_comm = NULL;
    if (count > 0)
      mps_comm = (int64_t*)malloc(sizeof(int64_t) * count);
    base_i->Multipoles = mps_comm;
    for (int64_t i = 0; i < nodes; i++) {
      int64_t offset_i = offsets[i + ibegin];
      int64_t n = base_i->DIML[i + ibegin];
      if (n > 0)
        memcpy(&mps_comm[offset_i], cms[i], sizeof(int64_t) * n);
      free(cms[i]);
    }
    free(cms);
    dist_int_64(mps_comm, base_i->Offsets, &comm[l]);

    double* mat_comm = NULL;
    if (count_m > 0)
      mat_comm = (double*)malloc(sizeof(int64_t) * count_m);
    double* mat_iter = mat_comm;
    for (int64_t i = 0; i < xlen; i++) {
      int64_t m = base_i->DIMS[i];
      int64_t n = base_i->DIML[i];
      if (ibegin <= i && i < iend) {
        memcpy(mat_iter, mms[i], sizeof(double) * (m * m + n * n));
        free(mms[i]);
      }
      base_i->Uo[i] = (struct Matrix) { mat_iter, m, n };
      base_i->Uc[i] = (struct Matrix) { &mat_iter[m * n], m, m - n };
      base_i->R[i] = (struct Matrix) { &mat_iter[m * m], n, n };
      mms[i] = mat_iter;
      mat_iter = &mat_iter[m * m + n * n];
    }
    mms[xlen] = mat_iter;
    dist_double(mms, &comm[l]);
    free(mms);

    for (int64_t i = 0; i < xlen; i++) {
      int64_t gi = i;
      i_global(&gi, &comm[l]);
      struct Cell* ci = &leaves[gi];

      int64_t mi = base_i->DIML[i];
      int64_t offset_i = offsets[i];
      ci->Multipole = &mps_comm[offset_i];
      ci->lenMultipole = mi;
    }
  }
}

void evaluate(char NoF, struct Matrix* d, void(*ef)(double*), int64_t ncells, const struct Cell* cells, const struct Body* bodies, const struct CSC* csc, int64_t level) {
  int __mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &__mpi_rank);
  int64_t mpi_rank = __mpi_rank;
  
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

void solveRelErr(double* err_out, const struct Matrix* X, const struct Matrix* ref, const struct CellComm* comm) {
  int64_t ibegin = 0, iend = 0;
  self_local_range(&ibegin, &iend, comm);
  double err = 0.;
  double nrm = 0.;

  for (int64_t i = ibegin; i < iend; i++) {
    double e, n;
    struct Matrix work;
    matrixCreate(&work, X[i].M, X[i].N);

    maxpby(&work, X[i].A, 1., 0.);
    maxpby(&work, ref[i].A, -1., 1.);
    mnrm2(&work, &e);
    mnrm2(&ref[i], &n);

    matrixDestroy(&work);
    err = err + e * e;
    nrm = nrm + n * n;
  }

  double buf = err;
  MPI_Allreduce(&buf, &err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  buf = nrm;
  MPI_Allreduce(&buf, &nrm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  *err_out = sqrt(err / nrm);
}

