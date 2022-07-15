
#include "nbd.h"

#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "string.h"

struct SampleBodies 
{ int64_t LTlen, *FarLens, *FarAvails, **FarBodies, *CloseLens, *CloseAvails, **CloseBodies, *SkeLens, **Skeletons; };

void buildSampleBodies(struct SampleBodies* sample, int64_t sp_max_far, int64_t sp_max_near, int64_t nbodies, int64_t ncells, const struct Cell* cells, 
const struct CSC* rels, const int64_t* lt_child, const struct Base* basis_lo, int64_t level) {
  int __mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &__mpi_rank);
  int64_t mpi_rank = __mpi_rank;
  const int64_t LEN_CHILD = 2;
  
  int64_t jbegin = 0, jend = ncells;
  get_level(&jbegin, &jend, cells, level, -1);
  int64_t ibegin = jbegin, iend = jend;
  get_level(&ibegin, &iend, cells, level, mpi_rank);
  int64_t nodes = iend - ibegin;
  int64_t* arr_ctrl = (int64_t*)malloc(sizeof(int64_t) * nodes * 5);
  int64_t** arr_list = (int64_t**)malloc(sizeof(int64_t*) * nodes * 3);
  sample->LTlen = nodes;
  sample->FarLens = arr_ctrl;
  sample->CloseLens = &arr_ctrl[nodes];
  sample->FarAvails = &arr_ctrl[nodes * 2];
  sample->CloseAvails = &arr_ctrl[nodes * 3];
  sample->SkeLens = &arr_ctrl[nodes * 4];
  sample->FarBodies = arr_list;
  sample->CloseBodies = &arr_list[nodes];
  sample->Skeletons = &arr_list[nodes * 2];

  int64_t count_f = 0, count_c = 0, count_s = 0;
  for (int64_t i = 0; i < nodes; i++) {
    int64_t li = ibegin + i;
    int64_t nbegin = rels->ColIndex[i];
    int64_t nlen = rels->ColIndex[i + 1] - nbegin;
    const int64_t* ngbs = &rels->RowIndex[nbegin];

    int64_t far_avail = nbodies;
    int64_t close_avail = 0;
    for (int64_t j = 0; j < nlen; j++) {
      int64_t lj = ngbs[j] + jbegin;
      const struct Cell* cj = &cells[lj];
      int64_t len = cj->Body[1] - cj->Body[0];
      far_avail = far_avail - len;
      if (lj != li)
        close_avail = close_avail + len;
    }

    int64_t lc = lt_child[i];
    int64_t ske_len = 0;
    if (basis_lo != NULL && lc >= 0)
      for (int64_t j = 0; j < LEN_CHILD; j++)
        ske_len = ske_len + basis_lo->DimsLr[lc + j];
    else
      ske_len = cells[li].Body[1] - cells[li].Body[0];

    int64_t far_len = sp_max_far < far_avail ? sp_max_far : far_avail;
    int64_t close_len = sp_max_near < close_avail ? sp_max_near : close_avail;
    arr_ctrl[i] = far_len;
    arr_ctrl[i + nodes] = close_len;
    arr_ctrl[i + nodes * 2] = far_avail;
    arr_ctrl[i + nodes * 3] = close_avail;
    arr_ctrl[i + nodes * 4] = ske_len;
    count_f = count_f + far_len;
    count_c = count_c + close_len;
    count_s = count_s + ske_len;
  }

  int64_t* arr_bodies = NULL;
  if ((count_f + count_c + count_s) > 0)
    arr_bodies = (int64_t*)malloc(sizeof(int64_t) * (count_f + count_c + count_s));
  const struct Cell* leaves = &cells[jbegin];
  count_s = count_f + count_c;
  count_c = count_f;
  count_f = 0;
  for (int64_t i = 0; i < nodes; i++) {
    int64_t nbegin = rels->ColIndex[i];
    int64_t nlen = rels->ColIndex[i + 1] - nbegin;
    const int64_t* ngbs = &rels->RowIndex[nbegin];

    int64_t* remote = &arr_bodies[count_f];
    int64_t* close = &arr_bodies[count_c];
    int64_t* skeleton = &arr_bodies[count_s];
    int64_t far_len = arr_ctrl[i];
    int64_t close_len = arr_ctrl[i + nodes];
    int64_t far_avail = arr_ctrl[i + nodes * 2];
    int64_t close_avail = arr_ctrl[i + nodes * 3];
    int64_t ske_len = arr_ctrl[i + nodes * 4];

    int64_t box_i = 0;
    int64_t s_lens = 0;
    int64_t ic = ngbs[box_i];
    int64_t offset_i = leaves[ic].Body[0];
    int64_t len_i = leaves[ic].Body[1] - offset_i;

    int64_t li = i + ibegin - jbegin;
    int64_t cpos = 0;
    while (cpos < nlen && ngbs[cpos] != li)
      cpos = cpos + 1;

    for (int64_t j = 0; j < far_len; j++) {
      int64_t loc = (int64_t)((double)(far_avail * j) / far_len);
      while (box_i < nlen && loc + s_lens >= offset_i) {
        s_lens = s_lens + len_i;
        box_i = box_i + 1;
        ic = box_i < nlen ? ngbs[box_i] : ic;
        offset_i = leaves[ic].Body[0];
        len_i = leaves[ic].Body[1] - offset_i;
      }
      remote[j] = loc + s_lens;
    }

    box_i = (int64_t)(cpos == 0);
    s_lens = 0;
    ic = box_i < nlen ? ngbs[box_i] : ic;
    offset_i = leaves[ic].Body[0];
    len_i = leaves[ic].Body[1] - offset_i;

    for (int64_t j = 0; j < close_len; j++) {
      int64_t loc = (int64_t)((double)(close_avail * j) / close_len);
      while (loc - s_lens >= len_i) {
        s_lens = s_lens + len_i;
        box_i = box_i + 1;
        box_i = box_i + (int64_t)(box_i == cpos);
        ic = ngbs[box_i];
        offset_i = leaves[ic].Body[0];
        len_i = leaves[ic].Body[1] - offset_i;
      }
      close[j] = loc + offset_i - s_lens;
    }

    int64_t lc = lt_child[i];
    int64_t sbegin = cells[i + ibegin].Body[0];
    if (basis_lo != NULL && lc >= 0)
      memcpy(skeleton, basis_lo->Multipoles + basis_lo->Offsets[lc], sizeof(int64_t) * ske_len);
    else
      for (int64_t j = 0; j < ske_len; j++)
        skeleton[j] = j + sbegin;

    arr_list[i] = remote;
    arr_list[i + nodes] = close;
    arr_list[i + nodes * 2] = skeleton;
    count_f = count_f + far_len;
    count_c = count_c + close_len;
    count_s = count_s + ske_len;
  }
}

void sampleBodies_free(struct SampleBodies* sample) {
  int64_t* data = sample->FarBodies[0];
  if (data)
    free(data);
  free(sample->FarLens);
  free(sample->FarBodies);
}

void allocBasis(struct Base* basis, int64_t levels, int64_t ncells, const struct Cell* cells, const struct CellComm* comm) {
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nodes = 0;
    content_length(&nodes, &comm[i]);
    basis[i].Ulen = nodes;
    int64_t* arr_i = (int64_t*)malloc(sizeof(int64_t) * (nodes * 4 + 1));
    basis[i].Lchild = arr_i;
    basis[i].Dims = &arr_i[nodes];
    basis[i].DimsLr = &arr_i[nodes * 2];
    basis[i].Offsets = &arr_i[nodes * 3];
    basis[i].Multipoles = NULL;
    int64_t ibegin = 0, iend = ncells;
    get_level(&ibegin, &iend, cells, i, -1);
    for (int64_t j = 0; j < nodes; j++) {
      int64_t gj = j;
      i_global(&gj, &comm[i]);
      int64_t coc = cells[ibegin + gj].Child;
      arr_i[j] = -1;
      if (coc >= 0) {
        int64_t cc = coc - iend;
        i_local(&cc, &comm[i + 1]);
        arr_i[j] = cc;
      }
    }

    struct Matrix* arr_m = (struct Matrix*)calloc(nodes * 3, sizeof(struct Matrix));
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
    if (basis[i].Multipoles)
      free(basis[i].Multipoles);
    free(basis[i].Lchild);
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
  int64_t mpi_rank = comm->Proc[2];
  int64_t pbegin = comm->Comms.ColIndex[mpi_rank];
  int64_t plen = comm->Comms.ColIndex[mpi_rank + 1] - pbegin;
  const int64_t* row = &comm->Comms.RowIndex[pbegin];

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
    MPI_Bcast(arr_xlen, xlen, MPI_INT64_T, 0, comm->Comm_share);
}

void dist_int_64(int64_t arr[], const int64_t offsets[], const struct CellComm* comm) {
  int64_t mpi_rank = comm->Proc[2];
  int64_t pbegin = comm->Comms.ColIndex[mpi_rank];
  int64_t plen = comm->Comms.ColIndex[mpi_rank + 1] - pbegin;
  const int64_t* row = &comm->Comms.RowIndex[pbegin];

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
    MPI_Bcast(arr, alen, MPI_INT64_T, 0, comm->Comm_share);
}

void dist_double(double* arr[], const struct CellComm* comm) {
  int64_t mpi_rank = comm->Proc[2];
  int64_t pbegin = comm->Comms.ColIndex[mpi_rank];
  int64_t plen = comm->Comms.ColIndex[mpi_rank + 1] - pbegin;
  const int64_t* row = &comm->Comms.RowIndex[pbegin];
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
    MPI_Bcast(data, alen, MPI_DOUBLE, 0, comm->Comm_share);
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
    double** mms = (double**)malloc(sizeof(double*) * (xlen + 1));

    struct SampleBodies samples;
    buildSampleBodies(&samples, sp_pts, sp_pts, nbodies, ncells, cells, &rel_near[l], &base_i->Lchild[ibegin], l == levels ? NULL : &basis[l + 1], l);

    for (int64_t i = 0; i < nodes; i++) {
      int64_t lc = base_i->Lchild[i + ibegin];

      int64_t len_s = samples.FarLens[i] + (samples.CloseLens[i] > 0 ? samples.SkeLens[i] : 0);
      struct Matrix S;
      matrixCreate(&S, samples.SkeLens[i], len_s);

      if (len_s > 0) {
        struct Matrix S_lr;
        if (samples.FarLens[i] > 0) {
          S_lr = (struct Matrix){ S.A, samples.SkeLens[i], samples.FarLens[i] };
          gen_matrix(ef, samples.SkeLens[i], samples.FarLens[i], bodies, bodies, S_lr.A, samples.Skeletons[i], samples.FarBodies[i]);
        }

        if (samples.CloseLens[i] > 0) {
          struct Matrix S_dn = (struct Matrix){ &S.A[samples.SkeLens[i] * samples.FarLens[i]], samples.SkeLens[i], samples.SkeLens[i] };
          struct Matrix S_dn_work;
          matrixCreate(&S_dn_work, samples.SkeLens[i], samples.CloseLens[i]);
          gen_matrix(ef, samples.SkeLens[i], samples.CloseLens[i], bodies, bodies, S_dn_work.A, samples.Skeletons[i], samples.CloseBodies[i]);
          mmult('N', 'T', &S_dn_work, &S_dn_work, &S_dn, 1., 0.);
          if (samples.FarLens[i] > 0)
            normalizeA(&S_dn, &S_lr);
          matrixDestroy(&S_dn_work);
        }
      }

      int64_t rank = mrank > 0 ? (mrank < len_s ? mrank : len_s) : len_s;
      double* mat = (double*)malloc(sizeof(double) * (samples.SkeLens[i] * samples.SkeLens[i] + rank * rank));
      int32_t* pa = (int32_t*)malloc(sizeof(int32_t) * samples.SkeLens[i]);
      struct Matrix U = { mat, samples.SkeLens[i], samples.SkeLens[i] };
      if (rank > 0)
        lraID(epi, &S, &U, pa, &rank);

      struct Matrix Q = { mat, samples.SkeLens[i], rank };
      struct Matrix R = { &mat[samples.SkeLens[i] * samples.SkeLens[i]], rank, rank };
      if (rank > 0) {
        if (lc >= 0)
          updateSubU(&Q, &(basis[l + 1].R)[lc], &(basis[l + 1].R)[lc + 1]);
        qr_full(&Q, &R);

        for (int64_t j = 0; j < rank; j++) {
          int64_t piv = (int64_t)pa[j] - 1;
          if (piv != j) 
          { int64_t c = samples.Skeletons[i][piv]; samples.Skeletons[i][piv] = samples.Skeletons[i][j]; samples.Skeletons[i][j] = c; }
        }
      }

      matrixDestroy(&S);
      mms[i + ibegin] = mat;
      free(pa);
      base_i->Dims[i + ibegin] = samples.SkeLens[i];
      base_i->DimsLr[i + ibegin] = rank;
    }
    dist_int_64_xlen(base_i->Dims, &comm[l]);
    dist_int_64_xlen(base_i->DimsLr, &comm[l]);

    int64_t count = 0;
    int64_t count_m = 0;
    int64_t* offsets = base_i->Offsets;
    for (int64_t i = 0; i < xlen; i++) {
      int64_t m = base_i->Dims[i];
      int64_t n = base_i->DimsLr[i];
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
      int64_t n = base_i->DimsLr[i + ibegin];
      if (n > 0)
        memcpy(&mps_comm[offset_i], samples.Skeletons[i], sizeof(int64_t) * n);
    }
    dist_int_64(mps_comm, base_i->Offsets, &comm[l]);

    double* mat_comm = NULL;
    if (count_m > 0)
      mat_comm = (double*)malloc(sizeof(int64_t) * count_m);
    double* mat_iter = mat_comm;
    for (int64_t i = 0; i < xlen; i++) {
      int64_t m = base_i->Dims[i];
      int64_t n = base_i->DimsLr[i];
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
    sampleBodies_free(&samples);
  }
}

void evalS(void(*ef)(double*), struct Matrix* S, const struct Base* basis, const struct Body* bodies, const struct CSC* rels, const struct CellComm* comm) {
  int64_t ibegin = 0, iend = 0;
  self_local_range(&ibegin, &iend, comm);
  int64_t lbegin = ibegin;
  i_global(&lbegin, comm);

  for (int64_t x = 0; x < rels->N; x++) {
    for (int64_t yx = rels->ColIndex[x]; yx < rels->ColIndex[x + 1]; yx++) {
      int64_t y = rels->RowIndex[yx];
      int64_t box_y = y;
      i_local(&box_y, comm);
      int64_t m = basis->DimsLr[box_y];
      int64_t n = basis->DimsLr[x + ibegin];
      int64_t* multipoles = basis->Multipoles;
      int64_t off_y = basis->Offsets[box_y];
      int64_t off_x = basis->Offsets[x + ibegin];
      gen_matrix(ef, m, n, bodies, bodies, S[yx].A, &multipoles[off_y], &multipoles[off_x]);
      rsr(&basis->R[box_y], &basis->R[x + ibegin], &S[yx]);
    }
  }
}

