
#include "nbd.h"
#include "profile.h"

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
  sample->SkeLens = &arr_ctrl[nodes * 2];
  sample->FarAvails = &arr_ctrl[nodes * 3];
  sample->CloseAvails = &arr_ctrl[nodes * 4];
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
    arr_ctrl[i + nodes * 2] = ske_len;
    arr_ctrl[i + nodes * 3] = far_avail;
    arr_ctrl[i + nodes * 4] = close_avail;
    count_f = count_f + far_len;
    count_c = count_c + close_len;
    count_s = count_s + ske_len;
  }

  int64_t* arr_bodies = NULL;
  if ((count_f + count_c + count_s) > 0)
    arr_bodies = (int64_t*)malloc(sizeof(int64_t) * (count_f + count_c + count_s));
  count_s = count_f + count_c;
  count_c = count_f;
  count_f = 0;
  for (int64_t i = 0; i < nodes; i++) {
    int64_t* remote = &arr_bodies[count_f];
    int64_t* close = &arr_bodies[count_c];
    int64_t* skeleton = &arr_bodies[count_s];
    int64_t far_len = arr_ctrl[i];
    int64_t close_len = arr_ctrl[i + nodes];
    int64_t ske_len = arr_ctrl[i + nodes * 2];
    arr_list[i] = remote;
    arr_list[i + nodes] = close;
    arr_list[i + nodes * 2] = skeleton;
    count_f = count_f + far_len;
    count_c = count_c + close_len;
    count_s = count_s + ske_len;
  }

#pragma omp parallel for
  for (int64_t i = 0; i < nodes; i++) {
    int64_t nbegin = rels->ColIndex[i];
    int64_t nlen = rels->ColIndex[i + 1] - nbegin;
    const int64_t* ngbs = &rels->RowIndex[nbegin];

    int64_t* remote = arr_list[i];
    int64_t* close = arr_list[i + nodes];
    int64_t* skeleton = arr_list[i + nodes * 2];
    int64_t far_len = arr_ctrl[i];
    int64_t close_len = arr_ctrl[i + nodes];
    int64_t ske_len = arr_ctrl[i + nodes * 2];
    int64_t far_avail = arr_ctrl[i + nodes * 3];
    int64_t close_avail = arr_ctrl[i + nodes * 4];

    int64_t box_i = 0;
    int64_t s_lens = 0;
    int64_t ic = jbegin + ngbs[box_i];
    int64_t offset_i = cells[ic].Body[0];
    int64_t len_i = cells[ic].Body[1] - offset_i;

    int64_t li = i + ibegin - jbegin;
    int64_t cpos = 0;
    while (cpos < nlen && ngbs[cpos] != li)
      cpos = cpos + 1;

    for (int64_t j = 0; j < far_len; j++) {
      int64_t loc = (int64_t)((double)(far_avail * j) / far_len);
      while (box_i < nlen && loc + s_lens >= offset_i) {
        s_lens = s_lens + len_i;
        box_i = box_i + 1;
        ic = box_i < nlen ? (jbegin + ngbs[box_i]) : ic;
        offset_i = cells[ic].Body[0];
        len_i = cells[ic].Body[1] - offset_i;
      }
      remote[j] = loc + s_lens;
    }

    box_i = (int64_t)(cpos == 0);
    s_lens = 0;
    ic = box_i < nlen ? (jbegin + ngbs[box_i]) : ic;
    offset_i = cells[ic].Body[0];
    len_i = cells[ic].Body[1] - offset_i;

    for (int64_t j = 0; j < close_len; j++) {
      int64_t loc = (int64_t)((double)(close_avail * j) / close_len);
      while (loc - s_lens >= len_i) {
        s_lens = s_lens + len_i;
        box_i = box_i + 1;
        box_i = box_i + (int64_t)(box_i == cpos);
        ic = jbegin + ngbs[box_i];
        offset_i = cells[ic].Body[0];
        len_i = cells[ic].Body[1] - offset_i;
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
  }
}

void sampleBodies_free(struct SampleBodies* sample) {
  int64_t* data = sample->FarBodies[0];
  if (data)
    free(data);
  free(sample->FarLens);
  free(sample->FarBodies);
}

void dist_int_64_xlen(int64_t arr_xlen[], const struct CellComm* comm) {
  int64_t mpi_rank = comm->Proc[2];
  int64_t pbegin = comm->Comms.ColIndex[mpi_rank];
  int64_t plen = comm->Comms.ColIndex[mpi_rank + 1] - pbegin;
  const int64_t* row = &comm->Comms.RowIndex[pbegin];
#ifdef _PROF
  double stime = MPI_Wtime();
#endif
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
#ifdef _PROF
  double etime = MPI_Wtime() - stime;
  recordCommTime(etime);
#endif
}

void dist_int_64(int64_t arr[], const int64_t offsets[], const struct CellComm* comm) {
  int64_t mpi_rank = comm->Proc[2];
  int64_t pbegin = comm->Comms.ColIndex[mpi_rank];
  int64_t plen = comm->Comms.ColIndex[mpi_rank + 1] - pbegin;
  const int64_t* row = &comm->Comms.RowIndex[pbegin];
#ifdef _PROF
  double stime = MPI_Wtime();
#endif
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
#ifdef _PROF
  double etime = MPI_Wtime() - stime;
  recordCommTime(etime);
#endif
}

void dist_double(double* arr[], const struct CellComm* comm) {
  int64_t mpi_rank = comm->Proc[2];
  int64_t pbegin = comm->Comms.ColIndex[mpi_rank];
  int64_t plen = comm->Comms.ColIndex[mpi_rank + 1] - pbegin;
  const int64_t* row = &comm->Comms.RowIndex[pbegin];
  double* data = arr[0];
#ifdef _PROF
  double stime = MPI_Wtime();
#endif
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
#ifdef _PROF
  double etime = MPI_Wtime() - stime;
  recordCommTime(etime);
#endif
}

void buildBasis(void(*ef)(double*), struct Base basis[], int64_t ncells, struct Cell* cells, const struct CSC* rel_near, int64_t levels, 
const struct CellComm* comm, const struct Body* bodies, int64_t nbodies, double epi, int64_t mrank, int64_t sp_pts) {

  for (int64_t l = levels; l >= 0; l--) {
    int64_t xlen = 0;
    content_length(&xlen, &comm[l]);
    basis[l].Ulen = xlen;
    int64_t* arr_i = (int64_t*)malloc(sizeof(int64_t) * (xlen * 4 + 1));
    basis[l].Lchild = arr_i;
    basis[l].Dims = &arr_i[xlen];
    basis[l].DimsLr = &arr_i[xlen * 2];
    basis[l].Offsets = &arr_i[xlen * 3];
    basis[l].Multipoles = NULL;
    int64_t jbegin = 0, jend = ncells;
    get_level(&jbegin, &jend, cells, l, -1);
    for (int64_t j = 0; j < xlen; j++) {
      int64_t gj = j;
      i_global(&gj, &comm[l]);
      int64_t lc = cells[jbegin + gj].Child;
      arr_i[j] = -1;
      if (lc >= 0) {
        arr_i[j] = lc - jend;
        i_local(&arr_i[j], &comm[l + 1]);
      }
    }

    struct Matrix* arr_m = (struct Matrix*)calloc(xlen * 3, sizeof(struct Matrix));
    basis[l].Uo = arr_m;
    basis[l].Uc = &arr_m[xlen];
    basis[l].R = &arr_m[xlen * 2];

    int64_t ibegin = 0, iend = xlen;
    self_local_range(&ibegin, &iend, &comm[l]);
    int64_t nodes = iend - ibegin;

    struct SampleBodies samples;
    buildSampleBodies(&samples, sp_pts, sp_pts, nbodies, ncells, cells, &rel_near[l], &basis[l].Lchild[ibegin], l == levels ? NULL : &basis[l + 1], l);

    int64_t count = 0;
    int64_t count_m = 0;
    for (int64_t i = 0; i < nodes; i++) {
      int64_t ske_len = samples.SkeLens[i];
      int64_t len_m = samples.FarLens[i];
      len_m = len_m < samples.CloseLens[i] ? samples.CloseLens[i] : len_m;
      len_m = len_m < ske_len ? ske_len : len_m;
      basis[l].Dims[i + ibegin] = ske_len;
      count = count + ske_len;
      count_m = count_m + ske_len * (ske_len * 2 + len_m + 2);
    }

    int32_t* ipiv_data = (int32_t*)malloc(sizeof(int32_t) * count);
    int32_t** ipiv_ptrs = (int32_t**)malloc(sizeof(int32_t*) * nodes);
    double* matrix_data = (double*)malloc(sizeof(double) * count_m);
    double** matrix_ptrs = (double**)malloc(sizeof(double*) * (xlen + 1));

    count = 0;
    count_m = 0;
    for (int64_t i = 0; i < nodes; i++) {
      int64_t ske_len = samples.SkeLens[i];
      int64_t len_m = samples.FarLens[i];
      len_m = len_m < samples.CloseLens[i] ? samples.CloseLens[i] : len_m;
      len_m = len_m < ske_len ? ske_len : len_m;
      ipiv_ptrs[i] = &ipiv_data[count];
      matrix_ptrs[i + ibegin] = &matrix_data[count_m];
      count = count + ske_len;
      count_m = count_m + ske_len * (ske_len * 2 + len_m + 2);
    }

#pragma omp parallel for
    for (int64_t i = 0; i < nodes; i++) {
      int64_t ske_len = samples.SkeLens[i];
      int64_t len_s = samples.FarLens[i] + (samples.CloseLens[i] > 0 ? ske_len : 0);
      double* mat = matrix_ptrs[i + ibegin];
      struct Matrix S = (struct Matrix){ &mat[ske_len * ske_len], ske_len, len_s };

      struct Matrix S_dn = (struct Matrix){ &mat[ske_len * ske_len], ske_len, ske_len };
      double nrm_dn = 0.;
      double nrm_lr = 0.;
      struct Matrix S_dn_work = (struct Matrix){ &mat[ske_len * ske_len * 2], ske_len, samples.CloseLens[i] };
      gen_matrix(ef, ske_len, samples.CloseLens[i], bodies, bodies, S_dn_work.A, samples.Skeletons[i], samples.CloseBodies[i]);
      mmult('N', 'T', &S_dn_work, &S_dn_work, &S_dn, 1., 0.);
      nrm2_A(&S_dn, &nrm_dn);

      struct Matrix S_lr = (struct Matrix){ &mat[ske_len * ske_len * 2], ske_len, samples.FarLens[i] };
      gen_matrix(ef, ske_len, samples.FarLens[i], bodies, bodies, S_lr.A, samples.Skeletons[i], samples.FarBodies[i]);
      nrm2_A(&S_lr, &nrm_lr);
      double scale = (nrm_dn == 0. || nrm_lr == 0.) ? 1. : nrm_lr / nrm_dn;
      scal_A(&S_dn, scale);

      int64_t rank = ske_len < len_s ? ske_len : len_s;
      rank = mrank > 0 ? (mrank < rank ? mrank : rank) : rank;
      struct Matrix Q = (struct Matrix){ mat, ske_len, ske_len };
      double* Svec = &mat[ske_len * (ske_len + len_s)];
      svd_U(&S, &Q, Svec);

      if (epi > 0.) {
        int64_t r = 0;
        double sepi = Svec[0] * epi;
        while(r < rank && Svec[r] > sepi)
          r += 1;
        rank = r;
      }
      basis[l].DimsLr[i + ibegin] = rank;

      struct Matrix Rc = (struct Matrix){ &mat[ske_len * ske_len * 2], ske_len, ske_len };
      memset(Rc.A, 0, sizeof(double) * ske_len * ske_len);
      for (int64_t j = 0; j < ske_len; j++)
        Rc.A[j + j * ske_len] = 1.;
    }

    for (int64_t i = 0; i < nodes; i++) {
      int64_t ske_len = samples.SkeLens[i];
      int64_t rank = basis[l].DimsLr[i + ibegin];
      double* mat = matrix_ptrs[i + ibegin];
      int32_t* pa = ipiv_ptrs[i];
      struct Matrix Qo = (struct Matrix){ mat, ske_len, rank };
      id_row_batch(&Qo, pa, &mat[ske_len * rank]);

      struct Matrix Rc = (struct Matrix){ &mat[ske_len * ske_len * 2], ske_len, ske_len };
      int64_t lc = basis[l].Lchild[i + ibegin];
      for (int64_t j = 0; lc >= 0 && j < 2; j++) {
        int64_t diml = basis[l + 1].DimsLr[lc + j];
        int64_t off = basis[l + 1].Offsets[lc + j] - basis[l + 1].Offsets[lc];
        mat_cpy_batch(diml, diml, &(basis[l + 1].R)[lc + j], &Rc, 0, 0, off, off);
      }
    }
    id_row_flush();
    mat_cpy_flush();

#pragma omp parallel for
    for (int64_t i = 0; i < nodes; i++) {
      int64_t ske_len = samples.SkeLens[i];
      int64_t rank = basis[l].DimsLr[i + ibegin];
      double* mat = matrix_ptrs[i + ibegin];
      int32_t* pa = ipiv_ptrs[i];

      for (int64_t j = 0; j < rank; j++) {
        int64_t piv = (int64_t)pa[j] - 1;
        if (piv != j) { 
          int64_t c = samples.Skeletons[i][piv];
          samples.Skeletons[i][piv] = samples.Skeletons[i][j];
          samples.Skeletons[i][j] = c;
        }
      }

      struct Matrix Q = (struct Matrix){ mat, ske_len, ske_len };
      struct Matrix Qo = (struct Matrix){ mat, ske_len, rank };
      struct Matrix R = (struct Matrix){ &mat[ske_len * ske_len], rank, rank };
      struct Matrix Rc = (struct Matrix){ &mat[ske_len * ske_len * 2], ske_len, ske_len };      
      upper_tri_reflec_mult('L', &Rc, &Qo);
      qr_full(&Q, &R, Rc.A);
    }

    dist_int_64_xlen(basis[l].Dims, &comm[l]);
    dist_int_64_xlen(basis[l].DimsLr, &comm[l]);

    count = 0;
    count_m = 0;
    for (int64_t i = 0; i < xlen; i++) {
      int64_t m = basis[l].Dims[i];
      int64_t n = basis[l].DimsLr[i];
      basis[l].Offsets[i] = count;
      count = count + n;
      count_m = count_m + m * m + n * n;
    }
    basis[l].Offsets[xlen] = count;

    if (count > 0)
      basis[l].Multipoles = (int64_t*)malloc(sizeof(int64_t) * count);
    for (int64_t i = 0; i < nodes; i++) {
      int64_t offset = basis[l].Offsets[i + ibegin];
      int64_t n = basis[l].DimsLr[i + ibegin];
      if (n > 0)
        memcpy(&basis[l].Multipoles[offset], samples.Skeletons[i], sizeof(int64_t) * n);
    }
    dist_int_64(basis[l].Multipoles, basis[l].Offsets, &comm[l]);

    double* data_basis = NULL;
    if (count_m > 0)
      data_basis = (double*)malloc(sizeof(int64_t) * count_m);
    for (int64_t i = 0; i < xlen; i++) {
      int64_t m = basis[l].Dims[i];
      int64_t n = basis[l].DimsLr[i];
      int64_t size = m * m + n * n;
      if (ibegin <= i && i < iend && size > 0)
        memcpy(data_basis, matrix_ptrs[i], sizeof(double) * size);
      basis[l].Uo[i] = (struct Matrix){ data_basis, m, n };
      basis[l].Uc[i] = (struct Matrix){ &data_basis[m * n], m, m - n };
      basis[l].R[i] = (struct Matrix){ &data_basis[m * m], n, n };
      matrix_ptrs[i] = data_basis;
      data_basis = &data_basis[size];
    }
    matrix_ptrs[xlen] = data_basis;
    dist_double(matrix_ptrs, &comm[l]);

    free(ipiv_data);
    free(ipiv_ptrs);
    free(matrix_data);
    free(matrix_ptrs);
    sampleBodies_free(&samples);
  }
}

void basis_free(struct Base* basis) {
  double* data = basis->Uo[0].A;
  if (data)
    free(data);
  if (basis->Multipoles)
    free(basis->Multipoles);
  free(basis->Lchild);
  free(basis->Uo);
}

void evalS(void(*ef)(double*), struct Matrix* S, const struct Base* basis, const struct Body* bodies, const struct CSC* rels, const struct CellComm* comm) {
  int64_t ibegin = 0, iend = 0;
  self_local_range(&ibegin, &iend, comm);
  int64_t lbegin = ibegin;
  i_global(&lbegin, comm);

#pragma omp parallel for
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
      upper_tri_reflec_mult('L', &basis->R[box_y], &S[yx]);
      upper_tri_reflec_mult('R', &basis->R[x + ibegin], &S[yx]);
    }
  }
}

