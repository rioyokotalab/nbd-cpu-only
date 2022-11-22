
#include "nbd.h"
#include "profile.h"

#include "stdlib.h"
#include "string.h"
#include "math.h"

void allocNodes(struct Node A[], const struct Base basis[], const struct CSC rels_near[], const struct CSC rels_far[], const struct CellComm comm[], int64_t levels) {
  const int alignment = 16;
  int64_t* dim_max = (int64_t*)malloc(sizeof(int64_t) * (levels + 1) * 2);
  for (int64_t i = 0; i <= levels; i++) {
    int64_t ibegin = 0, iend = 0;
    self_local_range(&ibegin, &iend, &comm[i]);
    int64_t dimc_max = 0, dimr_max = 0;
    for (int64_t x = ibegin; x < iend; x++) {
      int64_t dimr = basis[i].DimsLr[x];
      int64_t dimc = basis[i].Dims[x] - dimr;
      dimc_max = dimc_max < dimc ? dimc : dimc_max;
      dimr_max = dimr_max < dimr ? dimr : dimr_max;
    }
    int64_t dimc_rem = dimc_max & (alignment - 1);
    int64_t dimr_rem = dimr_max & (alignment - 1);
    dim_max[i * 2] = dimc_max - dimc_rem + (dimc_rem ? alignment : 0);
    dim_max[i * 2 + 1] = dimr_max - dimr_rem + (dimr_rem ? alignment : 0);
  }
  MPI_Allreduce(MPI_IN_PLACE, dim_max, (levels + 1) * 2, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);

  for (int64_t i = 0; i <= levels; i++) {
    int64_t n_i = rels_near[i].N;
    int64_t nnz = rels_near[i].ColIndex[n_i];
    int64_t nnz_f = rels_far[i].ColIndex[n_i];
    int64_t llen = basis[i].Ulen;
    int64_t len_arr = nnz * 4 + nnz_f;

    struct Matrix* arr_m = (struct Matrix*)malloc(sizeof(struct Matrix) * len_arr);
    A[i].A = arr_m;
    A[i].A_cc = &arr_m[nnz];
    A[i].A_oc = &arr_m[nnz * 2];
    A[i].A_oo = &arr_m[nnz * 3];
    A[i].S = &arr_m[nnz * 4];

    A[i].lenA = nnz;
    A[i].lenS = nnz_f;
    A[i].dimR = dim_max[i * 2];
    A[i].dimS = dim_max[i * 2 + 1];

    int64_t dimn = A[i].dimR + A[i].dimS;
    int64_t stride = dimn * dimn;
    alloc_matrices_aligned(&A[i].A_ptr, &A[i].A_buf, dimn, dimn, nnz);
    alloc_matrices_aligned(&A[i].U_ptr, &A[i].U_buf, dimn, dimn, llen);

    int64_t nloc = 0, nend = 0;
    self_local_range(&nloc, &nend, &comm[i]);

    for (int64_t x = 0; x < n_i; x++) {
      int64_t box_x = nloc + x;
      int64_t dim_x = basis[i].Dims[box_x];
      int64_t diml_x = basis[i].DimsLr[box_x];
      int64_t dimc_x = dim_x - diml_x;

      for (int64_t yx = rels_near[i].ColIndex[x]; yx < rels_near[i].ColIndex[x + 1]; yx++) {
        int64_t y = rels_near[i].RowIndex[yx];
        int64_t dim_y = basis[i].Dims[y];
        int64_t diml_y = basis[i].DimsLr[y];
        int64_t dimc_y = dim_y - diml_y;

        arr_m[yx] = (struct Matrix) { A[i].A_buf + yx * stride, dim_y, dim_x, dimn }; // A
        arr_m[yx + nnz] = (struct Matrix) { A[i].A_buf + yx * stride, dimc_y, dimc_x, dimn }; // A_cc
        arr_m[yx + nnz * 2] = (struct Matrix) { A[i].A_buf + yx * stride + A[i].dimR, diml_y, dimc_x, dimn }; // A_oc
        arr_m[yx + nnz * 3] = (struct Matrix) { NULL, diml_y, diml_x, 0 }; // A_oo
      }

      for (int64_t yx = rels_far[i].ColIndex[x]; yx < rels_far[i].ColIndex[x + 1]; yx++) {
        int64_t y = rels_far[i].RowIndex[yx];
        int64_t diml_y = basis[i].DimsLr[y];
        arr_m[yx + nnz * 4] = (struct Matrix) { NULL, diml_y, diml_x, 0 }; // S
      }
    }

    for (int64_t x = 0; x < llen; x++) {
      struct Matrix Uc = (struct Matrix) { A[i].U_buf + x * stride, dimn, A[i].dimR, dimn };
      struct Matrix Uo = (struct Matrix) { A[i].U_buf + x * stride + A[i].dimR * dimn, dimn, A[i].dimS, dimn };
      mat_cpy(basis[i].Uc[x].M, basis[i].Uc[x].N, &basis[i].Uc[x], &Uc, 0, 0, 0, 0);
      mat_cpy(basis[i].Uo[x].M, basis[i].Uo[x].N, &basis[i].Uo[x], &Uo, 0, 0, 0, 0);
    }
    flush_buffer('S', A[i].U_ptr, A[i].U_buf, stride * llen);

    if (i > 0) {
      const struct CSC* rels_up = &rels_near[i - 1];
      const struct Matrix* Mup = A[i - 1].A;
      const int64_t* dims_lr = basis[i].DimsLr;
      const int64_t* lchild = basis[i - 1].Lchild;
      int64_t ploc = 0, pend = 0;
      self_local_range(&ploc, &pend, &comm[i - 1]);
      int64_t offset_x[2] = {0};
      int64_t offset_y[2] = {0};
      const int64_t NCHILD = 2;

      for (int64_t j = 0; j < rels_up->N; j++) {
        int64_t lj = j + ploc;
        int64_t cj = lchild[lj];
        int64_t x0 = cj - nloc;
        for (int64_t x = 0; x < NCHILD - 1; x++)
          offset_x[x + 1] = offset_x[x] + dims_lr[x + cj];

        for (int64_t ij = rels_up->ColIndex[j]; ij < rels_up->ColIndex[j + 1]; ij++) {
          int64_t li = rels_up->RowIndex[ij];
          int64_t y0 = lchild[li];
          for (int64_t y = 0; y < NCHILD - 1; y++)
            offset_y[y + 1] = offset_y[y] + dims_lr[y + y0];

          for (int64_t x = 0; x < NCHILD; x++)
            if ((x + x0) >= 0 && (x + x0) < rels_near[i].N)
              for (int64_t yx = rels_near[i].ColIndex[x + x0]; yx < rels_near[i].ColIndex[x + x0 + 1]; yx++)
                for (int64_t y = 0; y < NCHILD; y++)
                  if (rels_near[i].RowIndex[yx] == (y + y0)) {
                    arr_m[yx + nnz * 3].A = Mup[ij].A + Mup[ij].LDA * offset_x[x] + offset_y[y];
                    arr_m[yx + nnz * 3].LDA = Mup[ij].LDA;
                  }
          
          for (int64_t x = 0; x < NCHILD; x++)
            if ((x + x0) >= 0 && (x + x0) < rels_far[i].N)
              for (int64_t yx = rels_far[i].ColIndex[x + x0]; yx < rels_far[i].ColIndex[x + x0 + 1]; yx++)
                for (int64_t y = 0; y < NCHILD; y++)
                  if (rels_far[i].RowIndex[yx] == (y + y0)) {
                    arr_m[yx + nnz * 4].A = Mup[ij].A + Mup[ij].LDA * offset_x[x] + offset_y[y];
                    arr_m[yx + nnz * 4].LDA = Mup[ij].LDA;
                  }
        }
      }
    }
  }

  free(dim_max);
}

void node_free(struct Node* node) {
  free_matrices(node->A_ptr, node->A_buf);
  free_matrices(node->U_ptr, node->U_buf);
  free(node->A);
}

void merge_double(double* arr, int64_t alen, const struct CellComm* comm) {
#ifdef _PROF
  double stime = MPI_Wtime();
#endif
  if (comm->Comm_merge != MPI_COMM_NULL)
    MPI_Allreduce(MPI_IN_PLACE, arr, alen, MPI_DOUBLE, MPI_SUM, comm->Comm_merge);
  if (comm->Proc[1] - comm->Proc[0] > 1)
    MPI_Bcast(arr, alen, MPI_DOUBLE, 0, comm->Comm_share);
#ifdef _PROF
  double etime = MPI_Wtime() - stime;
  recordCommTime(etime);
#endif
}

void factorA_mov_mem(char dir, struct Node A[], int64_t levels) {
  for (int64_t i = 0; i <= levels; i++) {
    int64_t stride = (A[i].dimR + A[i].dimS) * (A[i].dimR + A[i].dimS);
    flush_buffer(dir, A[i].A_ptr, A[i].A_buf, stride * A[i].lenA);
  }
}

void factorA(struct Node A[], const struct Base basis[], const struct CSC rels[], const struct CellComm comm[], int64_t levels) {

  for (int64_t i = levels; i > 0; i--) {
    int64_t ibegin = 0, iend = 0;
    self_local_range(&ibegin, &iend, &comm[i]);
    int64_t llen = basis[i].Ulen;
    int64_t nnz = A[i].lenA;
    int64_t dimc = A[i].dimR;
    int64_t dimr = A[i].dimS;

    double** A_next = (double**)malloc(sizeof(double*) * nnz);
    int64_t n_next = A[i - 1].dimR + A[i - 1].dimS;
    int64_t nnz_next = A[i - 1].lenA;

    for (int64_t x = 0; x < nnz; x++)
      A_next[x] = A[i - 1].A_ptr + (A[i].A_oo[x].A - A[i - 1].A_buf);

    batch_cholesky_factor(dimc, dimr, A[i].U_ptr, A[i].A_ptr, n_next, A_next, llen, rels[i].N, ibegin, rels[i].RowIndex, rels[i].ColIndex, basis[i].DimsLr);

    merge_double(A[i - 1].A_ptr, n_next * n_next * nnz_next, &comm[i - 1]);
    free(A_next);
#ifdef _PROF
    record_factor_flops(dimc, dimr, nnz, iend - ibegin);
#endif
  }
  chol_decomp(A[0].A_ptr, A[0].dimR);
}

void allocRightHandSides(char mvsv, struct RightHandSides rhs[], const struct Base base[], int64_t levels) {
  for (int64_t i = 0; i <= levels; i++) {
    int64_t len = base[i].Ulen;
    int64_t len_arr = len * 4;
    struct Matrix* arr_m = (struct Matrix*)malloc(sizeof(struct Matrix) * len_arr);
    rhs[i].Xlen = len;
    rhs[i].X = arr_m;
    rhs[i].XcM = &arr_m[len];
    rhs[i].XoL = &arr_m[len * 2];
    rhs[i].B = &arr_m[len * 3];

    int64_t count = 0;
    for (int64_t j = 0; j < len; j++) {
      int64_t dim = base[i].Dims[j];
      int64_t diml = base[i].DimsLr[j];
      int64_t dimc = diml;
      int64_t dimb = dim;
      if (mvsv == 'S' || mvsv == 's')
      { dimc = dim - diml; dimb = 0; }
      arr_m[j] = (struct Matrix) { NULL, dim, 1, dim }; // X
      arr_m[j + len] = (struct Matrix) { NULL, dimc, 1, dimc }; // Xc
      arr_m[j + len * 2] = (struct Matrix) { NULL, diml, 1, diml }; // Xo
      arr_m[j + len * 3] = (struct Matrix) { NULL, dimb, 1, dimb }; // B
      count = count + dim + dimc + diml + dimb;
    }

    double* data = NULL;
    if (count > 0)
      data = (double*)calloc(count, sizeof(double));
    
    for (int64_t j = 0; j < len_arr; j++) {
      arr_m[j].A = data;
      int64_t len = arr_m[j].M;
      data = data + len;
    }
  }
}

void rightHandSides_free(struct RightHandSides* rhs) {
  double* data = rhs->X[0].A;
  if (data)
    free(data);
  free(rhs->X);
}

void svAccFw(struct Matrix* Xc, struct Matrix* Xo, const struct Matrix* X, const struct Matrix* Uc, const struct Matrix* Uo, const struct Matrix* A_cc, const struct Matrix* A_oc, const struct CSC* rels, const struct CellComm* comm) {
  int64_t ibegin = 0, iend = 0;
  self_local_range(&ibegin, &iend, comm);

  for (int64_t x = 0; x < rels->N; x++) {
    mmult('T', 'N', &Uc[x + ibegin], &X[x + ibegin], &Xc[x + ibegin], 1., 1.);
    mmult('T', 'N', &Uo[x + ibegin], &X[x + ibegin], &Xo[x + ibegin], 1., 1.);
    int64_t xx;
    lookupIJ(&xx, rels, x + ibegin, x);
    mat_solve('F', &Xc[x + ibegin], &A_cc[xx]);

    for (int64_t yx = rels->ColIndex[x]; yx < rels->ColIndex[x + 1]; yx++) {
      int64_t y = rels->RowIndex[yx];
      if (y > x + ibegin)
        mmult('N', 'N', &A_cc[yx], &Xc[x + ibegin], &Xc[y], -1., 1.);
      mmult('N', 'N', &A_oc[yx], &Xc[x + ibegin], &Xo[y], -1., 1.);
    }
  }
}

void svAccBk(struct Matrix* Xc, const struct Matrix* Xo, struct Matrix* X, const struct Matrix* Uc, const struct Matrix* Uo, const struct Matrix* A_cc, const struct Matrix* A_oc, const struct CSC* rels, const struct CellComm* comm) {
  int64_t ibegin = 0, iend = 0;
  self_local_range(&ibegin, &iend, comm);

  for (int64_t x = rels->N - 1; x >= 0; x--) {
    for (int64_t yx = rels->ColIndex[x]; yx < rels->ColIndex[x + 1]; yx++) {
      int64_t y = rels->RowIndex[yx];
      mmult('T', 'N', &A_oc[yx], &Xo[y], &Xc[x + ibegin], -1., 1.);
      if (y > x + ibegin)
        mmult('T', 'N', &A_cc[yx], &Xc[y], &Xc[x + ibegin], -1., 1.);
    }

    int64_t xx;
    lookupIJ(&xx, rels, x + ibegin, x);
    mat_solve('B', &Xc[x + ibegin], &A_cc[xx]);
    mmult('N', 'N', &Uc[x + ibegin], &Xc[x + ibegin], &X[x + ibegin], 1., 0.);
    mmult('N', 'N', &Uo[x + ibegin], &Xo[x + ibegin], &X[x + ibegin], 1., 1.);
  }
}

void permuteAndMerge(char fwbk, struct Matrix* px, struct Matrix* nx, const int64_t* lchild, const struct CellComm* comm) {
  int64_t nloc = 0, nend = 0;
  self_local_range(&nloc, &nend, comm);
  int64_t nboxes = nend - nloc;

  if (fwbk == 'F' || fwbk == 'f')
    for (int64_t i = 0; i < nboxes; i++) {
      int64_t c = i + nloc;
      int64_t c0 = lchild[c];
      int64_t c1 = c0 + 1;
      mat_cpy(px[c0].M, 1, &px[c0], &nx[c], 0, 0, 0, 0);
      mat_cpy(px[c1].M, 1, &px[c1], &nx[c], 0, 0, nx[c].M - px[c1].M, 0);
    }
  else if (fwbk == 'B' || fwbk == 'b')
    for (int64_t i = 0; i < nboxes; i++) {
      int64_t c = i + nloc;
      int64_t c0 = lchild[c];
      int64_t c1 = c0 + 1;
      mat_cpy(px[c0].M, 1, &nx[c], &px[c0], 0, 0, 0, 0);
      mat_cpy(px[c1].M, 1, &nx[c], &px[c1], nx[c].M - px[c1].M, 0, 0, 0);
    }
}

void dist_double_svfw(char fwbk, double* arr[], const struct CellComm* comm) {
  int64_t plen = comm->Proc[0] == comm->worldRank ? comm->lenTargets : 0;
  const int64_t* row = comm->ProcTargets;
  double* data = arr[0];
  int is_all = fwbk == 'A' || fwbk == 'a';
  int64_t lbegin = 0;
#ifdef _PROF
  double stime = MPI_Wtime();
#endif
  for (int64_t i = 0; i < plen; i++) {
    int64_t p = row[i];
    int is_fw = (fwbk == 'F' || fwbk == 'f') && p <= comm->worldRank;
    int is_bk = (fwbk == 'B' || fwbk == 'b') && comm->worldRank < p;
    int64_t llen = comm->ProcBoxesEnd[p] - comm->ProcBoxes[p];
    if (is_all || is_fw || is_bk) {
      int64_t offset = arr[lbegin] - data;
      int64_t len = arr[lbegin + llen] - arr[lbegin];
      MPI_Allreduce(MPI_IN_PLACE, &data[offset], len, MPI_DOUBLE, MPI_SUM, comm->Comm_box[p]);
    }
    lbegin = lbegin + llen;
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

void dist_double_svbk(char fwbk, double* arr[], const struct CellComm* comm) {
  int64_t plen = comm->Proc[0] == comm->worldRank ? comm->lenTargets : 0;
  const int64_t* row = comm->ProcTargets;
  double* data = arr[0];
  int is_all = fwbk == 'A' || fwbk == 'a';
  int64_t lend;
  content_length(&lend, comm);
#ifdef _PROF
  double stime = MPI_Wtime();
#endif
  for (int64_t i = plen - 1; i >= 0; i--) {
    int64_t p = row[i];
    int is_fw = (fwbk == 'F' || fwbk == 'f') && p <= comm->worldRank;
    int is_bk = (fwbk == 'B' || fwbk == 'b') && comm->worldRank < p;
    int64_t llen = comm->ProcBoxesEnd[p] - comm->ProcBoxes[p];
    int64_t lbegin = lend - llen;
    if (is_all || is_fw || is_bk) {
      int64_t offset = arr[lbegin] - data;
      int64_t len = arr[lbegin + llen] - arr[lbegin];
      MPI_Bcast(&data[offset], len, MPI_DOUBLE, comm->ProcRootI[p], comm->Comm_box[p]);
    }
    lend = lbegin;
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

void solveA(struct RightHandSides rhs[], const struct Node A[], const struct Base basis[], const struct CSC rels[], double* X, const struct CellComm comm[], int64_t levels) {
  int64_t ibegin = 0, iend = 0;
  self_local_range(&ibegin, &iend, &comm[levels]);
  int64_t lenX = (rhs[levels].X[iend - 1].A - rhs[levels].X[ibegin].A) + rhs[levels].X[iend - 1].M;
  memcpy(rhs[levels].X[ibegin].A, X, lenX * sizeof(double));

  for (int64_t i = levels; i > 0; i--) {
    int64_t xlen = rhs[i].Xlen;
    double** arr_comm = (double**)malloc(sizeof(double*) * (xlen + 1));
    for (int64_t j = 0; j < xlen; j++)
      arr_comm[j] = rhs[i].XcM[j].A;
    arr_comm[xlen] = arr_comm[xlen - 1] + rhs[i].XcM[xlen - 1].M;
    dist_double_svfw('F', arr_comm, &comm[i]);

    svAccFw(rhs[i].XcM, rhs[i].XoL, rhs[i].X, basis[i].Uc, basis[i].Uo, A[i].A_cc, A[i].A_oc, &rels[i], &comm[i]);
    dist_double_svfw('B', arr_comm, &comm[i]);

    for (int64_t j = 0; j < xlen; j++)
      arr_comm[j] = rhs[i].XoL[j].A;
    arr_comm[xlen] = arr_comm[xlen - 1] + rhs[i].XoL[xlen - 1].M;
    dist_double_svfw('A', arr_comm, &comm[i]);

    free(arr_comm);
    permuteAndMerge('F', rhs[i].XoL, rhs[i - 1].X, basis[i - 1].Lchild, &comm[i - 1]);
  }
  mat_solve('A', &rhs[0].X[0], &A[0].A[0]);
  
  for (int64_t i = 1; i <= levels; i++) {
    permuteAndMerge('B', rhs[i].XoL, rhs[i - 1].X, basis[i - 1].Lchild, &comm[i - 1]);
    int64_t xlen = rhs[i].Xlen;
    double** arr_comm = (double**)malloc(sizeof(double*) * (xlen + 1));
    for (int64_t j = 0; j < xlen; j++)
      arr_comm[j] = rhs[i].XoL[j].A;
    arr_comm[xlen] = arr_comm[xlen - 1] + rhs[i].XoL[xlen - 1].M;
    dist_double_svbk('A', arr_comm, &comm[i]);

    for (int64_t j = 0; j < xlen; j++)
      arr_comm[j] = rhs[i].XcM[j].A;
    arr_comm[xlen] = arr_comm[xlen - 1] + rhs[i].XcM[xlen - 1].M;
    dist_double_svbk('B', arr_comm, &comm[i]);
    
    svAccBk(rhs[i].XcM, rhs[i].XoL, rhs[i].X, basis[i].Uc, basis[i].Uo, A[i].A_cc, A[i].A_oc, &rels[i], &comm[i]);
    dist_double_svbk('F', arr_comm, &comm[i]);
    free(arr_comm);
  }
  memcpy(X, rhs[levels].X[ibegin].A, lenX * sizeof(double));
}

void horizontalPass(struct Matrix* B, const struct Matrix* X, const struct Matrix* A, const struct CSC* rels, const struct CellComm* comm) {
  int64_t ibegin = 0, iend = 0;
  self_local_range(&ibegin, &iend, comm);
#pragma omp parallel for
  for (int64_t y = 0; y < rels->N; y++)
    for (int64_t xy = rels->ColIndex[y]; xy < rels->ColIndex[y + 1]; xy++) {
      int64_t x = rels->RowIndex[xy];
      mmult('T', 'N', &A[xy], &X[x], &B[y + ibegin], 1., 1.);
    }
}

void matVecA(struct RightHandSides rhs[], const struct Node A[], const struct Base basis[], const struct CSC rels_near[], const struct CSC rels_far[], double* X, const struct CellComm comm[], int64_t levels) {
  int64_t ibegin = 0, iend = 0;
  self_local_range(&ibegin, &iend, &comm[levels]);
  int64_t lenX = (rhs[levels].X[iend - 1].A - rhs[levels].X[ibegin].A) + rhs[levels].X[iend - 1].M;
  memcpy(rhs[levels].X[ibegin].A, X, lenX * sizeof(double));

  int64_t xlen = rhs[levels].Xlen;
  double** arr_comm = (double**)malloc(sizeof(double*) * (xlen + 1));
  for (int64_t j = 0; j < xlen; j++)
    arr_comm[j] = rhs[levels].X[j].A;
  arr_comm[xlen] = arr_comm[xlen - 1] + rhs[levels].X[xlen - 1].M;
  dist_double_svbk('A', arr_comm, &comm[levels]);
  free(arr_comm);

  for (int64_t i = levels; i > 0; i--) {
    self_local_range(&ibegin, &iend, &comm[i]);
    int64_t iboxes = iend - ibegin;
    for (int64_t j = 0; j < iboxes; j++)
      mmult('T', 'N', &basis[i].Uo[j + ibegin], &rhs[i].X[j + ibegin], &rhs[i].XcM[j + ibegin], 1., 0.);
    xlen = rhs[i].Xlen;
    arr_comm = (double**)malloc(sizeof(double*) * (xlen + 1));
    for (int64_t j = 0; j < xlen; j++)
      arr_comm[j] = rhs[i].XcM[j].A;
    arr_comm[xlen] = arr_comm[xlen - 1] + rhs[i].XcM[xlen - 1].M;
    dist_double_svbk('A', arr_comm, &comm[i]);
    free(arr_comm);
    permuteAndMerge('F', rhs[i].XcM, rhs[i - 1].X, basis[i - 1].Lchild, &comm[i - 1]);
  }
  
  for (int64_t i = 1; i <= levels; i++) {
    permuteAndMerge('B', rhs[i].XoL, rhs[i - 1].B, basis[i - 1].Lchild, &comm[i - 1]);
    horizontalPass(rhs[i].XoL, rhs[i].XcM, A[i].S, &rels_far[i], &comm[i]);
    self_local_range(&ibegin, &iend, &comm[i]);
    int64_t iboxes = iend - ibegin;
    for (int64_t j = 0; j < iboxes; j++)
      mmult('N', 'N', &basis[i].Uo[j + ibegin], &rhs[i].XoL[j + ibegin], &rhs[i].B[j + ibegin], 1., 0.);
  }
  horizontalPass(rhs[levels].B, rhs[levels].X, A[levels].A, &rels_near[levels], &comm[levels]);
  memcpy(X, rhs[levels].B[ibegin].A, lenX * sizeof(double));
}


void solveRelErr(double* err_out, const double* X, const double* ref, int64_t lenX) {
  double err[2] = { 0., 0. };
  for (int64_t i = 0; i < lenX; i++) {
    double diff = X[i] - ref[i];
    err[0] = err[0] + diff * diff;
    err[1] = err[1] + ref[i] * ref[i];
  }
  MPI_Allreduce(MPI_IN_PLACE, err, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  *err_out = sqrt(err[0] / err[1]);
}

