
#include "nbd.h"
#include "profile.h"

#include "stdlib.h"
#include "string.h"
#include "math.h"

void allocNodes(struct Node A[], const struct Base basis[], const struct CSC rels_near[], const struct CSC rels_far[], const struct CellComm comm[], int64_t levels) {
  for (int64_t i = 0; i <= levels; i++) {
    int64_t n_i = rels_near[i].N;
    int64_t nnz = rels_near[i].ColIndex[n_i];
    int64_t nnz_f = rels_far[i].ColIndex[n_i];
    int64_t len_arr = nnz * 4 + nnz_f;
    struct Matrix* arr_m = (struct Matrix*)malloc(sizeof(struct Matrix) * len_arr);
    A[i].A = arr_m;
    A[i].A_cc = &arr_m[nnz];
    A[i].A_oc = &arr_m[nnz * 2];
    A[i].A_oo = &arr_m[nnz * 3];
    A[i].S = &arr_m[nnz * 4];
    A[i].lenA = nnz;
    A[i].lenS = nnz_f;

    int64_t ibegin = 0, iend = 0;
    self_local_range(&ibegin, &iend, &comm[i]);

    int64_t count = 0;
    for (int64_t x = 0; x < rels_near[i].N; x++) {
      int64_t box_x = ibegin + x;
      int64_t dim_x = basis[i].Dims[box_x];
      int64_t diml_x = basis[i].DimsLr[box_x];
      int64_t dimc_x = dim_x - diml_x;

      for (int64_t yx = rels_near[i].ColIndex[x]; yx < rels_near[i].ColIndex[x + 1]; yx++) {
        int64_t y = rels_near[i].RowIndex[yx];
        int64_t box_y = y;
        i_local(&box_y, &comm[i]);
        int64_t dim_y = basis[i].Dims[box_y];
        int64_t diml_y = basis[i].DimsLr[box_y];
        int64_t dimc_y = dim_y - diml_y;
        arr_m[yx].M = dim_y; // A
        arr_m[yx].N = dim_x;
        arr_m[yx + nnz].M = dimc_y; // A_cc
        arr_m[yx + nnz].N = dimc_x;
        arr_m[yx + nnz * 2].M = diml_y; // A_oc
        arr_m[yx + nnz * 2].N = dimc_x;
        arr_m[yx + nnz * 3].M = diml_y; // A_oo
        arr_m[yx + nnz * 3].N = diml_x;
        count += dim_y * dim_x;
      }

      for (int64_t yx = rels_far[i].ColIndex[x]; yx < rels_far[i].ColIndex[x + 1]; yx++) {
        int64_t y = rels_far[i].RowIndex[yx];
        int64_t box_y = y;
        i_local(&box_y, &comm[i]);
        int64_t diml_y = basis[i].DimsLr[box_y];
        arr_m[yx + nnz * 4].M = diml_y; // S
        arr_m[yx + nnz * 4].N = diml_x;
        count += diml_y * diml_x;
      }
    }

    double* data = NULL;
    if (count > 0)
      data = (double*)calloc(count, sizeof(double));
    
    for (int64_t x = 0; x < nnz; x++) {
      int64_t dim_y = arr_m[x].M;
      int64_t dim_x = arr_m[x].N;
      int64_t dimc_y = arr_m[x + nnz].M;
      int64_t dimc_x = arr_m[x + nnz].N;
      arr_m[x].A = data; // A
      arr_m[x + nnz].A = data; // A_cc
      arr_m[x + nnz * 2].A = data + dimc_y * dimc_x; // A_oc
      arr_m[x + nnz * 3].A = data + dim_y * dimc_x; // A_oo
      data = data + dim_y * dim_x;
    }

    for (int64_t x = 0; x < nnz_f; x++) {
      int64_t y = x + nnz * 4;
      arr_m[y].A = data;
      data = data + arr_m[y].M * arr_m[y].N;
    }
  }
}

void node_free(struct Node* node) {
  double* data = node->A[0].A;
  if (data)
    free(data);
  free(node->A);
}

void factorNode(struct Matrix* A_cc, struct Matrix* A_oc, struct Matrix* A_oo, struct Matrix* A, const struct Matrix* Uc, const struct Matrix* Uo, const struct CSC* rels, const struct CellComm* comm) {
  int64_t nnz = rels->ColIndex[rels->N];
  int64_t alen = (int64_t)(A[nnz - 1].A - A[0].A) + A[nnz - 1].M * A[nnz - 1].N;
  double* data = (double*)malloc(sizeof(double) * alen);
  struct Matrix* AV_c = (struct Matrix*)malloc(sizeof(struct Matrix) * nnz * 2);
  struct Matrix* AV_o = &AV_c[nnz];

  for (int64_t x = 0; x < nnz; x++) {
    int64_t dim_y = A[x].M;
    int64_t dim_x = A[x].N;
    int64_t dimc_x = A_cc[x].N;
    int64_t diml_x = dim_x - dimc_x;
    AV_c[x] = (struct Matrix){ data, dim_y, dimc_x };
    AV_o[x] = (struct Matrix){ &data[dim_y * dimc_x], dim_y, diml_x };
    data = data + dim_y * dim_x;
  }
  
  int64_t ibegin = 0, iend = 0;
  self_local_range(&ibegin, &iend, comm);
  int64_t lbegin = ibegin;
  i_global(&lbegin, comm);

#pragma omp parallel for
  for (int64_t x = 0; x < rels->N; x++) {
    for (int64_t yx = rels->ColIndex[x]; yx < rels->ColIndex[x + 1]; yx++) {
      int64_t y = rels->RowIndex[yx];
      i_local(&y, comm);
      mmult('N', 'N', &A[yx], &Uc[x + ibegin], &AV_c[yx], 1., 0.);
      mmult('N', 'N', &A[yx], &Uo[x + ibegin], &AV_o[yx], 1., 0.);
      mmult('T', 'N', &Uc[y], &AV_c[yx], &A_cc[yx], 1., 0.);
      mmult('T', 'N', &Uo[y], &AV_c[yx], &A_oc[yx], 1., 0.);
      mmult('T', 'N', &Uo[y], &AV_o[yx], &A_oo[yx], 1., 0.);
    }  // Skeleton and Redundancy decomposition

    int64_t xx;
    lookupIJ(&xx, rels, x + lbegin, x);
    chol_decomp(&A_cc[xx]); // Factorization

    for (int64_t yx = rels->ColIndex[x]; yx < rels->ColIndex[x + 1]; yx++) {
      int64_t y = rels->RowIndex[yx];
      trsm_lowerA(&A_oc[yx], &A_cc[xx]);
      if (y > x + lbegin)
        trsm_lowerA(&A_cc[yx], &A_cc[xx]);
    } // Lower elimination
    mmult('N', 'T', &A_oc[xx], &A_oc[xx], &A_oo[xx], -1., 1.); // Schur Complement
  }

  free(AV_c[0].A);
  free(AV_c);
}

void nextNode(struct Matrix* Mup, const struct Matrix* Mlow, const int64_t* lchild, const struct CSC* rels_up, const struct CSC* rels_low, const struct CellComm* comm_up, const struct CellComm* comm_low) {
  int64_t nloc = 0, nend = 0, ploc = 0, pend = 0;
  self_local_range(&nloc, &nend, comm_up);
  self_local_range(&ploc, &pend, comm_low);

  for (int64_t j = 0; j < rels_up->N; j++) {
    int64_t cj0 = lchild[j + nloc] - ploc;
    int64_t cj1 = cj0 + 1;

    for (int64_t ij = rels_up->ColIndex[j]; ij < rels_up->ColIndex[j + 1]; ij++) {
      int64_t li = rels_up->RowIndex[ij];
      i_local(&li, comm_up);
      int64_t ci0 = lchild[li];
      i_global(&ci0, comm_low);
      int64_t ci1 = ci0 + 1;

      int64_t i00, i01, i10, i11;
      lookupIJ(&i00, rels_low, ci0, cj0);
      lookupIJ(&i01, rels_low, ci0, cj1);
      lookupIJ(&i10, rels_low, ci1, cj0);
      lookupIJ(&i11, rels_low, ci1, cj1);

      if (i00 >= 0)
        cpyMatToMat(Mlow[i00].M, Mlow[i00].N, &Mlow[i00], &Mup[ij], 0, 0, 0, 0);
      if (i01 >= 0)
        cpyMatToMat(Mlow[i01].M, Mlow[i01].N, &Mlow[i01], &Mup[ij], 0, 0, 0, Mup[ij].N - Mlow[i01].N);
      if (i10 >= 0)
        cpyMatToMat(Mlow[i10].M, Mlow[i10].N, &Mlow[i10], &Mup[ij], 0, 0, Mup[ij].M - Mlow[i10].M, 0);
      if (i11 >= 0)
        cpyMatToMat(Mlow[i11].M, Mlow[i11].N, &Mlow[i11], &Mup[ij], 0, 0, Mup[ij].M - Mlow[i11].M, Mup[ij].N - Mlow[i11].N);
    }
  }
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

void factorA(struct Node A[], const struct Base basis[], const struct CSC rels_near[], const struct CSC rels_far[], const struct CellComm comm[], int64_t levels) {
  for (int64_t i = 1; i <= levels; i++)
    nextNode(A[i - 1].A, A[i].S, basis[i - 1].Lchild, &rels_near[i - 1], &rels_far[i], &comm[i - 1], &comm[i]);

  for (int64_t i = levels; i > 0; i--) {
    factorNode(A[i].A_cc, A[i].A_oc, A[i].A_oo, A[i].A, basis[i].Uc, basis[i].Uo, &rels_near[i], &comm[i]);
    int64_t inxt = i - 1;
    nextNode(A[inxt].A,A[i].A_oo, basis[inxt].Lchild, &rels_near[inxt], &rels_near[i], &comm[inxt], &comm[i]);

    int64_t alst = rels_near[inxt].ColIndex[rels_near[inxt].N] - 1;
    int64_t alen = (int64_t)(A[inxt].A[alst].A - A[inxt].A[0].A) + A[inxt].A[alst].M * A[inxt].A[alst].N;
    merge_double(A[inxt].A[0].A, alen, &comm[inxt]);
  }
  chol_decomp(&A[0].A[0]);
}

void allocRightHandSides(char mvsv, struct RightHandSides rhs[], const struct Base base[], int64_t levels) {
  int use_c = (mvsv == 'S') || (mvsv == 's');
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
      int64_t dimc = use_c ? dim - diml : diml;
      arr_m[j].M = dim; // X
      arr_m[j].N = 1;
      arr_m[j + len].M = dimc; // Xc
      arr_m[j + len].N = 1;
      arr_m[j + len * 2].M = diml; // Xo
      arr_m[j + len * 2].N = 1;
      arr_m[j + len * 3].M = dim; // B
      arr_m[j + len * 3].N = 1;
      count = count + dim * 2 + dimc + diml;
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
  int64_t lbegin = ibegin;
  i_global(&lbegin, comm);

  for (int64_t x = 0; x < rels->N; x++) {
    mmult('T', 'N', &Uc[x + ibegin], &X[x + ibegin], &Xc[x + ibegin], 1., 1.);
    mmult('T', 'N', &Uo[x + ibegin], &X[x + ibegin], &Xo[x + ibegin], 1., 1.);
    int64_t xx;
    lookupIJ(&xx, rels, x + lbegin, x);
    mat_solve('F', &Xc[x + ibegin], &A_cc[xx]);

    for (int64_t yx = rels->ColIndex[x]; yx < rels->ColIndex[x + 1]; yx++) {
      int64_t y = rels->RowIndex[yx];
      int64_t box_y = y;
      i_local(&box_y, comm);
      if (y > x + lbegin)
        mmult('N', 'N', &A_cc[yx], &Xc[x + ibegin], &Xc[box_y], -1., 1.);
      mmult('N', 'N', &A_oc[yx], &Xc[x + ibegin], &Xo[box_y], -1., 1.);
    }
  }
}

void svAccBk(struct Matrix* Xc, const struct Matrix* Xo, struct Matrix* X, const struct Matrix* Uc, const struct Matrix* Uo, const struct Matrix* A_cc, const struct Matrix* A_oc, const struct CSC* rels, const struct CellComm* comm) {
  int64_t ibegin = 0, iend = 0;
  self_local_range(&ibegin, &iend, comm);
  int64_t lbegin = ibegin;
  i_global(&lbegin, comm);

  for (int64_t x = rels->N - 1; x >= 0; x--) {
    for (int64_t yx = rels->ColIndex[x]; yx < rels->ColIndex[x + 1]; yx++) {
      int64_t y = rels->RowIndex[yx];
      int64_t box_y = y;
      i_local(&box_y, comm);
      mmult('T', 'N', &A_oc[yx], &Xo[box_y], &Xc[x + ibegin], -1., 1.);
      if (y > x + lbegin)
        mmult('T', 'N', &A_cc[yx], &Xc[box_y], &Xc[x + ibegin], -1., 1.);
    }

    int64_t xx;
    lookupIJ(&xx, rels, x + lbegin, x);
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
      cpyMatToMat(px[c0].M, 1, &px[c0], &nx[c], 0, 0, 0, 0);
      cpyMatToMat(px[c1].M, 1, &px[c1], &nx[c], 0, 0, nx[c].M - px[c1].M, 0);
    }
  else if (fwbk == 'B' || fwbk == 'b')
    for (int64_t i = 0; i < nboxes; i++) {
      int64_t c = i + nloc;
      int64_t c0 = lchild[c];
      int64_t c1 = c0 + 1;
      cpyMatToMat(px[c0].M, 1, &nx[c], &px[c0], 0, 0, 0, 0);
      cpyMatToMat(px[c1].M, 1, &nx[c], &px[c1], nx[c].M - px[c1].M, 0, 0, 0);
    }
}

void dist_double_svfw(char fwbk, double* arr[], const struct CellComm* comm) {
  int64_t mpi_rank = comm->Proc[2];
  int64_t pbegin = comm->Comms.ColIndex[mpi_rank];
  int64_t plen = comm->Comms.ColIndex[mpi_rank + 1] - pbegin;
  const int64_t* row = &comm->Comms.RowIndex[pbegin];
  double* data = arr[0];
  int is_all = fwbk == 'A' || fwbk == 'a';
#ifdef _PROF
  double stime = MPI_Wtime();
#endif
  for (int64_t i = 0; i < plen; i++) {
    int64_t p = row[i];
    int is_fw = (fwbk == 'F' || fwbk == 'f') && p <= mpi_rank;
    int is_bk = (fwbk == 'B' || fwbk == 'b') && mpi_rank < p;
    if (is_all || is_fw || is_bk) {
      int64_t lbegin = comm->ProcBoxes[p];
      int64_t llen = comm->ProcBoxesEnd[p] - lbegin;
      i_local(&lbegin, comm);
      int64_t offset = arr[lbegin] - data;
      int64_t len = arr[lbegin + llen] - arr[lbegin];
      MPI_Allreduce(MPI_IN_PLACE, &data[offset], len, MPI_DOUBLE, MPI_SUM, comm->Comm_box[p]);
    }
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
  int64_t mpi_rank = comm->Proc[2];
  int64_t pbegin = comm->Comms.ColIndex[mpi_rank];
  int64_t plen = comm->Comms.ColIndex[mpi_rank + 1] - pbegin;
  const int64_t* row = &comm->Comms.RowIndex[pbegin];
  double* data = arr[0];
  int is_all = fwbk == 'A' || fwbk == 'a';
#ifdef _PROF
  double stime = MPI_Wtime();
#endif
  for (int64_t i = plen - 1; i >= 0; i--) {
    int64_t p = row[i];
    int is_fw = (fwbk == 'F' || fwbk == 'f') && p <= mpi_rank;
    int is_bk = (fwbk == 'B' || fwbk == 'b') && mpi_rank < p;
    if (is_all || is_fw || is_bk) {
      int64_t lbegin = comm->ProcBoxes[p];
      int64_t llen = comm->ProcBoxesEnd[p] - lbegin;
      i_local(&lbegin, comm);
      int64_t offset = arr[lbegin] - data;
      int64_t len = arr[lbegin + llen] - arr[lbegin];
      MPI_Bcast(&data[offset], len, MPI_DOUBLE, comm->ProcRootI[p], comm->Comm_box[p]);
    }
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
  cpyMatToMat(rhs[0].X[0].M, 1, &rhs[0].X[0], &rhs[0].B[0], 0, 0, 0, 0);
  mat_solve('A', &rhs[0].B[0], &A[0].A[0]);
  
  for (int64_t i = 1; i <= levels; i++) {
    permuteAndMerge('B', rhs[i].XoL, rhs[i - 1].B, basis[i - 1].Lchild, &comm[i - 1]);
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
    
    svAccBk(rhs[i].XcM, rhs[i].XoL, rhs[i].B, basis[i].Uc, basis[i].Uo, A[i].A_cc, A[i].A_oc, &rels[i], &comm[i]);
    dist_double_svbk('F', arr_comm, &comm[i]);
    free(arr_comm);
  }
  memcpy(X, rhs[levels].B[ibegin].A, lenX * sizeof(double));
}

void horizontalPass(struct Matrix* B, const struct Matrix* X, const struct Matrix* A, const struct CSC* rels, const struct CellComm* comm) {
  int64_t ibegin = 0, iend = 0;
  self_local_range(&ibegin, &iend, comm);
#pragma omp parallel for
  for (int64_t y = 0; y < rels->N; y++)
    for (int64_t xy = rels->ColIndex[y]; xy < rels->ColIndex[y + 1]; xy++) {
      int64_t x = rels->RowIndex[xy];
      i_local(&x, comm);
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

