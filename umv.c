
#include "nbd.h"

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
        count += dim_y * dim_x + dimc_y * dimc_x + diml_y * dimc_x + diml_y * diml_x;
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
    
    for (int64_t j = 0; j < len_arr; j++) {
      arr_m[j].A = data;
      int64_t len = arr_m[j].M * arr_m[j].N;
      data = data + len;
    }
  }
}

void deallocNode(struct Node* node, int64_t levels) {
  for (int64_t i = 0; i <= levels; i++) {
    double* data = node[i].A[0].A;
    if (data)
      free(data);
    free(node[i].A);
  }
}

void node_mem(int64_t* bytes, const struct Node* node, int64_t levels) {
  int64_t count = 0;
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nnz = node[i].lenA;
    int64_t nnz_f = node[i].lenS;
    int64_t bytes_a, bytes_cc, bytes_oc, bytes_oo, bytes_s;

    matrix_mem(&bytes_a, &node[i].A[0], nnz);
    matrix_mem(&bytes_cc, &node[i].A_cc[0], nnz);
    matrix_mem(&bytes_oc, &node[i].A_oc[0], nnz);
    matrix_mem(&bytes_oo, &node[i].A_oo[0], nnz);
    matrix_mem(&bytes_s, &node[i].S[0], nnz_f);

    count = count + bytes_a + bytes_cc + bytes_oc + bytes_oo + bytes_s;
  }
  *bytes = count;
}

void factorNode(struct Matrix* A_cc, struct Matrix* A_oc, struct Matrix* A_oo, const struct Matrix* A, const struct Matrix* Uc, const struct Matrix* Uo, const struct CSC* rels, const struct CellComm* comm) {
  int64_t ibegin = 0, iend = 0;
  self_local_range(&ibegin, &iend, comm);
  int64_t lbegin = ibegin;
  i_global(&lbegin, comm);

  for (int64_t x = 0; x < rels->N; x++) {
    for (int64_t yx = rels->ColIndex[x]; yx < rels->ColIndex[x + 1]; yx++) {
      int64_t y = rels->RowIndex[yx];
      i_local(&y, comm);
      utav(&Uc[y], &A[yx], &Uc[x + ibegin], &A_cc[yx]);
      utav(&Uo[y], &A[yx], &Uc[x + ibegin], &A_oc[yx]);
      utav(&Uo[y], &A[yx], &Uo[x + ibegin], &A_oo[yx]);
    }

    int64_t xx;
    lookupIJ(&xx, rels, x + lbegin, x);
    chol_decomp(&A_cc[xx]);

    for (int64_t yx = rels->ColIndex[x]; yx < rels->ColIndex[x + 1]; yx++) {
      int64_t y = rels->RowIndex[yx];
      trsm_lowerA(&A_oc[yx], &A_cc[xx]);
      if (y > x + lbegin)
        trsm_lowerA(&A_cc[yx], &A_cc[xx]);
    }
    mmult('N', 'T', &A_oc[xx], &A_oc[xx], &A_oo[xx], -1., 1.);
  }
}

void nextNode(struct Matrix* Mup, const struct CSC* rels_up, const struct Matrix* Mlow, const struct Matrix* Slow, const int64_t* lchild, 
const struct CSC* rels_low_near, const struct CSC* rels_low_far, const struct CellComm* comm_up, const struct CellComm* comm_low) {
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
      lookupIJ(&i00, rels_low_near, ci0, cj0);
      lookupIJ(&i01, rels_low_near, ci0, cj1);
      lookupIJ(&i10, rels_low_near, ci1, cj0);
      lookupIJ(&i11, rels_low_near, ci1, cj1);

      if (i00 >= 0)
        cpyMatToMat(Mlow[i00].M, Mlow[i00].N, &Mlow[i00], &Mup[ij], 0, 0, 0, 0);
      if (i01 >= 0)
        cpyMatToMat(Mlow[i01].M, Mlow[i01].N, &Mlow[i01], &Mup[ij], 0, 0, 0, Mup[ij].N - Mlow[i01].N);
      if (i10 >= 0)
        cpyMatToMat(Mlow[i10].M, Mlow[i10].N, &Mlow[i10], &Mup[ij], 0, 0, Mup[ij].M - Mlow[i10].M, 0);
      if (i11 >= 0)
        cpyMatToMat(Mlow[i11].M, Mlow[i11].N, &Mlow[i11], &Mup[ij], 0, 0, Mup[ij].M - Mlow[i11].M, Mup[ij].N - Mlow[i11].N);

      lookupIJ(&i00, rels_low_far, ci0, cj0);
      lookupIJ(&i01, rels_low_far, ci0, cj1);
      lookupIJ(&i10, rels_low_far, ci1, cj0);
      lookupIJ(&i11, rels_low_far, ci1, cj1);

      if (i00 >= 0)
        cpyMatToMat(Slow[i00].M, Slow[i00].N, &Slow[i00], &Mup[ij], 0, 0, 0, 0);
      if (i01 >= 0)
        cpyMatToMat(Slow[i01].M, Slow[i01].N, &Slow[i01], &Mup[ij], 0, 0, 0, Mup[ij].N - Slow[i01].N);
      if (i10 >= 0)
        cpyMatToMat(Slow[i10].M, Slow[i10].N, &Slow[i10], &Mup[ij], 0, 0, Mup[ij].M - Slow[i10].M, 0);
      if (i11 >= 0)
        cpyMatToMat(Slow[i11].M, Slow[i11].N, &Slow[i11], &Mup[ij], 0, 0, Mup[ij].M - Slow[i11].M, Mup[ij].N - Slow[i11].N);
    }
  }
}

void merge_double(double* arr, int64_t alen, const struct CellComm* comm) {
  if (comm->Comm_merge != MPI_COMM_NULL) {
    double* arr_sum = (double*)malloc(sizeof(double) * alen);
    MPI_Reduce(arr, arr_sum, alen, MPI_DOUBLE, MPI_SUM, 0, comm->Comm_merge);
    memcpy(arr, arr_sum, sizeof(double) * alen);
    free(arr_sum);
  }

  if (comm->Proc[1] - comm->Proc[0] > 1)
    MPI_Bcast(arr, alen, MPI_DOUBLE, 0, comm->Comm_share);
}

void factorA(struct Node A[], const struct Base basis[], const struct CSC rels_near[], const struct CSC rels_far[], const struct CellComm comm[], int64_t levels) {
  for (int64_t i = levels; i > 0; i--) {
    factorNode(A[i].A_cc, A[i].A_oc, A[i].A_oo, A[i].A, basis[i].Uc, basis[i].Uo, &rels_near[i], &comm[i]);
    nextNode(A[i - 1].A, &rels_near[i - 1], A[i].A_oo, A[i].S, basis[i - 1].Lchild, &rels_near[i], &rels_far[i], &comm[i - 1], &comm[i]);

    double* abegin = A[i - 1].A[0].A;
    int64_t alen = A[i - 1].A_cc[0].A - abegin;
    merge_double(abegin, alen, &comm[i - 1]);
  }
  chol_decomp(&A[0].A[0]);
}

void allocRightHandSides(struct RightHandSides rhs[], const struct Base base[], int64_t levels) {
  for (int64_t i = 0; i <= levels; i++) {
    int64_t len = base[i].Ulen;
    int64_t len_arr = len * 3;
    struct Matrix* arr_m = (struct Matrix*)malloc(sizeof(struct Matrix) * len_arr);
    rhs[i].Xlen = len;
    rhs[i].X = arr_m;
    rhs[i].Xc = &arr_m[len];
    rhs[i].Xo = &arr_m[len * 2];

    int64_t count = 0;
    for (int64_t j = 0; j < len; j++) {
      int64_t dim = base[i].Dims[j];
      int64_t diml = base[i].DimsLr[j];
      int64_t dimc = dim - diml;
      arr_m[j].M = dim; // X
      arr_m[j].N = 1;
      arr_m[j + len].M = dimc; // Xc
      arr_m[j + len].N = 1;
      arr_m[j + len * 2].M = diml; // Xo
      arr_m[j + len * 2].N = 1;
      count += dim + dimc + diml;
    }

    double* data = NULL;
    if (count > 0)
      data = (double*)calloc(count, sizeof(double));
    
    for (int64_t j = 0; j < len_arr; j++) {
      arr_m[j].A = data;
      int64_t len = arr_m[j].M * arr_m[j].N;
      data = data + len;
    }
  }
}

void deallocRightHandSides(struct RightHandSides* rhs, int64_t levels) {
  for (int i = 0; i <= levels; i++) {
    double* data = rhs[i].X[0].A;
    if (data)
      free(data);
    free(rhs[i].X);
  }
}

void RightHandSides_mem(int64_t* bytes, const struct RightHandSides* rhs, int64_t levels) {
  int64_t count = 0;
  for (int64_t i = 0; i <= levels; i++) {
    int64_t len = rhs[i].Xlen;
    int64_t bytes_x, bytes_o, bytes_c;
    matrix_mem(&bytes_x, &rhs[i].X[0], len);
    matrix_mem(&bytes_o, &rhs[i].Xo[0], len);
    matrix_mem(&bytes_c, &rhs[i].Xc[0], len);

    count = count + bytes_x + bytes_o + bytes_c;
  }
  *bytes = count;
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
      cpyMatToMat(px[c0].M, nx[c].N, &px[c0], &nx[c], 0, 0, 0, 0);
      cpyMatToMat(px[c1].M, nx[c].N, &px[c1], &nx[c], 0, 0, nx[c].M - px[c1].M, 0);
    }
  else if (fwbk == 'B' || fwbk == 'b')
    for (int64_t i = 0; i < nboxes; i++) {
      int64_t c = i + nloc;
      int64_t c0 = lchild[c];
      int64_t c1 = c0 + 1;
      cpyMatToMat(px[c0].M, nx[c].N, &nx[c], &px[c0], 0, 0, 0, 0);
      cpyMatToMat(px[c1].M, nx[c].N, &nx[c], &px[c1], nx[c].M - px[c1].M, 0, 0, 0);
    }
}

#include "dist.h"

void solveA(struct RightHandSides rhs[], const struct Node A[], const struct Base basis[], const struct CSC rels[], const struct Matrix* X, const struct CellComm comm[], int64_t levels) {
  int64_t ibegin = 0, iend = 0;
  self_local_range(&ibegin, &iend, &comm[levels]);
  int64_t iboxes = iend - ibegin;

  for (int64_t i = 0; i < iboxes; i++)
    cpyMatToMat(X[i + ibegin].M, X[i + ibegin].N, &X[i + ibegin], &rhs[levels].X[i + ibegin], 0, 0, 0, 0);

  for (int64_t i = levels; i > 0; i--) {
    recvFwSubstituted(rhs[i].Xc, i);
    svAccFw(rhs[i].Xc, rhs[i].Xo, rhs[i].X, basis[i].Uc, basis[i].Uo, A[i].A_cc, A[i].A_oc, &rels[i], &comm[i]);
    sendFwSubstituted(rhs[i].Xc, i);
    distributeSubstituted(rhs[i].Xo, i);
    DistributeMatricesList(rhs[i].Xo, i);
    permuteAndMerge('F', rhs[i].Xo, rhs[i - 1].X, basis[i - 1].Lchild, &comm[i - 1]);
  }
  mat_solve('A', &rhs[0].X[0], &A[0].A[0]);
  
  for (int64_t i = 1; i <= levels; i++) {
    permuteAndMerge('B', rhs[i].Xo, rhs[i - 1].X, basis[i - 1].Lchild, &comm[i - 1]);
    DistributeMatricesList(rhs[i].Xo, i);
    recvBkSubstituted(rhs[i].Xc, i);
    svAccBk(rhs[i].Xc, rhs[i].Xo, rhs[i].X, basis[i].Uc, basis[i].Uo, A[i].A_cc, A[i].A_oc, &rels[i], &comm[i]);
    sendBkSubstituted(rhs[i].Xc, i);
  }
}

