
#include "nbd.h"
#include "dist.h"

#include "stdlib.h"
#include "math.h"

void allocNodes(struct Node* nodes, const struct Base B[], const struct CSC rels_near[], const struct CSC rels_far[], int64_t levels) {
  for (int64_t i = 0; i <= levels; i++) {
    int64_t n_i = rels_near[i].N;
    int64_t nnz = rels_near[i].COL_INDEX[n_i];
    int64_t nnz_f = rels_far[i].COL_INDEX[n_i];
    struct Matrix* arr_m = (struct Matrix*)malloc(sizeof(struct Matrix) * (nnz * 4 + nnz_f));
    nodes[i].A = arr_m;
    nodes[i].A_cc = &arr_m[nnz];
    nodes[i].A_oc = &arr_m[nnz * 2];
    nodes[i].A_oo = &arr_m[nnz * 3];
    nodes[i].S = &arr_m[nnz * 4];
    nodes[i].lenA = nnz;
    nodes[i].lenS = nnz_f;

    int64_t ibegin = 0, iend = 0;
    selfLocalRange(&ibegin, &iend, i);

    for (int64_t x = 0; x < rels_near[i].N; x++) {
      int64_t box_x = ibegin + x;
      int64_t dim_x = B[i].DIMS[box_x];
      int64_t diml_x = B[i].DIML[box_x];
      int64_t dimc_x = dim_x - diml_x;

      for (int64_t yx = rels_near[i].COL_INDEX[x]; yx < rels_near[i].COL_INDEX[x + 1]; yx++) {
        int64_t y = rels_near[i].ROW_INDEX[yx];
        int64_t box_y = y;
        iLocal(&box_y, y, i);
        int64_t dim_y = B[i].DIMS[box_y];
        int64_t diml_y = B[i].DIML[box_y];
        int64_t dimc_y = dim_y - diml_y;

        matrixCreate(&nodes[i].A[yx], dim_y, dim_x);
        matrixCreate(&nodes[i].A_cc[yx], dimc_y, dimc_x);
        matrixCreate(&nodes[i].A_oc[yx], diml_y, dimc_x);
        matrixCreate(&nodes[i].A_oo[yx], diml_y, diml_x);
        zeroMatrix(&nodes[i].A[yx]);
      }

      for (int64_t yx = rels_far[i].COL_INDEX[x]; yx < rels_far[i].COL_INDEX[x + 1]; yx++) {
        int64_t y = rels_far[i].ROW_INDEX[yx];
        int64_t box_y = y;
        iLocal(&box_y, y, i);
        int64_t diml_y = B[i].DIML[box_y];
        matrixCreate(&nodes[i].S[yx], diml_y, diml_x);
      }
    }
  }
}

void deallocNode(struct Node* node, int64_t levels) {
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nnz = node[i].lenA;
    for (int64_t n = 0; n < nnz; n++) {
      matrixDestroy(&node[i].A[n]);
      matrixDestroy(&node[i].A_cc[n]);
      matrixDestroy(&node[i].A_oc[n]);
      matrixDestroy(&node[i].A_oo[n]);
    }

    int64_t nnz_f = node[i].lenS;
    for (int64_t n = 0; n < nnz_f; n++) {
      matrixDestroy(&node[i].S[n]);
    }

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

void factorNode(struct Matrix* A_cc, struct Matrix* A_oc, struct Matrix* A_oo, const struct Matrix* A, const struct Matrix* Uc, const struct Matrix* Uo, const struct CSC* rels, int64_t level) {
  int64_t ibegin = 0, iend = 0, lbegin = 0;
  selfLocalRange(&ibegin, &iend, level);
  iGlobal(&lbegin, ibegin, level);

  for (int64_t x = 0; x < rels->N; x++) {
    for (int64_t yx = rels->COL_INDEX[x]; yx < rels->COL_INDEX[x + 1]; yx++) {
      int64_t y = rels->ROW_INDEX[yx];
      int64_t box_y = y;
      iLocal(&box_y, y, level);
      utav(&Uc[box_y], &A[yx], &Uc[x + ibegin], &A_cc[yx]);
      utav(&Uo[box_y], &A[yx], &Uc[x + ibegin], &A_oc[yx]);
      utav(&Uo[box_y], &A[yx], &Uo[x + ibegin], &A_oo[yx]);
    }

    int64_t xx;
    lookupIJ(&xx, rels, x + lbegin, x);
    chol_decomp(&A_cc[xx]);

    for (int64_t yx = rels->COL_INDEX[x]; yx < rels->COL_INDEX[x + 1]; yx++) {
      int64_t y = rels->ROW_INDEX[yx];
      trsm_lowerA(&A_oc[yx], &A_cc[xx]);
      if (y > x + lbegin)
        trsm_lowerA(&A_cc[yx], &A_cc[xx]);
    }
    mmult('N', 'T', &A_oc[xx], &A_oc[xx], &A_oo[xx], -1., 1.);
  }
}

void updateS(struct Matrix* S, const struct Matrix* R, const struct CSC* rels, int64_t level) {
  int64_t ibegin = 0, iend = 0, lbegin = 0;
  selfLocalRange(&ibegin, &iend, level);
  iGlobal(&lbegin, ibegin, level);

  for (int64_t x = 0; x < rels->N; x++) {
    for (int64_t yx = rels->COL_INDEX[x]; yx < rels->COL_INDEX[x + 1]; yx++) {
      int64_t y = rels->ROW_INDEX[yx];
      int64_t box_y = y;
      iLocal(&box_y, y, level);
      rsr(&R[box_y], &R[x + ibegin], &S[yx]);
    }
  }
}

void nextNode(struct Matrix* Mup, const struct CSC* rels_up, const struct Matrix* Mlow, const struct Matrix* Slow, const int64_t* lchild, const struct CSC* rels_low_near, const struct CSC* rels_low_far, int64_t nlevel) {
  int64_t plevel = nlevel + 1;
  int64_t nloc = 0, nend = 0, ploc = 0, pend = 0;
  selfLocalRange(&nloc, &nend, nlevel);
  selfLocalRange(&ploc, &pend, plevel);

  for (int64_t j = 0; j < rels_up->N; j++) {
    int64_t cj0 = lchild[j + nloc] - ploc;
    int64_t cj1 = cj0 + 1;

    for (int64_t ij = rels_up->COL_INDEX[j]; ij < rels_up->COL_INDEX[j + 1]; ij++) {
      int64_t i = rels_up->ROW_INDEX[ij];
      int64_t li = i;
      iLocal(&li, i, nlevel);
      int64_t cli0 = lchild[li];
      int64_t ci0 = cli0;
      iGlobal(&ci0, cli0, plevel);
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
  
  int comm_needed;
  butterflyComm(&comm_needed, plevel);
  if (comm_needed)
    butterflySumA(Mup, rels_up->COL_INDEX[rels_up->N], plevel);
}

void factorA(struct Node A[], const struct Base B[], const struct CSC rels_near[], const struct CSC rels_far[], int64_t levels) {
  for (int64_t i = levels; i > 0; i--) {
    factorNode(A[i].A_cc, A[i].A_oc, A[i].A_oo, A[i].A, B[i].Uc, B[i].Uo, &rels_near[i], i);
    updateS(A[i].S, B[i].R, &rels_far[i], i);
    nextNode(A[i - 1].A, &rels_near[i - 1], A[i].A_oo, A[i].S, B[i - 1].Lchild, &rels_near[i], &rels_far[i], i - 1);
  }
  chol_decomp(&A[0].A[0]);
}

void svAccFw(struct Matrix* Xc, struct Matrix* Xo, const struct Matrix* X, const struct Matrix* Uc, const struct Matrix* Uo, const struct Matrix* A_cc, const struct Matrix* A_oc, const struct CSC* rels, int64_t level) {
  int64_t ibegin = 0, iend = 0, lbegin = 0;
  selfLocalRange(&ibegin, &iend, level);
  iGlobal(&lbegin, ibegin, level);
  recvFwSubstituted(Xc, level);

  for (int64_t x = 0; x < rels->N; x++) {
    mmult('T', 'N', &Uc[x + ibegin], &X[x + ibegin], &Xc[x + ibegin], 1., 1.);
    mmult('T', 'N', &Uo[x + ibegin], &X[x + ibegin], &Xo[x + ibegin], 1., 1.);
    int64_t xx;
    lookupIJ(&xx, rels, x + lbegin, x);
    mat_solve('F', &Xc[x + ibegin], &A_cc[xx]);

    for (int64_t yx = rels->COL_INDEX[x]; yx < rels->COL_INDEX[x + 1]; yx++) {
      int64_t y = rels->ROW_INDEX[yx];
      int64_t box_y = y;
      iLocal(&box_y, y, level);
      if (y > x + lbegin)
        mmult('N', 'N', &A_cc[yx], &Xc[x + ibegin], &Xc[box_y], -1., 1.);
      mmult('N', 'N', &A_oc[yx], &Xc[x + ibegin], &Xo[box_y], -1., 1.);
    }
  }

  sendFwSubstituted(Xc, level);
  distributeSubstituted(Xo, level);
}

void svAccBk(struct Matrix* Xc, const struct Matrix* Xo, struct Matrix* X, const struct Matrix* Uc, const struct Matrix* Uo, const struct Matrix* A_cc, const struct Matrix* A_oc, const struct CSC* rels, int64_t level) {
  int64_t ibegin = 0, iend = 0, lbegin = 0;
  selfLocalRange(&ibegin, &iend, level);
  iGlobal(&lbegin, ibegin, level);
  recvBkSubstituted(Xc, level);

  for (int64_t x = rels->N - 1; x >= 0; x--) {
    for (int64_t yx = rels->COL_INDEX[x]; yx < rels->COL_INDEX[x + 1]; yx++) {
      int64_t y = rels->ROW_INDEX[yx];
      int64_t box_y = y;
      iLocal(&box_y, y, level);
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
  
  sendBkSubstituted(Xc, level);
}

void permuteAndMerge(char fwbk, struct Matrix* px, struct Matrix* nx, const int64_t* lchild, int64_t nlevel) {
  int64_t nloc = 0, nend = 0;
  selfLocalRange(&nloc, &nend, nlevel);
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

void allocRightHandSides(struct RightHandSides st[], const struct Base base[], int64_t levels) {
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nodes = base[i].Ulen;
    struct Matrix* arr_m = (struct Matrix*)malloc(sizeof(struct Matrix) * (nodes * 3));
    st[i].Xlen = nodes;
    st[i].X = arr_m;
    st[i].Xc = &arr_m[nodes];
    st[i].Xo = &arr_m[nodes * 2];

    for (int64_t n = 0; n < nodes; n++) {
      int64_t dim = base[i].DIMS[n];
      int64_t dim_o = base[i].DIML[n];
      int64_t dim_c = dim - dim_o;
      matrixCreate(&st[i].X[n], dim, 1);
      matrixCreate(&st[i].Xc[n], dim_c, 1);
      matrixCreate(&st[i].Xo[n], dim_o, 1);
      zeroMatrix(&st[i].X[n]);
      zeroMatrix(&st[i].Xc[n]);
      zeroMatrix(&st[i].Xo[n]);
    }
  }
}

void deallocRightHandSides(struct RightHandSides* st, int64_t levels) {
  for (int i = 0; i <= levels; i++) {
    int64_t nodes = st[i].Xlen;
    for (int64_t n = 0; n < nodes; n++) {
      matrixDestroy(&st[i].X[n]);
      matrixDestroy(&st[i].Xc[n]);
      matrixDestroy(&st[i].Xo[n]);
    }

    free(st[i].X);
  }
}

void RightHandSides_mem(int64_t* bytes, const struct RightHandSides* st, int64_t levels) {
  int64_t count = 0;
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nodes = st[i].Xlen;
    int64_t bytes_x, bytes_o, bytes_c;
    matrix_mem(&bytes_x, &st[i].X[0], nodes);
    matrix_mem(&bytes_o, &st[i].Xo[0], nodes);
    matrix_mem(&bytes_c, &st[i].Xc[0], nodes);

    count = count + bytes_x + bytes_o + bytes_c;
  }
  *bytes = count;
}

void solveA(struct RightHandSides st[], const struct Node A[], const struct Base B[], const struct CSC rels[], const struct Matrix* X, int64_t levels) {
  int64_t ibegin = 0;
  int64_t iend = (int64_t)1 << levels;
  selfLocalRange(&ibegin, &iend, levels);

  for (int64_t i = ibegin; i < iend; i++)
    cpyMatToMat(X[i].M, X[i].N, &X[i], &st[levels].X[i], 0, 0, 0, 0);

  for (int64_t i = levels; i > 0; i--) {
    svAccFw(st[i].Xc, st[i].Xo, st[i].X, B[i].Uc, B[i].Uo, A[i].A_cc, A[i].A_oc, &rels[i], i);
    DistributeMatricesList(st[i].Xo, i);
    permuteAndMerge('F', st[i].Xo, st[i - 1].X, B[i - 1].Lchild, i - 1);
  }
  mat_solve('A', &st[0].X[0], &A[0].A[0]);
  
  for (int64_t i = 1; i <= levels; i++) {
    permuteAndMerge('B', st[i].Xo, st[i - 1].X, B[i - 1].Lchild, i - 1);
    DistributeMatricesList(st[i].Xo, i);
    svAccBk(st[i].Xc, st[i].Xo, st[i].X, B[i].Uc, B[i].Uo, A[i].A_cc, A[i].A_oc, &rels[i], i);
  }
}

