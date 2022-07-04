
#include "umv.h"
#include "dist.h"

#include "stdlib.h"

void splitA(Matrix* A_out, const CSC& rels, const Matrix* A, const Matrix* U, const Matrix* V, int64_t level) {
  int64_t ibegin = 0, iend = 0;
  selfLocalRange(&ibegin, &iend, level);
  const Matrix* vlocal = &V[ibegin];

#pragma omp parallel for
  for (int64_t x = 0; x < rels.N; x++) {
    for (int64_t yx = rels.COL_INDEX[x]; yx < rels.COL_INDEX[x + 1]; yx++) {
      int64_t y = rels.ROW_INDEX[yx];
      int64_t box_y = y;
      iLocal(&box_y, y, level);
      utav('N', &U[box_y], &A[yx], &vlocal[x], &A_out[yx]);
    }
  }
}

void splitS(Matrix* S_out, const CSC& rels, const Matrix* S, const Matrix* U, const Matrix* V, int64_t level) {
  int64_t ibegin = 0, iend = 0;
  selfLocalRange(&ibegin, &iend, level);
  const Matrix* vlocal = &V[ibegin];

#pragma omp parallel for
  for (int64_t x = 0; x < rels.N; x++) {
    for (int64_t yx = rels.COL_INDEX[x]; yx < rels.COL_INDEX[x + 1]; yx++) {
      int64_t y = rels.ROW_INDEX[yx];
      int64_t box_y = y;
      iLocal(&box_y, y, level);
      utav('T', &U[box_y], &S[yx], &vlocal[x], &S_out[yx]);
    }
  }
}

void factorAcc(Matrix* A_cc, const CSC& rels, int64_t level) {
  int64_t ibegin = 0, iend = 0, lbegin = 0;
  selfLocalRange(&ibegin, &iend, level);
  iGlobal(&lbegin, ibegin, level);

#pragma omp parallel for
  for (int64_t i = 0; i < rels.N; i++) {
    int64_t ii;
    lookupIJ(&ii, &rels, i + lbegin, i);
    Matrix& A_ii = A_cc[ii];
    chol_decomp(&A_ii);

    for (int64_t yi = rels.COL_INDEX[i]; yi < rels.COL_INDEX[i + 1]; yi++) {
      int64_t y = rels.ROW_INDEX[yi];
      if (y > i + lbegin)
        trsm_lowerA(&A_cc[yi], &A_ii);
    }
  }
}

void factorAoc(Matrix* A_oc, const Matrix* A_cc, const CSC& rels, int64_t level) {
  int64_t ibegin = 0, iend = 0, lbegin = 0;
  selfLocalRange(&ibegin, &iend, level);
  iGlobal(&lbegin, ibegin, level);

#pragma omp parallel for
  for (int64_t i = 0; i < rels.N; i++) {
    int64_t ii;
    lookupIJ(&ii, &rels, i + lbegin, i);
    const Matrix& A_ii = A_cc[ii];
    for (int64_t yi = rels.COL_INDEX[i]; yi < rels.COL_INDEX[i + 1]; yi++)
      trsm_lowerA(&A_oc[yi], &A_ii);
  }
}

void schurCmplm(Matrix* S, const Matrix* A_oc, const CSC& rels, int64_t level) {
  int64_t ibegin = 0, iend = 0, lbegin = 0;
  selfLocalRange(&ibegin, &iend, level);
  iGlobal(&lbegin, ibegin, level);

#pragma omp parallel for
  for (int64_t i = 0; i < rels.N; i++) {
    int64_t ii;
    lookupIJ(&ii, &rels, i + lbegin, i);
    const Matrix& A_ii = A_oc[ii];
    Matrix& S_i = S[ii];
    mmult('N', 'T', &A_ii, &A_ii, &S_i, -1., 1.);
  }
}


void allocNodes(Node* nodes, const CSC rels_near[], const CSC rels_far[], int64_t levels) {
  for (int64_t i = 0; i <= levels; i++) {
    int64_t n_i = rels_near[i].N;
    int64_t nnz = rels_near[i].COL_INDEX[n_i];
    nodes[i].A.resize(nnz);
    nodes[i].A_cc.resize(nnz);
    nodes[i].A_oc.resize(nnz);
    nodes[i].A_oo.resize(nnz);
    int64_t nnz_f = rels_far[i].COL_INDEX[n_i];
    nodes[i].S.resize(nnz_f);
    nodes[i].S_oo.resize(nnz_f);
    nodes[i].lenA = nnz;
    nodes[i].lenS = nnz_f;
  }
}

void deallocNode(Node* node, int64_t levels) {
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
      matrixDestroy(&node[i].S_oo[n]);
    }

    node[i].A.clear();
    node[i].A_cc.clear();
    node[i].A_oc.clear();
    node[i].A_oo.clear();
    node[i].S.clear();
    node[i].S_oo.clear();
    node[i].lenA = 0;
    node[i].lenS = 0;
  }
}

void node_mem(int64_t* bytes, const Node* node, int64_t levels) {
  int64_t count = 0;
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nnz = node[i].lenA;
    int64_t nnz_f = node[i].lenS;

    int64_t bytes_a, bytes_cc, bytes_oc, bytes_oo;
    int64_t bytes_s, bytes_soo;

    matrix_mem(&bytes_a, &node[i].A[0], nnz);
    matrix_mem(&bytes_cc, &node[i].A_cc[0], nnz);
    matrix_mem(&bytes_oc, &node[i].A_oc[0], nnz);
    matrix_mem(&bytes_oo, &node[i].A_oo[0], nnz);

    matrix_mem(&bytes_s, &node[i].S[0], nnz_f);
    matrix_mem(&bytes_soo, &node[i].S_oo[0], nnz_f);

    count = count + bytes_a + bytes_cc + bytes_oc + bytes_oo + bytes_s + bytes_soo;
  }
  *bytes = count;
}

void allocA(Matrix* A, const CSC& rels, const int64_t dims[], int64_t level) {
  int64_t ibegin = 0, iend = 0;
  selfLocalRange(&ibegin, &iend, level);

  for (int64_t j = 0; j < rels.N; j++) {
    int64_t box_j = ibegin + j;
    int64_t n_j = dims[box_j];

    for (int64_t ij = rels.COL_INDEX[j]; ij < rels.COL_INDEX[j + 1]; ij++) {
      int64_t i = rels.ROW_INDEX[ij];
      int64_t box_i = i;
      iLocal(&box_i, i, level);
      int64_t n_i = dims[box_i];

      Matrix& A_ij = A[ij];
      matrixCreate(&A_ij, n_i, n_j);
      zeroMatrix(&A_ij);
    }
  }
}

void allocSubMatrices(Node& n, const CSC& rels, const int64_t dims[], const int64_t diml[], int64_t level) {
  int64_t ibegin = 0, iend = 0;
  selfLocalRange(&ibegin, &iend, level);

  for (int64_t j = 0; j < rels.N; j++) {
    int64_t box_j = ibegin + j;
    int64_t diml_j = diml[box_j];
    int64_t dimc_j = dims[box_j] - diml_j;

    for (int64_t ij = rels.COL_INDEX[j]; ij < rels.COL_INDEX[j + 1]; ij++) {
      int64_t i = rels.ROW_INDEX[ij];
      int64_t box_i = i;
      iLocal(&box_i, i, level);
      int64_t diml_i = diml[box_i];
      int64_t dimc_i = dims[box_i] - diml_i;

      matrixCreate(&n.A_cc[ij], dimc_i, dimc_j);
      matrixCreate(&n.A_oc[ij], diml_i, dimc_j);
      matrixCreate(&n.A_oo[ij], diml_i, diml_j);
    }
  }
}

void factorNode(Node& n, const Base& basis, const CSC& rels_near, const CSC& rels_far, int64_t level) {
  allocSubMatrices(n, rels_near, &basis.DIMS[0], &basis.DIML[0], level);
  allocA(n.S_oo.data(), rels_far, &basis.DIML[0], level);

  splitA(n.A_cc.data(), rels_near, n.A.data(), basis.Uc.data(), basis.Uc.data(), level);
  splitA(n.A_oc.data(), rels_near, n.A.data(), basis.Uo.data(), basis.Uc.data(), level);
  splitA(n.A_oo.data(), rels_near, n.A.data(), basis.Uo.data(), basis.Uo.data(), level);
  splitS(n.S_oo.data(), rels_far, n.S.data(), basis.R.data(), basis.R.data(), level);

  factorAcc(n.A_cc.data(), rels_near, level);
  factorAoc(n.A_oc.data(), n.A_cc.data(), rels_near, level);
  schurCmplm(n.A_oo.data(), n.A_oc.data(), rels_near, level);
}

void nextNode(Node& Anext, const CSC& rels_up, const Node& Aprev, const CSC& rels_low_near, const CSC& rels_low_far, int64_t nlevel) {
  Matrix* Mup = Anext.A.data();
  const Matrix* Mlow = Aprev.A_oo.data();
  const Matrix* Slow = Aprev.S_oo.data();

  int64_t plevel = nlevel + 1;
  int64_t nloc = 0;
  int64_t nend = (int64_t)1 << nlevel;
  int64_t ploc = 0;
  int64_t pend = (int64_t)1 << plevel;
  selfLocalRange(&nloc, &nend, nlevel);
  selfLocalRange(&ploc, &pend, plevel);

  int64_t nbegin = 0;
  int64_t pbegin = 0;
  iGlobal(&nbegin, nloc, nlevel);
  iGlobal(&pbegin, ploc, plevel);

#pragma omp parallel for
  for (int64_t j = 0; j < rels_up.N; j++) {
    int64_t cj = (j + nbegin) << 1;
    int64_t cj0 = cj - pbegin;
    int64_t cj1 = cj + 1 - pbegin;

    for (int64_t ij = rels_up.COL_INDEX[j]; ij < rels_up.COL_INDEX[j + 1]; ij++) {
      int64_t i = rels_up.ROW_INDEX[ij];
      int64_t ci0 = i << 1;
      int64_t ci1 = (i << 1) + 1;
      int64_t i00, i01, i10, i11;
      lookupIJ(&i00, &rels_low_near, ci0, cj0);
      lookupIJ(&i01, &rels_low_near, ci0, cj1);
      lookupIJ(&i10, &rels_low_near, ci1, cj0);
      lookupIJ(&i11, &rels_low_near, ci1, cj1);

      if (i00 >= 0) {
        const Matrix& m00 = Mlow[i00];
        cpyMatToMat(m00.M, m00.N, &m00, &Mup[ij], 0, 0, 0, 0);
      }

      if (i01 >= 0) {
        const Matrix& m01 = Mlow[i01];
        int64_t xbegin = Mup[ij].N - m01.N;
        cpyMatToMat(m01.M, m01.N, &m01, &Mup[ij], 0, 0, 0, xbegin);
      }

      if (i10 >= 0) {
        const Matrix& m10 = Mlow[i10];
        int64_t ybegin = Mup[ij].M - m10.M;
        cpyMatToMat(m10.M, m10.N, &m10, &Mup[ij], 0, 0, ybegin, 0);
      }

      if (i11 >= 0) {
        const Matrix& m11 = Mlow[i11];
        int64_t ybegin = Mup[ij].M - m11.M;
        int64_t xbegin = Mup[ij].N - m11.N;
        cpyMatToMat(m11.M, m11.N, &m11, &Mup[ij], 0, 0, ybegin, xbegin);
      }

      lookupIJ(&i00, &rels_low_far, ci0, cj0);
      lookupIJ(&i01, &rels_low_far, ci0, cj1);
      lookupIJ(&i10, &rels_low_far, ci1, cj0);
      lookupIJ(&i11, &rels_low_far, ci1, cj1);

      if (i00 >= 0) {
        const Matrix& m00 = Slow[i00];
        cpyMatToMat(m00.M, m00.N, &m00, &Mup[ij], 0, 0, 0, 0);
      }

      if (i01 >= 0) {
        const Matrix& m01 = Slow[i01];
        int64_t xbegin = Mup[ij].N - m01.N;
        cpyMatToMat(m01.M, m01.N, &m01, &Mup[ij], 0, 0, 0, xbegin);
      }

      if (i10 >= 0) {
        const Matrix& m10 = Slow[i10];
        int64_t ybegin = Mup[ij].M - m10.M;
        cpyMatToMat(m10.M, m10.N, &m10, &Mup[ij], 0, 0, ybegin, 0);
      }

      if (i11 >= 0) {
        const Matrix& m11 = Slow[i11];
        int64_t ybegin = Mup[ij].M - m11.M;
        int64_t xbegin = Mup[ij].N - m11.N;
        cpyMatToMat(m11.M, m11.N, &m11, &Mup[ij], 0, 0, ybegin, xbegin);
      }
    }
  }
  
  int comm_needed;
  butterflyComm(&comm_needed, plevel);
  if (comm_needed)
    butterflySumA(Mup, Anext.lenA, plevel);
}


void factorA(Node A[], const Base B[], const CSC rels_near[], const CSC rels_far[], int64_t levels) {
  for (int64_t i = levels - 1; i >= 0; i--)
    allocA(A[i].A.data(), rels_near[i], B[i].DIMS.data(), i);

  for (int64_t i = levels; i > 0; i--) {
    Node& Ai = A[i];
    const Base& Bi = B[i];
    factorNode(Ai, Bi, rels_near[i], rels_far[i], i);

    Node& An = A[i - 1];
    nextNode(An, rels_near[i - 1], Ai, rels_near[i], rels_far[i], i - 1);
  }
  chol_decomp(&A[0].A[0]);
}

void allocSpDense(SpDense& sp, int64_t levels) {
  sp.Levels = levels;
  sp.D.resize(levels + 1);
  sp.Basis.resize(levels + 1);
  sp.RelsNear.resize(levels + 1);
  sp.RelsFar.resize(levels + 1);
}

void deallocSpDense(SpDense* sp) {
  int64_t level = sp->Levels;
  deallocBasis(&(sp->Basis)[0], level);
  deallocNode(&(sp->D)[0], level);
  
  sp->Levels = 0;
  sp->Basis.clear();
  sp->D.clear();

  for (int64_t i = 0; i <= level; i++) {
    free(sp->RelsFar[i].COL_INDEX);
    free(sp->RelsNear[i].COL_INDEX);
  }
  sp->RelsNear.clear();
  sp->RelsFar.clear();
}

void factorSpDense(SpDense& sp) {
  factorA(&sp.D[0], &sp.Basis[0], &sp.RelsNear[0], &sp.RelsFar[0], sp.Levels);
}
