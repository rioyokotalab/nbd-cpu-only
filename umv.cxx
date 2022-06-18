
#include "umv.h"
#include "dist.h"

void splitA(Matrix* A_out, const CSC& rels, const Matrix* A, const Matrix* U, const Matrix* V, int64_t level) {
  int64_t ibegin = 0, iend = 0;
  selfLocalRange(&ibegin, &iend, level);
  const Matrix* vlocal = &V[ibegin];

#pragma omp parallel for
  for (int64_t x = 0; x < rels.N; x++) {
    for (int64_t yx = rels.COLS_NEAR[x]; yx < rels.COLS_NEAR[x + 1]; yx++) {
      int64_t y = rels.ROWS_NEAR[yx];
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
    for (int64_t yx = rels.COLS_FAR[x]; yx < rels.COLS_FAR[x + 1]; yx++) {
      int64_t y = rels.ROWS_FAR[yx];
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
    lookupIJ('N', ii, rels, i + lbegin, i);
    Matrix& A_ii = A_cc[ii];
    chol_decomp(&A_ii);

    for (int64_t yi = rels.COLS_NEAR[i]; yi < rels.COLS_NEAR[i + 1]; yi++) {
      int64_t y = rels.ROWS_NEAR[yi];
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
    lookupIJ('N', ii, rels, i + lbegin, i);
    const Matrix& A_ii = A_cc[ii];
    for (int64_t yi = rels.COLS_NEAR[i]; yi < rels.COLS_NEAR[i + 1]; yi++)
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
    lookupIJ('N', ii, rels, i + lbegin, i);
    const Matrix& A_ii = A_oc[ii];
    Matrix& S_i = S[ii];
    mmult('N', 'T', &A_ii, &A_ii, &S_i, -1., 1.);
  }
}


void allocNodes(Node* nodes, const CSC rels[], int64_t levels) {
  for (int64_t i = 0; i <= levels; i++) {
    int64_t n_i = rels[i].N;
    int64_t nnz = rels[i].COLS_NEAR[n_i];
    nodes[i].A.resize(nnz);
    nodes[i].A_cc.resize(nnz);
    nodes[i].A_oc.resize(nnz);
    nodes[i].A_oo.resize(nnz);
    int64_t nnz_f = rels[i].COLS_FAR[n_i];
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
  int64_t ibegin = 0, iend;
  selfLocalRange(&ibegin, &iend, level);

  for (int64_t j = 0; j < rels.N; j++) {
    int64_t box_j = ibegin + j;
    int64_t n_j = dims[box_j];

    for (int64_t ij = rels.COLS_NEAR[j]; ij < rels.COLS_NEAR[j + 1]; ij++) {
      int64_t i = rels.ROWS_NEAR[ij];
      int64_t box_i = i;
      iLocal(&box_i, i, level);
      int64_t n_i = dims[box_i];

      Matrix& A_ij = A[ij];
      matrixCreate(&A_ij, n_i, n_j);
      zeroMatrix(&A_ij);
    }
  }
}

void allocS(Matrix* S, const CSC& rels, const int64_t diml[], int64_t level) {
  int64_t ibegin = 0, iend;
  selfLocalRange(&ibegin, &iend, level);

  for (int64_t j = 0; j < rels.N; j++) {
    int64_t box_j = ibegin + j;
    int64_t n_j = diml[box_j];

    for (int64_t ij = rels.COLS_FAR[j]; ij < rels.COLS_FAR[j + 1]; ij++) {
      int64_t i = rels.ROWS_FAR[ij];
      int64_t box_i = i;
      iLocal(&box_i, i, level);
      int64_t n_i = diml[box_i];

      Matrix& S_ij = S[ij];
      matrixCreate(&S_ij, n_i, n_j);
      zeroMatrix(&S_ij);
    }
  }
}

void allocSubMatrices(Node& n, const CSC& rels, const int64_t dims[], const int64_t diml[], int64_t level) {
  int64_t ibegin = 0, iend;
  selfLocalRange(&ibegin, &iend, level);

  for (int64_t j = 0; j < rels.N; j++) {
    int64_t box_j = ibegin + j;
    int64_t diml_j = diml[box_j];
    int64_t dimc_j = dims[box_j] - diml_j;

    for (int64_t ij = rels.COLS_NEAR[j]; ij < rels.COLS_NEAR[j + 1]; ij++) {
      int64_t i = rels.ROWS_NEAR[ij];
      int64_t box_i = i;
      iLocal(&box_i, i, level);
      int64_t diml_i = diml[box_i];
      int64_t dimc_i = dims[box_i] - diml_i;

      matrixCreate(&n.A_cc[ij], dimc_i, dimc_j);
      matrixCreate(&n.A_oc[ij], diml_i, dimc_j);
      matrixCreate(&n.A_oo[ij], diml_i, diml_j);
    }

    for (int64_t ij = rels.COLS_FAR[j]; ij < rels.COLS_FAR[j + 1]; ij++) {
      int64_t i = rels.ROWS_FAR[ij];
      int64_t box_i = i;
      iLocal(&box_i, i, level);
      int64_t diml_i = diml[box_i];
      matrixCreate(&n.S_oo[ij], diml_i, diml_j);
    }
  }
}

void factorNode(Node& n, const Base& basis, const CSC& rels, int64_t level) {
  allocSubMatrices(n, rels, &basis.DIMS[0], &basis.DIML[0], level);
  splitA(n.A_cc.data(), rels, n.A.data(), basis.Uc.data(), basis.Uc.data(), level);
  splitA(n.A_oc.data(), rels, n.A.data(), basis.Uo.data(), basis.Uc.data(), level);
  splitA(n.A_oo.data(), rels, n.A.data(), basis.Uo.data(), basis.Uo.data(), level);
  splitS(n.S_oo.data(), rels, n.S.data(), basis.R.data(), basis.R.data(), level);

  factorAcc(n.A_cc.data(), rels, level);
  factorAoc(n.A_oc.data(), n.A_cc.data(), rels, level);
  schurCmplm(n.A_oo.data(), n.A_oc.data(), rels, level);
}

void nextNode(Node& Anext, const CSC& rels_up, const Node& Aprev, const CSC& rels_low, int64_t nlevel) {
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

    for (int64_t ij = rels_up.COLS_NEAR[j]; ij < rels_up.COLS_NEAR[j + 1]; ij++) {
      int64_t i = rels_up.ROWS_NEAR[ij];
      int64_t ci0 = i << 1;
      int64_t ci1 = (i << 1) + 1;
      int64_t i00, i01, i10, i11;
      lookupIJ('N', i00, rels_low, ci0, cj0);
      lookupIJ('N', i01, rels_low, ci0, cj1);
      lookupIJ('N', i10, rels_low, ci1, cj0);
      lookupIJ('N', i11, rels_low, ci1, cj1);

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

      lookupIJ('F', i00, rels_low, ci0, cj0);
      lookupIJ('F', i01, rels_low, ci0, cj1);
      lookupIJ('F', i10, rels_low, ci1, cj0);
      lookupIJ('F', i11, rels_low, ci1, cj1);

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


void factorA(Node A[], const Base B[], const CSC rels[], int64_t levels) {
  for (int64_t i = levels - 1; i >= 0; i--)
    allocA(A[i].A.data(), rels[i], B[i].DIMS.data(), i);

  for (int64_t i = levels; i > 0; i--) {
    Node& Ai = A[i];
    const Base& Bi = B[i];
    const CSC& ri = rels[i];
    factorNode(Ai, Bi, ri, i);

    Node& An = A[i - 1];
    const CSC& rn = rels[i - 1];
    nextNode(An, rn, Ai, ri, i - 1);
  }
  chol_decomp(&A[0].A[0]);
}

void allocSpDense(SpDense& sp, const Cell* cells, int64_t levels) {
  sp.Levels = levels;
  sp.D.resize(levels + 1);
  sp.Basis.resize(levels + 1);
  sp.Rels.resize(levels + 1);

  relationsNear(&sp.Rels[0], cells, levels);
  allocNodes(sp.D.data(), sp.Rels.data(), levels);
  allocBasis(sp.Basis.data(), levels);
}

void deallocSpDense(SpDense* sp) {
  int64_t level = sp->Levels;
  deallocBasis(&(sp->Basis)[0], level);
  deallocNode(&(sp->D)[0], level);
  
  sp->Levels = 0;
  sp->Basis.clear();
  sp->D.clear();
  sp->Rels.clear();
}

void factorSpDense(SpDense& sp) {
  factorA(&sp.D[0], &sp.Basis[0], &sp.Rels[0], sp.Levels);
}
