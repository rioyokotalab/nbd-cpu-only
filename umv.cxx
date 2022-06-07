
#include "umv.hxx"
#include "dist.hxx"

#include <cstdio>

using namespace nbd;

void nbd::splitA(Matrices& A_out, const CSC& rels, const Matrices& A, const Matrices& U, const Matrices& V, int64_t level) {
  int64_t ibegin = 0, iend;
  selfLocalRange(&ibegin, &iend, level);
  const Matrix* vlocal = &V[ibegin];

#pragma omp parallel for
  for (int64_t x = 0; x < rels.N; x++) {
    for (int64_t yx = rels.COLS_NEAR[x]; yx < rels.COLS_NEAR[x + 1]; yx++) {
      int64_t y = rels.ROWS_NEAR[yx];
      int64_t box_y = y;
      iLocal(&box_y, y, level);
      utav('N', U[box_y], A[yx], vlocal[x], A_out[yx]);
    }
  }
}

void nbd::splitS(Matrices& S_out, const CSC& rels, const Matrices& S, const Matrices& U, const Matrices& V, int64_t level) {
  int64_t ibegin = 0, iend;
  selfLocalRange(&ibegin, &iend, level);
  const Matrix* vlocal = &V[ibegin];

#pragma omp parallel for
  for (int64_t x = 0; x < rels.N; x++) {
    for (int64_t yx = rels.COLS_FAR[x]; yx < rels.COLS_FAR[x + 1]; yx++) {
      int64_t y = rels.ROWS_FAR[yx];
      int64_t box_y = y;
      iLocal(&box_y, y, level);
      utav('T', U[box_y], S[yx], vlocal[x], S_out[yx]);
    }
  }
}

void nbd::factorAcc(Matrices& A_cc, const CSC& rels) {
  int64_t lbegin = rels.CBGN;
#pragma omp parallel for
  for (int64_t i = 0; i < rels.N; i++) {
    int64_t ii;
    lookupIJ('N', ii, rels, i + lbegin, i + lbegin);
    Matrix& A_ii = A_cc[ii];
    chol_decomp(A_ii);

    for (int64_t yi = rels.COLS_NEAR[i]; yi < rels.COLS_NEAR[i + 1]; yi++) {
      int64_t y = rels.ROWS_NEAR[yi];
      if (y > i + lbegin)
        trsm_lowerA(A_cc[yi], A_ii);
    }
  }
}

void nbd::factorAoc(Matrices& A_oc, const Matrices& A_cc, const CSC& rels) {
  int64_t lbegin = rels.CBGN;
#pragma omp parallel for
  for (int64_t i = 0; i < rels.N; i++) {
    int64_t ii;
    lookupIJ('N', ii, rels, i + lbegin, i + lbegin);
    const Matrix& A_ii = A_cc[ii];
    for (int64_t yi = rels.COLS_NEAR[i]; yi < rels.COLS_NEAR[i + 1]; yi++)
      trsm_lowerA(A_oc[yi], A_ii);
  }
}

void nbd::schurCmplm(Matrices& S, const Matrices& A_oc, const CSC& rels) {
  int64_t lbegin = rels.CBGN;
#pragma omp parallel for
  for (int64_t i = 0; i < rels.N; i++) {
    int64_t ii;
    lookupIJ('N', ii, rels, i + lbegin, i + lbegin);
    const Matrix& A_ii = A_oc[ii];
    Matrix& S_i = S[ii];
    mmult('N', 'T', A_ii, A_ii, S_i, -1., 1.);
  }
}


void nbd::allocNodes(Node* nodes, const CSC rels[], int64_t levels) {
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nnz = rels[i].NNZ_NEAR;
    nodes[i].A.resize(nnz);
    nodes[i].A_cc.resize(nnz);
    nodes[i].A_oc.resize(nnz);
    nodes[i].A_oo.resize(nnz);
    int64_t nnz_f = rels[i].NNZ_FAR;
    nodes[i].S.resize(nnz_f);
    nodes[i].S_oo.resize(nnz_f);
  }
}

void nbd::allocA(Matrices& A, const CSC& rels, const int64_t dims[], int64_t level) {
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
      cMatrix(A_ij, n_i, n_j);
      zeroMatrix(A_ij);
    }
  }
}

void nbd::allocS(Matrices& S, const CSC& rels, const int64_t diml[], int64_t level) {
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
      cMatrix(S_ij, n_i, n_j);
      zeroMatrix(S_ij);
    }
  }
}

void nbd::allocSubMatrices(Node& n, const CSC& rels, const int64_t dims[], const int64_t diml[], int64_t level) {
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

      cMatrix(n.A_cc[ij], dimc_i, dimc_j);
      cMatrix(n.A_oc[ij], diml_i, dimc_j);
      cMatrix(n.A_oo[ij], diml_i, diml_j);
    }

    for (int64_t ij = rels.COLS_FAR[j]; ij < rels.COLS_FAR[j + 1]; ij++) {
      int64_t i = rels.ROWS_FAR[ij];
      int64_t box_i = i;
      iLocal(&box_i, i, level);
      int64_t diml_i = diml[box_i];
      cMatrix(n.S_oo[ij], diml_i, diml_j);
    }
  }
}

void nbd::factorNode(Node& n, const Base& basis, const CSC& rels, int64_t level) {
  allocSubMatrices(n, rels, &basis.DIMS[0], &basis.DIML[0], level);
  splitA(n.A_cc, rels, n.A, basis.Uc, basis.Uc, level);
  splitA(n.A_oc, rels, n.A, basis.Uo, basis.Uc, level);
  splitA(n.A_oo, rels, n.A, basis.Uo, basis.Uo, level);
  splitS(n.S_oo, rels, n.S, basis.R, basis.R, level);

  factorAcc(n.A_cc, rels);
  factorAoc(n.A_oc, n.A_cc, rels);
  schurCmplm(n.A_oo, n.A_oc, rels);
}

void nbd::nextNode(Node& Anext, const CSC& rels_up, const Node& Aprev, const CSC& rels_low, int64_t nlevel) {
  Matrices& Mup = Anext.A;
  const Matrices& Mlow = Aprev.A_oo;
  const Matrices& Slow = Aprev.S_oo;

  int64_t nbegin = rels_up.CBGN;
  int64_t clevel = nlevel + 1;

#pragma omp parallel for
  for (int64_t j = 0; j < rels_up.N; j++) {
    int64_t gj = j + nbegin;
    int64_t cj0 = (gj << 1);
    int64_t cj1 = (gj << 1) + 1;

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
        cpyMatToMat(m00.M, m00.N, m00, Mup[ij], 0, 0, 0, 0);
      }

      if (i01 >= 0) {
        const Matrix& m01 = Mlow[i01];
        int64_t xbegin = Mup[ij].N - m01.N;
        cpyMatToMat(m01.M, m01.N, m01, Mup[ij], 0, 0, 0, xbegin);
      }

      if (i10 >= 0) {
        const Matrix& m10 = Mlow[i10];
        int64_t ybegin = Mup[ij].M - m10.M;
        cpyMatToMat(m10.M, m10.N, m10, Mup[ij], 0, 0, ybegin, 0);
      }

      if (i11 >= 0) {
        const Matrix& m11 = Mlow[i11];
        int64_t ybegin = Mup[ij].M - m11.M;
        int64_t xbegin = Mup[ij].N - m11.N;
        cpyMatToMat(m11.M, m11.N, m11, Mup[ij], 0, 0, ybegin, xbegin);
      }

      lookupIJ('F', i00, rels_low, ci0, cj0);
      lookupIJ('F', i01, rels_low, ci0, cj1);
      lookupIJ('F', i10, rels_low, ci1, cj0);
      lookupIJ('F', i11, rels_low, ci1, cj1);

      if (i00 >= 0) {
        const Matrix& m00 = Slow[i00];
        cpyMatToMat(m00.M, m00.N, m00, Mup[ij], 0, 0, 0, 0);
      }

      if (i01 >= 0) {
        const Matrix& m01 = Slow[i01];
        int64_t xbegin = Mup[ij].N - m01.N;
        cpyMatToMat(m01.M, m01.N, m01, Mup[ij], 0, 0, 0, xbegin);
      }

      if (i10 >= 0) {
        const Matrix& m10 = Slow[i10];
        int64_t ybegin = Mup[ij].M - m10.M;
        cpyMatToMat(m10.M, m10.N, m10, Mup[ij], 0, 0, ybegin, 0);
      }

      if (i11 >= 0) {
        const Matrix& m11 = Slow[i11];
        int64_t ybegin = Mup[ij].M - m11.M;
        int64_t xbegin = Mup[ij].N - m11.N;
        cpyMatToMat(m11.M, m11.N, m11, Mup[ij], 0, 0, ybegin, xbegin);
      }
    }
  }
  
  if (rels_low.N == rels_up.N)
    butterflySumA(Mup.data(), Mup.size(), clevel);
}


void nbd::factorA(Node A[], const Base B[], const CSC rels[], int64_t levels) {
  for (int64_t i = levels - 1; i >= 0; i--)
    allocA(A[i].A, rels[i], B[i].DIMS.data(), i);

  for (int64_t i = levels; i > 0; i--) {
    Node& Ai = A[i];
    const Base& Bi = B[i];
    const CSC& ri = rels[i];
    factorNode(Ai, Bi, ri, i);

    Node& An = A[i - 1];
    const CSC& rn = rels[i - 1];
    nextNode(An, rn, Ai, ri, i - 1);
  }
  chol_decomp(A[0].A[0]);
}

void nbd::allocSpDense(SpDense& sp, const CSC rels[], int64_t levels) {
  sp.Levels = levels;
  sp.D.resize(levels + 1);
  sp.Basis.resize(levels + 1);
  allocNodes(sp.D.data(), rels, levels);
  allocBasis(sp.Basis.data(), levels);
}

void nbd::factorSpDense(SpDense& sp, const CSC rels[]) {
  factorA(&sp.D[0], &sp.Basis[0], rels, sp.Levels);
}
