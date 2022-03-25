
#include "umv.hxx"
#include "dist.hxx"

using namespace nbd;

void nbd::splitA(Matrices& A_out, const CSC& rels, const Matrices& A, const Matrices& U, const Matrices& V, int64_t level) {
  int64_t ibegin = 0, iend;
  selfLocalRange(ibegin, iend, level);
  const Matrix* vlocal = &V[ibegin];

  for (int64_t x = 0; x < rels.N; x++) {
    for (int64_t yx = rels.CSC_COLS[x]; yx < rels.CSC_COLS[x + 1]; yx++) {
      int64_t y = rels.CSC_ROWS[yx];
      int64_t box_y = y;
      neighborsILocal(box_y, y, level);
      utav('N', U[box_y], A[yx], vlocal[x], A_out[yx]);
    }
  }
}

void nbd::factorAcc(Matrices& A_cc, const CSC& rels) {
  int64_t lbegin = rels.CBGN;
  for (int64_t i = 0; i < rels.N; i++) {
    int64_t ii;
    lookupIJ(ii, rels, i + lbegin, i + lbegin);
    Matrix& A_ii = A_cc[ii];
    chol_decomp(A_ii);

    for (int64_t yi = rels.CSC_COLS[i]; yi < rels.CSC_COLS[i + 1]; yi++) {
      int64_t y = rels.CSC_ROWS[yi];
      if (y > i + lbegin)
        trsm_lowerA(A_cc[yi], A_ii);
    }
  }
}

void nbd::factorAoc(Matrices& A_oc, const Matrices& A_cc, const CSC& rels) {
  int64_t lbegin = rels.CBGN;

  for (int64_t i = 0; i < rels.N; i++) {
    int64_t ii;
    lookupIJ(ii, rels, i + lbegin, i + lbegin);
    const Matrix& A_ii = A_cc[ii];
    for (int64_t yi = rels.CSC_COLS[i]; yi < rels.CSC_COLS[i + 1]; yi++)
      trsm_lowerA(A_oc[yi], A_ii);
  }
}

void nbd::schurCmplm(Matrices& S, const Matrices& A_oc, const CSC& rels) {
  int64_t lbegin = rels.CBGN;

  for (int64_t i = 0; i < rels.N; i++) {
    int64_t ii;
    lookupIJ(ii, rels, i + lbegin, i + lbegin);
    const Matrix& A_iit = A_oc[ii];
    for (int64_t yi = rels.CSC_COLS[i]; yi < rels.CSC_COLS[i + 1]; yi++) {
      const Matrix& A_yi = A_oc[yi];
      Matrix& S_yi = S[yi];
      mmult('N', 'T', A_yi, A_iit, S_yi, -1., 0.);
    }
  }
}

void nbd::axatLocal(Matrices& A, const CSC& rels) {
  int64_t lbegin = rels.CBGN;
  int64_t lend = lbegin + rels.N;

  for (int64_t i = 0; i < rels.N; i++)
    for (int64_t ji = rels.CSC_COLS[i]; ji < rels.CSC_COLS[i + 1]; ji++) {
      int64_t j = rels.CSC_ROWS[ji];
      if (j > i + lbegin && j < lend) {
        Matrix& A_ji = A[ji];
        int64_t ij;
        lookupIJ(ij, rels, i + lbegin, j);
        Matrix& A_ij = A[ij];
        axat(A_ji, A_ij);
      }
    }
}

void nbd::allocNodes(Nodes& nodes, const CSC rels[], int64_t levels) {
  nodes.resize(levels + 1);
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nnz = rels[i].NNZ;
    nodes[i].A.resize(nnz);
    nodes[i].A_cc.resize(nnz);
    nodes[i].A_oc.resize(nnz);
    nodes[i].A_oo.resize(nnz);
    nodes[i].S.resize(nnz);
  }
}

void nbd::allocA(Matrices& A, const CSC& rels, const int64_t dims[], int64_t level) {
  int64_t ibegin = 0, iend;
  selfLocalRange(ibegin, iend, level);

  for (int64_t j = 0; j < rels.N; j++) {
    int64_t box_j = ibegin + j;
    int64_t nbodies_j = dims[box_j];

    for (int64_t ij = rels.CSC_COLS[j]; ij < rels.CSC_COLS[j + 1]; ij++) {
      int64_t i = rels.CSC_ROWS[ij];
      int64_t box_i = i;
      neighborsILocal(box_i, i, level);
      int64_t nbodies_i = dims[box_i];

      Matrix& A_ij = A[ij];
      cMatrix(A_ij, nbodies_i, nbodies_j);
      zeroMatrix(A_ij);
    }
  }
}

void nbd::allocSubMatrices(Node& n, const CSC& rels, const int64_t dims[], const int64_t dimo[], int64_t level) {
  int64_t ibegin = 0, iend;
  selfLocalRange(ibegin, iend, level);

  for (int64_t j = 0; j < rels.N; j++) {
    int64_t box_j = ibegin + j;
    int64_t dimo_j = dimo[box_j];
    int64_t dimc_j = dims[box_j] - dimo_j;

    for (int64_t ij = rels.CSC_COLS[j]; ij < rels.CSC_COLS[j + 1]; ij++) {
      int64_t i = rels.CSC_ROWS[ij];
      int64_t box_i = i;
      neighborsILocal(box_i, i, level);
      int64_t dimo_i = dimo[box_i];
      int64_t dimc_i = dims[box_i] - dimo_i;

      cMatrix(n.A_cc[ij], dimc_i, dimc_j);
      cMatrix(n.A_oc[ij], dimo_i, dimc_j);
      cMatrix(n.A_oo[ij], dimo_i, dimo_j);
      cMatrix(n.S[ij], dimo_i, dimo_j);
    }
  }
}

void nbd::factorNode(Node& n, Base& basis, const CSC& rels, double repi, const double* R, int64_t lenR, int64_t level) {
  sampleA(basis, repi, rels, n.A, R, lenR, level);
  
  allocSubMatrices(n, rels, &basis.DIMS[0], &basis.DIMO[0], level);
  splitA(n.A_cc, rels, n.A, basis.Uc, basis.Uc, level);
  splitA(n.A_oc, rels, n.A, basis.Uo, basis.Uc, level);
  splitA(n.A_oo, rels, n.A, basis.Uo, basis.Uo, level);

  factorAcc(n.A_cc, rels);
  factorAoc(n.A_oc, n.A_cc, rels);
  schurCmplm(n.S, n.A_oc, rels);

  axatLocal(n.S, rels);
  axatDistribute(n.S, rels, level);

  int64_t len = n.S.size();
  for (int64_t i = 0; i < len; i++)
    madd(n.A_oo[i], n.S[i]);
}

void nbd::nextNode(Node& Anext, Base& bsnext, const CSC& rels_up, const Node& Aprev, const Base& bsprev, const CSC& rels_low, int64_t nlevel) {
  Matrices& Mup = Anext.A;
  const Matrices& Mlow = Aprev.A_oo;

  nextBasisDims(bsnext, bsprev, nlevel);
  allocA(Mup, rels_up, &bsnext.DIMS[0], nlevel);
  int64_t nbegin = rels_up.CBGN;

  for (int64_t j = 0; j < rels_up.N; j++) {
    int64_t gj = j + nbegin;
    int64_t cj0 = (gj << 1);
    int64_t cj1 = (gj << 1) + 1;

    for (int64_t ij = rels_up.CSC_COLS[j]; ij < rels_up.CSC_COLS[j + 1]; ij++) {
      int64_t i = rels_up.CSC_ROWS[ij];
      int64_t ci0 = i << 1;
      int64_t ci1 = (i << 1) + 1;
      int64_t i00, i01, i10, i11;
      lookupIJ(i00, rels_low, ci0, cj0);
      lookupIJ(i01, rels_low, ci0, cj1);
      lookupIJ(i10, rels_low, ci1, cj0);
      lookupIJ(i11, rels_low, ci1, cj1);

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
    }
  }
  
  int64_t clevel = nlevel + 1;
  if (rels_low.N == rels_up.N)
    butterflySumA(Mup, clevel);
}


void nbd::factorA(Node A[], Base B[], const CSC rels[], int64_t levels, double repi, const double* R, int64_t lenR) {
  for (int64_t i = levels; i > 0; i--) {
    Node& Ai = A[i];
    Base& Bi = B[i];
    const CSC& ri = rels[i];
    factorNode(Ai, Bi, ri, repi, R, lenR, i);

    Node& An = A[i - 1];
    Base& Bn = B[i - 1];
    const CSC& rn = rels[i - 1];
    nextNode(An, Bn, rn, Ai, Bi, ri, i - 1);
  }
  chol_decomp(A[0].A[0]);
}
