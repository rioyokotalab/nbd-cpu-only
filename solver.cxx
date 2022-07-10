
#include "solver.h"
#include "dist.h"

#include "math.h"

void basisXoc(char fwbk, RightHandSides& vx, const Base& basis, int64_t level) {
  int64_t len = 0, lbegin = 0, lend = 0;
  selfLocalRange(&lbegin, &lend, level);
  contentLength(&len, level);

  if (fwbk == 'F' || fwbk == 'f')
    for (int64_t i = lbegin; i < lend; i++) {
      mmult('T', 'N', &basis.Uc[i], &vx.X[i], &vx.Xc[i], 1., 0.);
      mmult('T', 'N', &basis.Uo[i], &vx.X[i], &vx.Xo[i], 1., 0.);
    }
  else if (fwbk == 'B' || fwbk == 'b')
    for (int64_t i = lbegin; i < lend; i++) {
      mmult('N', 'N', &basis.Uc[i], &vx.Xc[i], &vx.X[i], 1., 0.);
      mmult('N', 'N', &basis.Uo[i], &vx.Xo[i], &vx.X[i], 1., 1.);
    }
}


void svAccFw(Matrix* Xc, const Matrix* A_cc, const CSC& rels, int64_t level) {
  int64_t ibegin = 0, iend = 0, lbegin = 0;
  selfLocalRange(&ibegin, &iend, level);
  iGlobal(&lbegin, ibegin, level);
  recvFwSubstituted(Xc, level);

  Matrix* xlocal = &Xc[ibegin];
  for (int64_t i = 0; i < rels.N; i++) {
    int64_t ii;
    lookupIJ(&ii, &rels, i + lbegin, i);
    const Matrix& A_ii = A_cc[ii];
    mat_solve('F', &xlocal[i], &A_ii);

    for (int64_t yi = rels.COL_INDEX[i]; yi < rels.COL_INDEX[i + 1]; yi++) {
      int64_t y = rels.ROW_INDEX[yi];
      const Matrix& A_yi = A_cc[yi];
      if (y > i + lbegin) {
        int64_t box_y = y;
        iLocal(&box_y, y, level);
        mmult('N', 'N', &A_yi, &xlocal[i], &Xc[box_y], -1., 1.);
      }
    }
  }

  sendFwSubstituted(Xc, level);
}

void svAccBk(Matrix* Xc, const Matrix* A_cc, const CSC& rels, int64_t level) {
  int64_t ibegin = 0, iend = 0, lbegin = 0;
  selfLocalRange(&ibegin, &iend, level);
  iGlobal(&lbegin, ibegin, level);
  recvBkSubstituted(Xc, level);

  Matrix* xlocal = &Xc[ibegin];
  for (int64_t i = rels.N - 1; i >= 0; i--) {
    for (int64_t yi = rels.COL_INDEX[i]; yi < rels.COL_INDEX[i + 1]; yi++) {
      int64_t y = rels.ROW_INDEX[yi];
      const Matrix& A_yi = A_cc[yi];
      if (y > i + lbegin) {
        int64_t box_y = y;
        iLocal(&box_y, y, level);
        mmult('T', 'N', &A_yi, &Xc[box_y], &xlocal[i], -1., 1.);
      }
    }

    int64_t ii;
    lookupIJ(&ii, &rels, i + lbegin, i);
    const Matrix& A_ii = A_cc[ii];
    mat_solve('B', &xlocal[i], &A_ii);
  }
  
  sendBkSubstituted(Xc, level);
}

void svAocFw(Matrix* Xo, const Matrix* Xc, const Matrix* A_oc, const CSC& rels, int64_t level) {
  int64_t ibegin = 0, iend;
  selfLocalRange(&ibegin, &iend, level);
  const Matrix* xlocal = &Xc[ibegin];

  for (int64_t x = 0; x < rels.N; x++)
    for (int64_t yx = rels.COL_INDEX[x]; yx < rels.COL_INDEX[x + 1]; yx++) {
      int64_t y = rels.ROW_INDEX[yx];
      int64_t box_y = y;
      iLocal(&box_y, y, level);
      const Matrix& A_yx = A_oc[yx];
      mmult('N', 'N', &A_yx, &xlocal[x], &Xo[box_y], -1., 1.);
    }
  distributeSubstituted(Xo, level);
}

void svAocBk(Matrix* Xc, const Matrix* Xo, const Matrix* A_oc, const CSC& rels, int64_t level) {
  int64_t ibegin = 0, iend;
  selfLocalRange(&ibegin, &iend, level);
  Matrix* xlocal = &Xc[ibegin];

  for (int64_t x = 0; x < rels.N; x++)
    for (int64_t yx = rels.COL_INDEX[x]; yx < rels.COL_INDEX[x + 1]; yx++) {
      int64_t y = rels.ROW_INDEX[yx];
      int64_t box_y = y;
      iLocal(&box_y, y, level);
      const Matrix& A_yx = A_oc[yx];
      mmult('T', 'N', &A_yx, &Xo[box_y], &xlocal[x], -1., 1.);
    }
}

void permuteAndMerge(char fwbk, Matrix* px, Matrix* nx, int64_t nlevel) {
  int64_t plevel = nlevel + 1;
  int64_t nloc = 0;
  int64_t nend = (int64_t)1 << nlevel;
  int64_t ploc = 0;
  int64_t pend = (int64_t)1 << plevel;
  selfLocalRange(&nloc, &nend, nlevel);
  selfLocalRange(&ploc, &pend, plevel);

  int64_t nboxes = nend - nloc;
  int64_t pboxes = pend - ploc;
  int64_t nbegin = 0;
  int64_t pbegin = 0;
  iGlobal(&nbegin, nloc, nlevel);
  iGlobal(&pbegin, ploc, plevel);

  if (fwbk == 'F' || fwbk == 'f') {
    for (int64_t i = 0; i < nboxes; i++) {
      int64_t p = (i + nbegin) << 1;
      int64_t c0 = p - pbegin;
      int64_t c1 = p + 1 - pbegin;
      Matrix& x0 = nx[i + nloc];

      if (c0 >= 0 && c0 < pboxes) {
        const Matrix& x1 = px[c0 + ploc];
        cpyMatToMat(x1.M, x0.N, &x1, &x0, 0, 0, 0, 0);
      }

      if (c1 >= 0 && c1 < pboxes) {
        const Matrix& x2 = px[c1 + ploc];
        cpyMatToMat(x2.M, x0.N, &x2, &x0, 0, 0, x0.M - x2.M, 0);
      }
    }
  }
  else if (fwbk == 'B' || fwbk == 'b') {
    for (int64_t i = 0; i < nboxes; i++) {
      int64_t p = (i + nbegin) << 1;
      int64_t c0 = p - pbegin;
      int64_t c1 = p + 1 - pbegin;
      const Matrix& x0 = nx[i + nloc];

      if (c0 >= 0 && c0 < pboxes) {
        Matrix& x1 = px[c0 + ploc];
        cpyMatToMat(x1.M, x0.N, &x0, &x1, 0, 0, 0, 0);
      }

      if (c1 >= 0 && c1 < pboxes) {
        Matrix& x2 = px[c1 + ploc];
        cpyMatToMat(x2.M, x0.N, &x0, &x2, x0.M - x2.M, 0, 0, 0);
      }
    }
  }
}

void allocRightHandSides(RightHandSides st[], const Base base[], int64_t levels) {
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nodes = (int64_t)1 << i;
    contentLength(&nodes, i);

    RightHandSides& rhs_i = st[i];
    rhs_i.Xlen = nodes;
    rhs_i.X.resize(nodes);
    rhs_i.Xc.resize(nodes);
    rhs_i.Xo.resize(nodes);

    for (int64_t n = 0; n < nodes; n++) {
      int64_t dim = base[i].DIMS[n];
      int64_t dim_o = base[i].DIML[n];
      int64_t dim_c = dim - dim_o;
      matrixCreate(&rhs_i.X[n], dim, 1);
      matrixCreate(&rhs_i.Xc[n], dim_c, 1);
      matrixCreate(&rhs_i.Xo[n], dim_o, 1);
      zeroMatrix(&rhs_i.X[n]);
      zeroMatrix(&rhs_i.Xc[n]);
      zeroMatrix(&rhs_i.Xo[n]);
    }
  }
}

void deallocRightHandSides(RightHandSides* st, int64_t levels) {
  for (int i = 0; i <= levels; i++) {
    int64_t nodes = st[i].Xlen;
    for (int64_t n = 0; n < nodes; n++) {
      matrixDestroy(&st[i].X[n]);
      matrixDestroy(&st[i].Xc[n]);
      matrixDestroy(&st[i].Xo[n]);
    }

    st[i].Xlen = 0;
    st[i].X.clear();
    st[i].Xc.clear();
    st[i].Xo.clear();
  }
}

void RightHandSides_mem(int64_t* bytes, const RightHandSides* st, int64_t levels) {
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

void solveA(RightHandSides st[], const Node A[], const Base B[], const CSC rels[], const Matrix* X, int64_t levels) {
  int64_t ibegin = 0;
  int64_t iend = (int64_t)1 << levels;
  selfLocalRange(&ibegin, &iend, levels);

  for (int64_t i = ibegin; i < iend; i++)
    cpyMatToMat(X[i].M, X[i].N, &X[i], &st[levels].X[i], 0, 0, 0, 0);

  for (int64_t i = levels; i > 0; i--) {
    basisXoc('F', st[i], B[i], i);
    svAccFw(st[i].Xc.data(), A[i].A_cc.data(), rels[i], i);
    svAocFw(st[i].Xo.data(), st[i].Xc.data(), A[i].A_oc.data(), rels[i], i);
    permuteAndMerge('F', st[i].Xo.data(), st[i - 1].X.data(), i - 1);

    int comm_needed;
    butterflyComm(&comm_needed, i);
    if (comm_needed)
      butterflySumA(st[i - 1].X.data(), st[i - 1].Xlen, i);
  }
  mat_solve('A', &st[0].X[0], &A[0].A[0]);
  
  for (int64_t i = 1; i <= levels; i++) {
    permuteAndMerge('B', st[i].Xo.data(), st[i - 1].X.data(), i - 1);
    DistributeMatricesList(st[i].Xo.data(), i);
    svAocBk(st[i].Xc.data(), st[i].Xo.data(), A[i].A_oc.data(), rels[i], i);
    svAccBk(st[i].Xc.data(), A[i].A_cc.data(), rels[i], i);
    basisXoc('B', st[i], B[i], i);
  }
}

void solveSpDense(RightHandSides st[], const SpDense& sp, const Matrix* X) {
  allocRightHandSides(st, &sp.Basis[0], sp.Levels);
  solveA(st, &sp.D[0], &sp.Basis[0], &sp.RelsNear[0], X, sp.Levels);
}

void solveRelErr(double* err_out, const Matrix* X, const Matrix* ref, int64_t level) {
  int64_t ibegin = 0;
  int64_t iend = (int64_t)1 << level;
  selfLocalRange(&ibegin, &iend, level);
  double err = 0.;
  double nrm = 0.;

  for (int64_t i = ibegin; i < iend; i++) {
    double e, n;
    Matrix work;
    matrixCreate(&work, X[i].M, X[i].N);

    maxpby(&work, X[i].A, 1., 0.);
    maxpby(&work, ref[i].A, -1., 1.);
    mnrm2(&work, &e);
    mnrm2(&ref[i], &n);

    matrixDestroy(&work);
    err = err + e * e;
    nrm = nrm + n * n;
  }

  *err_out = std::sqrt(err / nrm);
}
