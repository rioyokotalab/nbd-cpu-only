
#include "solver.hxx"
#include "dist.hxx"

using namespace nbd;

void nbd::basisXoc(char fwbk, RHS& vx, const Base& basis, int64_t level) {
  int64_t len = basis.DIMS.size();
  int64_t lbegin = 0;
  int64_t lend = len;
  selfLocalRange(lbegin, lend, level);
  Vectors& X = vx.X;
  Vectors& Xo = vx.Xo;
  Vectors& Xc = vx.Xc;

  if (fwbk == 'F' || fwbk == 'f')
    for (int64_t i = lbegin; i < lend; i++) {
      mvec('T', basis.Uc[i], X[i], Xc[i], 1., 0.);
      mvec('T', basis.Uo[i], X[i], Xo[i], 1., 0.);
    }
  else if (fwbk == 'B' || fwbk == 'b')
    for (int64_t i = lbegin; i < lend; i++) {
      mvec('N', basis.Uc[i], Xc[i], X[i], 1., 0.);
      mvec('N', basis.Uo[i], Xo[i], X[i], 1., 1.);
    }
}


void nbd::svAccFw(Vectors& Xc, const Matrices& A_cc, const CSC& rels, int64_t level) {
  int64_t lbegin = rels.CBGN;
  int64_t ibegin = 0, iend;
  selfLocalRange(ibegin, iend, level);
  recvFwSubstituted(Xc, level);

  Vector* xlocal = &Xc[ibegin];
  for (int64_t i = 0; i < rels.N; i++) {
    int64_t ii;
    lookupIJ('N', ii, rels, i + lbegin, i + lbegin);
    const Matrix& A_ii = A_cc[ii];
    mat_solve('F', xlocal[i], A_ii);

    for (int64_t yi = rels.COLS_NEAR[i]; yi < rels.COLS_NEAR[i + 1]; yi++) {
      int64_t y = rels.ROWS_NEAR[yi];
      const Matrix& A_yi = A_cc[yi];
      if (y > i + lbegin) {
        int64_t box_y = y;
        iLocal(box_y, y, level);
        mvec('N', A_yi, xlocal[i], Xc[box_y], -1., 1.);
      }
    }
  }

  sendFwSubstituted(Xc, level);
}

void nbd::svAccBk(Vectors& Xc, const Matrices& A_cc, const CSC& rels, int64_t level) {
  int64_t lbegin = rels.CBGN;
  int64_t ibegin = 0, iend;
  selfLocalRange(ibegin, iend, level);
  recvBkSubstituted(Xc, level);

  Vector* xlocal = &Xc[ibegin];
  for (int64_t i = rels.N - 1; i >= 0; i--) {
    for (int64_t yi = rels.COLS_NEAR[i]; yi < rels.COLS_NEAR[i + 1]; yi++) {
      int64_t y = rels.ROWS_NEAR[yi];
      const Matrix& A_yi = A_cc[yi];
      if (y > i + lbegin) {
        int64_t box_y = y;
        iLocal(box_y, y, level);
        mvec('T', A_yi, Xc[box_y], xlocal[i], -1., 1.);
      }
    }

    int64_t ii;
    lookupIJ('N', ii, rels, i + lbegin, i + lbegin);
    const Matrix& A_ii = A_cc[ii];
    mat_solve('B', xlocal[i], A_ii);
  }
  
  sendBkSubstituted(Xc, level);
}

void nbd::svAocFw(Vectors& Xo, const Vectors& Xc, const Matrices& A_oc, const CSC& rels, int64_t level) {
  int64_t ibegin = 0, iend;
  selfLocalRange(ibegin, iend, level);
  const Vector* xlocal = &Xc[ibegin];

  for (int64_t x = 0; x < rels.N; x++)
    for (int64_t yx = rels.COLS_NEAR[x]; yx < rels.COLS_NEAR[x + 1]; yx++) {
      int64_t y = rels.ROWS_NEAR[yx];
      int64_t box_y = y;
      iLocal(box_y, y, level);
      const Matrix& A_yx = A_oc[yx];
      mvec('N', A_yx, xlocal[x], Xo[box_y], -1., 1.);
    }
  distributeSubstituted(Xo, level);
}

void nbd::svAocBk(Vectors& Xc, const Vectors& Xo, const Matrices& A_oc, const CSC& rels, int64_t level) {
  int64_t ibegin = 0, iend;
  selfLocalRange(ibegin, iend, level);
  Vector* xlocal = &Xc[ibegin];

  for (int64_t x = 0; x < rels.N; x++)
    for (int64_t yx = rels.COLS_NEAR[x]; yx < rels.COLS_NEAR[x + 1]; yx++) {
      int64_t y = rels.ROWS_NEAR[yx];
      int64_t box_y = y;
      iLocal(box_y, y, level);
      const Matrix& A_yx = A_oc[yx];
      mvec('T', A_yx, Xo[box_y], xlocal[x], -1., 1.);
    }
}

void nbd::permuteAndMerge(char fwbk, Vectors& px, Vectors& nx, int64_t nlevel) {
  int64_t plevel = nlevel + 1;
  int64_t nloc = 0;
  int64_t nend = (int64_t)1 << nlevel;
  int64_t ploc = 0;
  int64_t pend = (int64_t)1 << plevel;
  selfLocalRange(nloc, nend, nlevel);
  selfLocalRange(ploc, pend, plevel);

  int64_t nboxes = nend - nloc;
  int64_t pboxes = pend - ploc;
  int64_t nbegin = 0;
  int64_t pbegin = 0;
  iGlobal(nbegin, nloc, nlevel);
  iGlobal(pbegin, ploc, plevel);

  if (fwbk == 'F' || fwbk == 'f') {
    for (int64_t i = 0; i < nboxes; i++) {
      int64_t p = (i + nbegin) << 1;
      int64_t c0 = p - pbegin;
      int64_t c1 = p + 1 - pbegin;
      Vector& x0 = nx[i + nloc];

      if (c0 >= 0 && c0 < pboxes) {
        const Vector& x1 = px[c0 + ploc];
        cpyVecToVec(x1.N, x1, x0, 0, 0);
      }

      if (c1 >= 0 && c1 < pboxes) {
        const Vector& x2 = px[c1 + ploc];
        cpyVecToVec(x2.N, x2, x0, 0, x0.N - x2.N);
      }
    }

    if (nboxes == pboxes)
      butterflySumX(nx, plevel);
  }
  else if (fwbk == 'B' || fwbk == 'b') {
    for (int64_t i = 0; i < nboxes; i++) {
      int64_t p = (i + nbegin) << 1;
      int64_t c0 = p - pbegin;
      int64_t c1 = p + 1 - pbegin;
      const Vector& x0 = nx[i + nloc];

      if (c0 >= 0 && c0 < pboxes) {
        Vector& x1 = px[c0 + ploc];
        cpyVecToVec(x1.N, x0, x1, 0, 0);
      }

      if (c1 >= 0 && c1 < pboxes) {
        Vector& x2 = px[c1 + ploc];
        cpyVecToVec(x2.N, x0, x2, x0.N - x2.N, 0);
      }
    }
  }
}

void nbd::allocRightHandSides(RHS st[], const Base base[], int64_t levels) {
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nodes = base[i].DIMS.size();
    RHS& rhs_i = st[i];
    Vectors& ix = rhs_i.X;
    Vectors& ixc = rhs_i.Xc;
    Vectors& ixo = rhs_i.Xo;
    ix.resize(nodes);
    ixc.resize(nodes);
    ixo.resize(nodes);

    for (int64_t n = 0; n < nodes; n++) {
      int64_t dim = base[i].DIMS[n];
      int64_t dim_o = base[i].DIML[n];
      int64_t dim_c = dim - dim_o;
      cVector(ix[n], dim);
      cVector(ixc[n], dim_c);
      cVector(ixo[n], dim_o);
      zeroVector(ix[n]);
      zeroVector(ixc[n]);
      zeroVector(ixo[n]);
    }
  }
}

void nbd::solveA(RHS st[], const Node A[], const Base B[], const CSC rels[], const Vectors& X, int64_t levels) {
  int64_t ibegin = 0;
  int64_t iend = (int64_t)1 << levels;
  selfLocalRange(ibegin, iend, levels);

  for (int64_t i = ibegin; i < iend; i++)
    cpyVecToVec(X[i].N, X[i], st[levels].X[i], 0, 0);
  DistributeVectorsList(st[levels].X, levels);

  for (int64_t i = levels; i > 0; i--) {
    basisXoc('F', st[i], B[i], i);
    svAccFw(st[i].Xc, A[i].A_cc, rels[i], i);
    svAocFw(st[i].Xo, st[i].Xc, A[i].A_oc, rels[i], i);
    permuteAndMerge('F', st[i].Xo, st[i - 1].X, i - 1);
  }
  mat_solve('A', st[0].X[0], A[0].A[0]);
  
  for (int64_t i = 1; i <= levels; i++) {
    permuteAndMerge('B', st[i].Xo, st[i - 1].X, i - 1);
    DistributeVectorsList(st[i].Xo, i);
    svAocBk(st[i].Xc, st[i].Xo, A[i].A_oc, rels[i], i);
    svAccBk(st[i].Xc, A[i].A_cc, rels[i], i);
    basisXoc('B', st[i], B[i], i);
  }
}

void nbd::solveSpDense(RHS st[], const SpDense& sp, const Vectors& X) {
  allocRightHandSides(st, &sp.Basis[0], sp.Levels);
  solveA(st, &sp.D[0], &sp.Basis[0], sp.Rels, X, sp.Levels);
}

void nbd::solveRelErr(double* err_out, const Vectors& X, const Vectors& ref, int64_t level) {
  int64_t ibegin = 0;
  int64_t iend = ref.size();
  selfLocalRange(ibegin, iend, level);
  double err = 0.;
  double nrm = 0.;

  for (int64_t i = ibegin; i < iend; i++) {
    double e, n;
    verr2(X[i], ref[i], &e);
    vnrm2(ref[i], &n);
    err = err + e;
    nrm = nrm + n;
  }

  *err_out = err / nrm;
}
