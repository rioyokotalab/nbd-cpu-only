
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
    for (int64_t i = lbegin; i < lend; i++)
      pvc_fw(X[i], basis.Uo[i], basis.Uc[i], Xo[i], Xc[i]);
  else if (fwbk == 'B' || fwbk == 'b')
    for (int64_t i = lbegin; i < lend; i++)
      pvc_bk(Xo[i], Xc[i], basis.Uo[i], basis.Uc[i], X[i]);
}


void nbd::svAccFw(Vectors& Xc, const Matrices& A_cc, const CSC& rels, int64_t level) {
  int64_t lbegin = rels.CBGN;
  int64_t ibegin = 0, iend;
  selfLocalRange(ibegin, iend, level);
  recvFwSubstituted(Xc, level);

  Vector* xlocal = &Xc[ibegin];
  for (int64_t i = 0; i < rels.N; i++) {
    int64_t ii;
    lookupIJ(ii, rels, i + lbegin, i + lbegin);
    const Matrix& A_ii = A_cc[ii];
    fw_solve(xlocal[i], A_ii);

    for (int64_t yi = rels.CSC_COLS[i]; yi < rels.CSC_COLS[i + 1]; yi++) {
      int64_t y = rels.CSC_ROWS[yi];
      const Matrix& A_yi = A_cc[yi];
      if (y > i + lbegin) {
        int64_t box_y = y;
        neighborsILocal(box_y, y, level);
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
    for (int64_t yi = rels.CSC_COLS[i]; yi < rels.CSC_COLS[i + 1]; yi++) {
      int64_t y = rels.CSC_ROWS[yi];
      const Matrix& A_yi = A_cc[yi];
      if (y > i + lbegin) {
        int64_t box_y = y;
        neighborsILocal(box_y, y, level);
        mvec('T', A_yi, Xc[box_y], xlocal[i], -1., 1.);
      }
    }

    int64_t ii;
    lookupIJ(ii, rels, i + lbegin, i + lbegin);
    const Matrix& A_ii = A_cc[ii];
    bk_solve(xlocal[i], A_ii);
  }
  
  sendBkSubstituted(Xc, level);
}

void nbd::svAocFw(Vectors& Xo, const Vectors& Xc, const Matrices& A_oc, const CSC& rels, int64_t level) {
  int64_t ibegin = 0, iend;
  selfLocalRange(ibegin, iend, level);
  const Vector* xlocal = &Xc[ibegin];

  for (int64_t x = 0; x < rels.N; x++)
    for (int64_t yx = rels.CSC_COLS[x]; yx < rels.CSC_COLS[x + 1]; yx++) {
      int64_t y = rels.CSC_ROWS[yx];
      int64_t box_y = y;
      neighborsILocal(box_y, y, level);
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
    for (int64_t yx = rels.CSC_COLS[x]; yx < rels.CSC_COLS[x + 1]; yx++) {
      int64_t y = rels.CSC_ROWS[yx];
      int64_t box_y = y;
      neighborsILocal(box_y, y, level);
      const Matrix& A_yx = A_oc[yx];
      mvec('T', A_yx, Xo[box_y], xlocal[x], -1., 1.);
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
      int64_t dim_o = base[i].DIMO[n];
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
  chol_solve(st[0].X[0], A[0].A[0]);
  
  for (int64_t i = 1; i <= levels; i++) {
    permuteAndMerge('B', st[i].Xo, st[i - 1].X, i - 1);
    DistributeVectorsList(st[i].Xo, i);
    svAocBk(st[i].Xc, st[i].Xo, A[i].A_oc, rels[i], i);
    svAccBk(st[i].Xc, A[i].A_cc, rels[i], i);
    basisXoc('B', st[i], B[i], i);
  }
}

void nbd::allocSpDense(SpDense& sp, const CSC rels[], int64_t levels) {
  sp.Levels = levels;
  sp.D.resize(levels + 1);
  sp.Basis.resize(levels + 1);
  sp.Rels = rels;
  allocNodes(sp.D, rels, levels);
  allocBasis(sp.Basis, levels);
}

void nbd::factorSpDense(SpDense& sp, const Cell* local, const Matrices& D, double epi, int64_t mrank, const double* R, int64_t lenR) {
  int64_t levels = sp.Levels;
  fillDimsFromCell(sp.Basis[levels], local, levels);

  int64_t nnz = sp.Rels[levels].NNZ;
  bool cless = true;
  for (int64_t i = 0; i < nnz; i++) {
    Matrix& Ai = sp.D[levels].A[i];
    const Matrix& Di = D[i];
    if (Ai.M != Di.M || Ai.N != Di.N)
      cMatrix(Ai, Di.M, Di.N);
    if (Ai.M > 0 && Ai.N > 0) {
      cpyMatToMat(Di.M, Di.N, Di, Ai, 0, 0, 0, 0);
      cless = false;
    }
  }
  if (!cless)
    factorA(&sp.D[0], &sp.Basis[0], sp.Rels, levels, epi, mrank, R, lenR);
}

void nbd::solveSpDense(RHS st[], const SpDense& sp, const Vectors& X) {
  allocRightHandSides(st, &sp.Basis[0], sp.Levels);
  solveA(st, &sp.D[0], &sp.Basis[0], sp.Rels, X, sp.Levels);
}

void nbd::solveH2(RHS st[], MatVec vx[], const SpDense sps[], EvalFunc ef, const Cell* root, const Base basis[], int64_t dim, const Vectors& X, int64_t levels) {
  solveSpDense(st, sps[levels], X);

  if (levels > 0) {
    h2MatVecLR(vx, ef, root, basis, dim, st[levels].X, levels);

    int64_t xlen = (int64_t)1 << (levels - 1);
    int64_t ibegin = 0;
    int64_t iend = xlen;
    neighborContentLength(xlen, levels - 1);
    selfLocalRange(ibegin, iend, levels - 1);

    Vectors Xi(xlen);
    for (int64_t i = 0; i < xlen; i++) {
      const Vector& vi = vx[levels - 1].B[i];
      if (Xi[i].N != vi.N)
        cVector(Xi[i], vi.N);
      if (i >= ibegin && i < iend)
        cpyVecToVec(vi.N, vi, Xi[i], 0, 0);
    }

    solveH2(st, vx, sps, ef, root, basis, dim, Xi, levels - 1);
    h2MatVecLR(vx, ef, root, basis, dim, st[levels - 1].X, levels - 1);

    for (int64_t i = ibegin; i < iend; i++) {
      Vector& vi = vx[levels - 1].B[i];
      vaxpby(vi, Xi[i].X.data(), 1., -1.);
    }
    permuteAndMerge('B', vx[levels].L, vx[levels - 1].B, levels - 1);
    interTrans('D', vx[levels], basis[levels].Uo, levels);

    ibegin = 0;
    iend = (int64_t)1 << levels;
    selfLocalRange(ibegin, iend, levels);
    for (int64_t i = ibegin; i < iend; i++) {
      Vector& vi = vx[levels].B[i];
      vaxpby(vi, X[i].X.data(), 1., -1.);
    }

    solveSpDense(st, sps[levels], vx[levels].B);
  }
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
