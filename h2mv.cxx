
#include "h2mv.hxx"
#include "dist.hxx"

#include <cmath>

using namespace nbd;

void nbd::interTrans(char updn, MatVec& vx, const Matrices& basis, int64_t level) {
  int64_t len = (int64_t)1 << level;
  int64_t lbegin = 0;
  int64_t lend = len;
  selfLocalRange(lbegin, lend, level);
  Vectors& X = vx.X;
  Vectors& M = vx.M;
  Vectors& L = vx.L;
  Vectors& B = vx.B;

  if (updn == 'U' || updn == 'u')
    for (int64_t i = lbegin; i < lend; i++)
      mvec('T', basis[i], X[i], M[i], 1., 0.);
  else if (updn == 'D' || updn == 'd')
    for (int64_t i = lbegin; i < lend; i++)
      mvec('N', basis[i], L[i], B[i], 1., 0.);
}


void nbd::horizontalPass(Vectors& B, const Vectors& X, EvalFunc ef, const Cell* cell, int64_t dim, int64_t level) {
  int64_t ibegin = 0;
  int64_t iend = (int64_t)1 << level;
  selfLocalRange(ibegin, iend, level);
  int64_t nodes = iend - ibegin;

  int64_t len = 0;
  std::vector<const Cell*> cells(nodes);
  findCellsAtLevel(&cells[0], &len, cell, level);

  for (int64_t i = 0; i < len; i++) {
    const Cell* ci = cells[i];
    int64_t lislen = ci->listNear.size();
    int64_t li = ci->ZID;
    neighborsILocal(li, ci->ZID, level);
    Vector& Bi = B[li];

    for (int64_t j = 0; j < lislen; j++) {
      const Cell* cj = ci->listNear[j];
      int64_t lj = cj->ZID;
      neighborsILocal(lj, cj->ZID, level);
      const Vector& Xj = X[lj];
      M2Lc(ef, ci, cj, dim, Xj, Bi);
    }
  }
}

void nbd::closeQuarter(Vectors& B, const Vectors& X, EvalFunc ef, const Cell* cell, int64_t dim, int64_t level) {
  int64_t ibegin = 0;
  int64_t iend = (int64_t)1 << level;
  selfLocalRange(ibegin, iend, level);
  int64_t nodes = iend - ibegin;

  int64_t len = 0;
  std::vector<const Cell*> cells(nodes);
  findCellsAtLevel(&cells[0], &len, cell, level);

  for (int64_t i = 0; i < len; i++) {
    const Cell* ci = cells[i];
    int64_t lislen = ci->listNear.size();
    int64_t li = ci->ZID;
    neighborsILocal(li, ci->ZID, level);
    Vector& Bi = B[li];

    for (int64_t j = 0; j < lislen; j++) {
      const Cell* cj = ci->listNear[j];
      int64_t lj = cj->ZID;
      neighborsILocal(lj, cj->ZID, level);
      const Vector& Xj = X[lj];
      P2P(ef, ci, cj, dim, Xj, Bi);
    }
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
  neighborsIGlobal(nbegin, nloc, nlevel);
  neighborsIGlobal(pbegin, ploc, plevel);

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

void nbd::allocMatVec(MatVec vx[], const Base base[], int64_t levels) {
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nodes = base[i].DIMS.size();
    MatVec& vx_i = vx[i];
    Vectors& ix = vx_i.X;
    Vectors& im = vx_i.M;
    Vectors& il = vx_i.L;
    Vectors& ib = vx_i.B;
    ix.resize(nodes);
    im.resize(nodes);
    il.resize(nodes);
    ib.resize(nodes);

    for (int64_t n = 0; n < nodes; n++) {
      int64_t dim = base[i].DIMS[n];
      int64_t dim_o = base[i].DIMO[n];
      cVector(ix[n], dim);
      cVector(ib[n], dim);
      cVector(im[n], dim_o);
      cVector(il[n], dim_o);
      zeroVector(ix[n]);
      zeroVector(ib[n]);
      zeroVector(im[n]);
      zeroVector(il[n]);
    }
  }
}

void nbd::resetMatVec(MatVec vx[], const Vectors& X, int64_t levels) {
  for (int64_t i = 0; i <= levels; i++) {
    MatVec& vx_i = vx[i];
    int64_t nodes = vx_i.X.size();
    Vectors& ix = vx_i.X;
    Vectors& im = vx_i.M;
    Vectors& il = vx_i.L;
    Vectors& ib = vx_i.B;

    for (int64_t n = 0; n < nodes; n++) {
      zeroVector(ix[n]);
      zeroVector(ib[n]);
      zeroVector(im[n]);
      zeroVector(il[n]);
    }
  }

  int64_t ibegin = 0;
  int64_t iend = (int64_t)1 << levels;
  selfLocalRange(ibegin, iend, levels);
  Vectors& xleaf = vx[levels].X;
  for (int64_t i = ibegin; i < iend; i++)
    cpyFromVector(X[i], xleaf[i].X.data());
}

void nbd::h2MatVecLR(MatVec vx[], EvalFunc ef, const Cell* root, const Base basis[], int64_t dim, const Vectors& X, int64_t levels) {
  resetMatVec(vx, X, levels);

  for (int64_t i = levels; i > 0; i--) {
    DistributeVectorsList(vx[i].X, i);
    interTrans('U', vx[i], basis[i].Uo, i);
    permuteAndMerge('F', vx[i].M, vx[i - 1].X, i - 1);
  }
  const Cell* local = root;
  horizontalPass(vx[0].B, vx[0].X, ef, local, dim, 0);

  for (int64_t i = 1; i <= levels; i++) {
    permuteAndMerge('B', vx[i].L, vx[i - 1].B, i - 1);
    interTrans('D', vx[i], basis[i].Uo, i);
    local = findLocalAtLevel(local, i);
    horizontalPass(vx[i].B, vx[i].X, ef, local, dim, i);
  }
}

void nbd::h2MatVecAll(MatVec vx[], EvalFunc ef, const Cell* root, const Base basis[], int64_t dim, const Vectors& X, int64_t levels) {
  h2MatVecLR(vx, ef, root, basis, dim, X, levels);
  const Cell* local = findLocalAtLevel(root, levels);
  closeQuarter(vx[levels].B, vx[levels].X, ef, local, dim, levels);
}

void nbd::h2MatVecReference(Vectors& B, EvalFunc ef, const Cell* root, int64_t dim, int64_t levels) {
  Vector X;
  cVector(X, root->NBODY);
  for (int64_t i = 0; i < root->NBODY; i++)
    X.X[i] = root->BODY[i].B;

  int64_t len = 0;
  std::vector<const Cell*> cells((int64_t)1 << levels);
  const Cell* local = findLocalAtLevel(root, levels);
  findCellsAtLevel(&cells[0], &len, local, levels);

  for (int64_t i = 0; i < len; i++) {
    const Cell* ci = cells[i];
    int64_t lislen = ci->listNear.size();
    int64_t li = ci->ZID;
    neighborsILocal(li, ci->ZID, levels);
    Vector& Bi = B[li];
    cVector(Bi, ci->NBODY);
    P2P(ef, ci, root, dim, X, Bi);
  }
}

/*

void woodburyP1(EvalFunc ef, const Cells& cells, int64_t dim, int64_t level, const Matrices& base, double* x1, double* x2, double* c1, double* c2) {
  int64_t len = cells.size();
  int64_t level_m = (int64_t)std::log2(len);

  if (level == level_m)
    upwardPassLeaf(cells, base, x1, c1);
  else {
    X2C(cells, level + 1, x1, c1);
    zeroC(&cells[0], c1, level, level);
    upwardPass(cells, level, base, c1);
  }

  multiplyC(ef, cells, dim, level, base, c1, c2);
  C2X(cells, level, c1, x1);
  C2X(cells, level, c2, x2);
}

void woodburyP2(EvalFunc ef, const Cells& cells, int64_t dim, int64_t level, const Matrices& base, double* x1, double* x2, double* c1, double* c2) {
  int64_t len = cells.size();
  int64_t level_m = (int64_t)std::log2(len);

  X2C(cells, level, x1, c1);
  multiplyC(ef, cells, dim, level, base, c1, c2);

  int64_t lenx;
  if (level == level_m) {
    downwardPassLeaf(cells, base, c2, x1);
    lenx = cells[0].NBODY;
  }
  else {
    zeroC(&cells[0], c2, level + 1, level + 1);
    downwardPass(cells, level, base, c2);
    lenx = C2X(cells, level + 1, c2, x1);
  }
  for (int64_t i = 0; i < lenx; i++)
    x2[i] = x2[i] - x1[i];
}


void h2inv_step(EvalFunc ef, const Cells& cells, int64_t dim, int64_t level, const Matrices& base, const Matrices d[], const CSC rels[], double* x1, double* x2, double* c1, double* c2) {
  Matrix d_dense;
  convertSparse2Dense(rels[level], d[level], d_dense);
  factorD(d_dense);

  int64_t range = (int64_t)1 << level;
  std::vector<int64_t> offs(range);
  range = 0;
  findCellsAtLevel(&offs[0], &range, &cells[0], &cells[0], level);
  int64_t len = 0;
  for (int64_t i = 0; i < range; i++) {
    int64_t ii = offs[i];
    len = len + cells[ii].Multipole.size();
  }

  std::vector<double> x3(len);
  std::vector<double> x4(len);

  invD(d_dense, x2);
  woodburyP1(ef, cells, dim, level, base, x2, &x3[0], c1, c2);
  std::copy(x3.begin(), x3.end(), x4.begin());

  if (level > 1)
    h2inv_step(ef, cells, dim, level - 1, base, d, rels, &x3[0], &x4[0], c1, c2);
  else {
    Matrix last;
    convertSparse2Dense(rels[0], d[0], last);
    factorD(last);
    invD(last, &x3[0]);
  }

  for (int64_t i = 0; i < len; i++)
    x2[i] = x2[i] - x3[i];
  woodburyP2(ef, cells, dim, level, base, x2, x1, c1, c2);
  invD(d_dense, x1);
}

void nbd::h2inv(EvalFunc ef, const Cells& cells, int64_t dim, const Matrices& base, const Matrices d[], const CSC rels[], double* x) {
  int64_t nbodies = cells[0].NBODY;
  std::vector<double> x2(nbodies);
  std::copy(x, x + nbodies, x2.begin());

  int64_t lm = cells.back().MPOS + cells.back().Multipole.size();
  std::vector<double> c1(lm, 0);
  std::vector<double> c2(lm, 0);

  int64_t len = cells.size();
  int64_t level_m = (int64_t)std::log2(len);
  h2inv_step(ef, cells, dim, level_m, base, d, rels, x, &x2[0], &c1[0], &c2[0]);
}*/