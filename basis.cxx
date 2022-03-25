
#include "basis.hxx"
#include "dist.hxx"

using namespace nbd;

void nbd::sampleC1(Matrices& C1, const CSC& rels, const Matrices& A, const double* R, int64_t lenR, int64_t level) {
  int64_t lbegin = rels.CBGN;
  int64_t ibegin = 0, iend;
  selfLocalRange(ibegin, iend, level);
  for (int64_t j = 0; j < rels.N; j++) {
    Matrix& cj = C1[j + ibegin];

    for (int64_t ij = rels.CSC_COLS[j]; ij < rels.CSC_COLS[j + 1]; ij++) {
      int64_t i = rels.CSC_ROWS[ij];
      if (i == j + lbegin)
        continue;
      const Matrix& Aij = A[ij];
      msample('T', lenR, Aij, R, cj);
    }

    int64_t jj;
    lookupIJ(jj, rels, j + lbegin, j + lbegin);
    const Matrix& Ajj = A[jj];
    minvl(Ajj, cj);
  }
}


void nbd::sampleC2(Matrices& C2, const CSC& rels, const Matrices& A, const Matrices& C1, int64_t level) {
  int64_t lbegin = rels.CBGN;
  int64_t ibegin = 0, iend;
  selfLocalRange(ibegin, iend, level);
  for (int64_t j = 0; j < rels.N; j++) {
    Matrix& cj = C2[j + ibegin];

    for (int64_t ij = rels.CSC_COLS[j]; ij < rels.CSC_COLS[j + 1]; ij++) {
      int64_t i = rels.CSC_ROWS[ij];
      if (i == j + lbegin)
        continue;
      int64_t box_i = i;
      neighborsILocal(box_i, i, level);

      const Matrix& Aij = A[ij];
      msample_m('T', Aij, C1[box_i], cj);
    }
  }
}

void nbd::orthoBasis(double repi, Matrices& C, int64_t dims_o[], int64_t level) {
  int64_t ibegin = 0;
  int64_t iend = C.size();
  selfLocalRange(ibegin, iend, level);
  for (int64_t i = ibegin; i < iend; i++)
    orthoBase(repi, C[i], &dims_o[i]);
}

void nbd::allocBasis(Basis& basis, int64_t levels) {
  basis.resize(levels + 1);
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nodes = (int64_t)1 << i;
    neighborContentLength(nodes, i);
    basis[i].DIMS.resize(nodes);
    basis[i].DIMO.resize(nodes);
    basis[i].Uo.resize(nodes);
    basis[i].Uc.resize(nodes);
  }
}

void nbd::fillDimsFromCell(Base& basis, const Cell* cell, int64_t level) {
  int64_t ibegin = 0;
  int64_t iend = basis.DIMS.size();
  selfLocalRange(ibegin, iend, level);
  int64_t nodes = iend - ibegin;

  int64_t len = 0;
  std::vector<const Cell*> leaves(nodes);
  findCellsAtLevel(&leaves[0], &len, cell, level);

  for (int64_t i = 0; i < len; i++) {
    const Cell* ci = leaves[i];
    int64_t ii = ci->ZID;
    int64_t ni = ci->NBODY;
    if (ci->NCHILD > 0)
      childMultipoleSize(&ni, *ci);
    int64_t mi = ci->Multipole.size();
    int64_t box_i = ii;
    neighborsILocal(box_i, ii, level);

    basis.DIMS[box_i] = ni;
    basis.DIMO[box_i] = mi;
    if (mi > 0) {
      cMatrix(basis.Uo[box_i], ni, mi);
      cMatrix(basis.Uc[box_i], mi, ni);
      cpyMatToMat(ci->Base.M, ci->Base.N, ci->Base, basis.Uo[box_i], 0, 0, 0, 0);
      cpyMatToMat(ci->Biv.M, ci->Biv.N, ci->Biv, basis.Uc[box_i], 0, 0, 0, 0);
    }
  }

  DistributeDims(&basis.DIMS[0], level);
  DistributeDims(&basis.DIMO[0], level);

  int64_t xlen = basis.DIMS.size();
  for (int64_t i = 0; i < xlen; i++) {
    int64_t m = basis.DIMS[i];
    int64_t n = basis.DIMO[i];
    int64_t msize = m * n;
    if (msize > 0) {
      cMatrix(basis.Uo[i], m, n);
      cMatrix(basis.Uc[i], n, m);
    }
  }

  DistributeMatricesList(basis.Uo, level);
  DistributeMatricesList(basis.Uc, level);
}

void nbd::allocUcUo(Base& basis, const Matrices& C, int64_t level) {
  int64_t len = basis.DIMS.size();
  int64_t lbegin = 0;
  int64_t lend = len;
  selfLocalRange(lbegin, lend, level);
  for (int64_t i = 0; i < len; i++) {
    int64_t dim = basis.DIMS[i];
    int64_t dim_o = basis.DIMO[i];
    int64_t dim_c = dim - dim_o;

    Matrix& Uo_i = basis.Uo[i];
    Matrix& Uc_i = basis.Uc[i];
    cMatrix(Uo_i, dim, dim_o);
    cMatrix(Uc_i, dim, dim_c);

    if (i >= lbegin && i < lend) {
      const Matrix& U = C[i];
      cpyMatToMat(dim, dim_o, U, Uo_i, 0, 0, 0, 0);
      cpyMatToMat(dim, dim_c, U, Uc_i, 0, dim_o, 0, 0);
    }
  }
}

void nbd::sampleA(Base& basis, double repi, const CSC& rels, const Matrices& A, const double* R, int64_t lenR, int64_t level) {
  Matrices C1(basis.DIMS.size());
  Matrices C2(basis.DIMS.size());

  int64_t len = basis.DIMS.size();
  for (int64_t i = 0; i < len; i++) {
    int64_t dim = basis.DIMS[i];
    cMatrix(C1[i], dim, dim);
    cMatrix(C2[i], dim, dim);
    zeroMatrix(C1[i]);
    zeroMatrix(C2[i]);
  }

  sampleC1(C1, rels, A, R, lenR, level);
  DistributeMatricesList(C1, level);
  sampleC2(C2, rels, A, C1, level);
  orthoBasis(repi, C2, &basis.DIMO[0], level);
  DistributeDims(&basis.DIMO[0], level);
  allocUcUo(basis, C2, level);
  
  DistributeMatricesList(basis.Uc, level);
  DistributeMatricesList(basis.Uo, level);
}


void nbd::nextBasisDims(Base& bsnext, const Base& bsprev, int64_t nlevel) {
  int64_t ibegin = 0;
  int64_t iend = bsnext.DIMS.size();
  selfLocalRange(ibegin, iend, nlevel);
  int64_t nboxes = iend - ibegin;

  int64_t* dims = &bsnext.DIMS[0];
  const int64_t* dimo = &bsprev.DIMO[0];
  int64_t clevel = nlevel + 1;

  for (int64_t i = 0; i < nboxes; i++) {
    int64_t nloc = i + ibegin;
    int64_t nrnk = nloc;
    neighborsIGlobal(nrnk, nloc, nlevel);

    int64_t c0rnk = nrnk << 1;
    int64_t c1rnk = (nrnk << 1) + 1;
    int64_t c0 = c0rnk;
    int64_t c1 = c1rnk;
    neighborsILocal(c0, c0rnk, clevel);
    neighborsILocal(c1, c1rnk, clevel);

    dims[nloc] = c0 >= 0 ? dimo[c0] : 0;
    dims[nloc] = dims[nloc] + (c1 >= 0 ? dimo[c1] : 0);
  }
  DistributeDims(dims, nlevel);
}

