
#include "basis.hxx"
#include "dist.hxx"

#include <unordered_set>
#include <iterator>

using namespace nbd;

void nbd::sampleC1(Matrices& C1, const CSC& rels, const Matrices& A, const double* R, int64_t lenR, int64_t level) {
  int64_t lbegin = rels.CBGN;
  int64_t ibegin = 0, iend;
  selfLocalRange(ibegin, iend, level);
#pragma omp parallel for
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
#pragma omp parallel for
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

void nbd::orthoBasis(double epi, int64_t mrank, Matrices& C, int64_t dims_o[], int64_t level) {
  int64_t ibegin = 0;
  int64_t iend = C.size();
  selfLocalRange(ibegin, iend, level);
  int64_t nodes = iend - ibegin;
#pragma omp parallel for
  for (int64_t i = 0; i < nodes; i++)
    orthoBase(epi, mrank, C[i + ibegin], &dims_o[i + ibegin]);
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

void nbd::evaluateLocal(EvalFunc ef, Base& basis, Cell* cell, int64_t level, const Bodies& bodies, double epi, int64_t mrank, int64_t sp_pts, int64_t dim) {
  int64_t ibegin = 0;
  int64_t iend = basis.DIMS.size();
  selfLocalRange(ibegin, iend, level);
  int64_t nodes = iend - ibegin;

  int64_t len = 0;
  std::vector<Cell*> leaves(nodes);
  findCellsAtLevelModify(&leaves[0], &len, cell, level);

#pragma omp parallel for
  for (int64_t i = 0; i < len; i++) {
    Cell* ci = leaves[i];
    int64_t ii = ci->ZID;
    int64_t box_i = ii;
    neighborsILocal(box_i, ii, level);

    evaluateBasis(ef, basis.Uo[box_i], basis.Uc[box_i], ci, bodies, epi, mrank, sp_pts, dim);
    int64_t ni;
    childMultipoleSize(&ni, *ci);
    int64_t mi = ci->Multipole.size();
    basis.DIMS[box_i] = ni;
    basis.DIMO[box_i] = mi;
  }

  DistributeDims(&basis.DIMS[0], level);
  DistributeDims(&basis.DIMO[0], level);

  int64_t xlen = basis.DIMS.size();
  for (int64_t i = 0; i < xlen; i++) {
    int64_t m = basis.DIMS[i];
    int64_t n = basis.DIMO[i];
    int64_t msize = m * n;
    if (msize > 0 && (i < ibegin || i >= iend)) {
      cMatrix(basis.Uo[i], m, n);
      cMatrix(basis.Uc[i], n, m);
    }
  }

  DistributeMatricesList(basis.Uo, level);
  DistributeMatricesList(basis.Uc, level);
}

void nbd::writeRemoteCoupling(const Base& basis, Cell* cell, int64_t level) {
  int64_t xlen = basis.DIMS.size();
  int64_t ibegin = 0;
  int64_t iend = xlen;
  selfLocalRange(ibegin, iend, level);

  int64_t count = 0;
  std::vector<int64_t> offsets(xlen);
  for (int64_t i = 0; i < xlen; i++) {
    offsets[i] = count;
    count = count + basis.DIMS[i];
  }

  if (count > 0) {
    int64_t len = 0;
    std::vector<Cell*> leaves(xlen);
    std::unordered_set<Cell*> neighbors;
    findCellsAtLevelModify(&leaves[0], &len, cell, level);

    std::vector<int64_t> mps_comm(count);
    for (int64_t i = 0; i < len; i++) {
      const Cell* ci = leaves[i];
      int64_t ii = ci->ZID;
      int64_t box_i = ii;
      neighborsILocal(box_i, ii, level);

      int64_t offset_i = offsets[box_i];
      collectChildMultipoles(*ci, &mps_comm[offset_i]);

      int64_t ni;
      childMultipoleSize(&ni, *ci);
      if (ni != basis.DIMS[box_i])
        butterflyUpdateMultipoles(&mps_comm[offset_i], ni, basis.DIMS[box_i], level + 1);

      int64_t nlen = ci->listNear.size();
      for (int64_t n = 0; n < nlen; n++)
        neighbors.insert(ci->listNear[n]);
    }

    DistributeMultipoles(mps_comm.data(), basis.DIMS.data(), level);

    std::unordered_set<Cell*>::iterator iter = neighbors.begin();
    int64_t nlen = neighbors.size();
    for (int64_t i = 0; i < nlen; i++) {
      Cell* ci = *iter;
      int64_t ii = ci->ZID;
      int64_t box_i = ii;
      neighborsILocal(box_i, ii, level);

      int64_t offset_i = offsets[box_i];
      writeChildMultipoles(*ci, &mps_comm[offset_i], basis.DIMS[box_i]);
      iter = std::next(iter);
    }
  }
}

void nbd::evaluateBaseAll(EvalFunc ef, Base basis[], Cells& cells, int64_t levels, const Bodies& bodies, double epi, int64_t mrank, int64_t sp_pts, int64_t dim) {
  for (int64_t i = levels; i >= 0; i--) {
    Cell* vlocal = findLocalAtLevelModify(&cells[0], i);
    if (i != levels) {
      nextBasisDims(basis[i], basis[i + 1], i);
      writeRemoteCoupling(basis[i], vlocal, i);
    }
    evaluateLocal(ef, basis[i], vlocal, i, bodies, epi, mrank, sp_pts, dim);
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
    int64_t box_i = ii;
    neighborsILocal(box_i, ii, level);

    int64_t ni;
    childMultipoleSize(&ni, *ci);
    basis.DIMS[box_i] = ni;
  }

  DistributeDims(&basis.DIMS[0], level);
  std::fill(basis.DIMO.begin(), basis.DIMO.end(), 0);
}

void nbd::allocUcUo(Base& basis, const Matrices& C, int64_t level) {
  int64_t len = basis.DIMS.size();
  int64_t lbegin = 0;
  int64_t lend = len;
  selfLocalRange(lbegin, lend, level);
#pragma omp parallel for
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

void nbd::sampleA(Base& basis, const CSC& rels, const Matrices& A, double epi, int64_t mrank, const double* R, int64_t lenR, int64_t level) {
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
  orthoBasis(epi, mrank, C2, &basis.DIMO[0], level);
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

