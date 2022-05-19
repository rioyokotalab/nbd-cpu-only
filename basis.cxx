
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

    for (int64_t ij = rels.COLS_NEAR[j]; ij < rels.COLS_NEAR[j + 1]; ij++) {
      int64_t i = rels.ROWS_NEAR[ij];
      if (i == j + lbegin)
        continue;
      const Matrix& Aij = A[ij];
      msample('T', lenR, Aij, R, cj);
    }

    int64_t jj;
    lookupIJ('N', jj, rels, j + lbegin, j + lbegin);
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

    for (int64_t ij = rels.COLS_NEAR[j]; ij < rels.COLS_NEAR[j + 1]; ij++) {
      int64_t i = rels.ROWS_NEAR[ij];
      if (i == j + lbegin)
        continue;
      int64_t box_i = i;
      iLocal(box_i, i, level);

      const Matrix& Aij = A[ij];
      msample_m('T', Aij, C1[box_i], cj);
    }
  }
}

void nbd::orthoBasis(double epi, Matrices& C, Matrices& U, int64_t dims_o[], int64_t level) {
  int64_t ibegin = 0;
  int64_t iend = C.size();
  selfLocalRange(ibegin, iend, level);
  int64_t nodes = iend - ibegin;

#pragma omp parallel for
  for (int64_t i = 0; i < nodes; i++) {
    int64_t ii = i + ibegin;
    dims_o[ii] = 0;
    updateU(epi, C[ii], U[ii], &dims_o[ii]);
  }
}

void nbd::allocBasis(Basis& basis, int64_t levels) {
  basis.resize(levels + 1);
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nodes = (int64_t)1 << i;
    contentLength(nodes, i);
    basis[i].DIMS.resize(nodes);
    basis[i].DIMO.resize(nodes);
    basis[i].DIML.resize(nodes);
    basis[i].Uo.resize(nodes);
    basis[i].Uc.resize(nodes);
    basis[i].Ulr.resize(nodes);
  }
}

void nbd::evaluateBasis(EvalFunc ef, Matrix& Base, Cell* cell, const Bodies& bodies, double epi, int64_t mrank, int64_t sp_pts, int64_t dim) {
  int64_t m;
  childMultipoleSize(&m, *cell);

  Bodies remote;
  remoteBodies(remote, sp_pts, *cell, bodies, dim);
  int64_t n = remote.size();

  if (m > 0 && n > 0) {
    std::vector<int64_t> cellm(m);
    collectChildMultipoles(*cell, cellm.data());
    
    Matrix a;
    cMatrix(a, m, n);
    M2Lmat_bodies(ef, m, n, cellm.data(), nullptr, cell->BODY, remote.data(), dim, a);

    int64_t rank = std::min(m, n);
    rank = mrank > 0 ? std::min(mrank, rank) : rank;
    std::vector<int64_t> pa(rank);
    cMatrix(Base, m, rank);

    int64_t iters;
    lraID(epi, rank, a, Base, pa.data(), &iters);

    if (cell->Multipole.size() != iters)
      cell->Multipole.resize(iters);
    for (int64_t i = 0; i < iters; i++) {
      int64_t ai = pa[i];
      cell->Multipole[i] = cellm[ai];
    }

    if (iters != rank)
      cMatrix(Base, m, iters);
  }
}

void nbd::evaluateLocal(EvalFunc ef, Base& basis, Cell* cell, int64_t level, const Bodies& bodies, double epi, int64_t mrank, int64_t sp_pts, int64_t dim) {
  int64_t xlen = basis.DIMS.size();
  int64_t ibegin = 0;
  int64_t iend = xlen;
  selfLocalRange(ibegin, iend, level);
  int64_t nodes = iend - ibegin;

  int64_t len = 0;
  std::vector<Cell*> leaves(nodes);
  findCellsAtLevelModify(&leaves[0], &len, cell, level);

  std::vector<int64_t>& dims = basis.DIMS;
  std::vector<int64_t>& diml = basis.DIML;

#pragma omp parallel for
  for (int64_t i = 0; i < len; i++) {
    Cell* ci = leaves[i];
    int64_t ii = ci->ZID;
    int64_t box_i = ii;
    iLocal(box_i, ii, level);

    evaluateBasis(ef, basis.Ulr[box_i], ci, bodies, epi, mrank, sp_pts, dim);
    int64_t ni;
    childMultipoleSize(&ni, *ci);
    int64_t mi = ci->Multipole.size();
    dims[box_i] = ni;
    diml[box_i] = mi;
  }

  DistributeDims(&dims[0], level);
  DistributeDims(&diml[0], level);

  for (int64_t i = 0; i < xlen; i++) {
    int64_t m = dims[i];
    int64_t n = diml[i];
    int64_t msize = m * n;
    if (msize > 0 && (i < ibegin || i >= iend))
      cMatrix(basis.Ulr[i], m, n);
  }

  DistributeMatricesList(basis.Ulr, level);
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
    count = count + basis.DIML[i];
  }

  int64_t len = 0;
  std::vector<Cell*> leaves(xlen);
  std::unordered_set<Cell*> neighbors;
  findCellsAtLevelModify(&leaves[0], &len, cell, level);

  std::vector<int64_t> mps_comm(count);
  for (int64_t i = 0; i < len; i++) {
    const Cell* ci = leaves[i];
    int64_t ii = ci->ZID;
    int64_t box_i = ii;
    iLocal(box_i, ii, level);

    int64_t offset_i = offsets[box_i];
    std::copy(ci->Multipole.begin(), ci->Multipole.end(), &mps_comm[offset_i]);

    int64_t nlen = ci->listFar.size();
    for (int64_t n = 0; n < nlen; n++)
      neighbors.emplace((ci->listFar)[n]);
  }

  DistributeMultipoles(mps_comm.data(), basis.DIML.data(), level);

  std::unordered_set<Cell*>::iterator iter = neighbors.begin();
  int64_t nlen = neighbors.size();
  for (int64_t i = 0; i < nlen; i++) {
    Cell* ci = *iter;
    int64_t ii = ci->ZID;
    int64_t box_i = ii;
    iLocal(box_i, ii, level);

    int64_t offset_i = offsets[box_i];
    int64_t end_i = offset_i + basis.DIML[box_i];
    if (ci->Multipole.size() != basis.DIML[box_i])
      ci->Multipole.resize(basis.DIML[box_i]);
    std::copy(&mps_comm[offset_i], &mps_comm[end_i], ci->Multipole.begin());
    iter = std::next(iter);
  }
}

void nbd::evaluateBaseAll(EvalFunc ef, Base basis[], Cells& cells, int64_t levels, const Bodies& bodies, double epi, int64_t mrank, int64_t sp_pts, int64_t dim) {
  int64_t mpi_levels;
  commRank(NULL, NULL, &mpi_levels);

  for (int64_t i = levels; i >= 0; i--) {
    Cell* vlocal = findLocalAtLevelModify(&cells[0], i);
    evaluateLocal(ef, basis[i], vlocal, i, bodies, epi, mrank, sp_pts, dim);
    writeRemoteCoupling(basis[i], vlocal, i);
    
    if (i <= mpi_levels && i > 0) {
      int64_t mlen = vlocal->Multipole.size();
      int64_t msib;
      butterflyUpdateDims(mlen, &msib, i);
      Cell* vsib = vlocal->SIBL;
      if (vsib->Multipole.size() != msib)
        vsib->Multipole.resize(msib);
      butterflyUpdateMultipoles(vlocal->Multipole.data(), mlen, vsib->Multipole.data(), msib, i);
    }
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
    iLocal(box_i, ii, level);

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
    int64_t dim_l = basis.DIML[i];

    Matrix& Uo_i = basis.Uo[i];
    Matrix& Uc_i = basis.Uc[i];
    Matrix& Ul_i = basis.Ulr[i];
    cMatrix(Uo_i, dim, dim_o);
    cMatrix(Uc_i, dim, dim_c);

    if (i >= lbegin && i < lend) {
      const Matrix& U = C[i];
      cpyMatToMat(dim, dim_o, U, Uo_i, 0, 0, 0, 0);
      cpyMatToMat(dim, dim_c, U, Uc_i, 0, dim_o, 0, 0);
    }
    else {
      cMatrix(Ul_i, dim_o, dim_l);
    }
  }
}

void nbd::sampleA(Base& basis, const CSC& rels, const Matrices& A, double epi, int64_t mrank, const double* R, int64_t lenR, int64_t level) {
  Matrices C1(basis.DIMS.size());
  Matrices C2(basis.DIMS.size());

  int64_t len = basis.DIMS.size();
  for (int64_t i = 0; i < len; i++) {
    int64_t dim = basis.DIMS[i];
    cMatrix(C1[i], dim, mrank);
    cMatrix(C2[i], dim, mrank);
    zeroMatrix(C1[i]);
    zeroMatrix(C2[i]);
  }

  sampleC1(C1, rels, A, R, lenR, level);
  DistributeMatricesList(C1, level);
  sampleC2(C2, rels, A, C1, level);
  orthoBasis(epi, C2, basis.Ulr, &basis.DIMO[0], level);
  DistributeDims(&basis.DIML[0], level);
  DistributeDims(&basis.DIMO[0], level);
  allocUcUo(basis, C2, level);
  
  DistributeMatricesList(basis.Uc, level);
  DistributeMatricesList(basis.Uo, level);
  DistributeMatricesList(basis.Ulr, level);
}


void nbd::nextBasisDims(Base& bsnext, const int64_t dimo[], int64_t nlevel) {
  int64_t ibegin = 0;
  int64_t iend = bsnext.DIMS.size();
  selfLocalRange(ibegin, iend, nlevel);
  int64_t nboxes = iend - ibegin;

  int64_t* dims = &bsnext.DIMS[0];
  int64_t clevel = nlevel + 1;

  for (int64_t i = 0; i < nboxes; i++) {
    int64_t nloc = i + ibegin;
    int64_t nrnk = nloc;
    iGlobal(nrnk, nloc, nlevel);

    int64_t c0rnk = nrnk << 1;
    int64_t c1rnk = (nrnk << 1) + 1;
    int64_t c0 = c0rnk;
    int64_t c1 = c1rnk;
    iLocal(c0, c0rnk, clevel);
    iLocal(c1, c1rnk, clevel);

    dims[nloc] = c0 >= 0 ? dimo[c0] : 0;
    dims[nloc] = dims[nloc] + (c1 >= 0 ? dimo[c1] : 0);
  }
  DistributeDims(dims, nlevel);
}

