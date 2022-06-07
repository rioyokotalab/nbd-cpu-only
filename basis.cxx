
#include "basis.hxx"
#include "dist.hxx"

#include <unordered_set>
#include <iterator>

using namespace nbd;

void nbd::allocBasis(Base* basis, int64_t levels) {
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nodes = (int64_t)1 << i;
    contentLength(&nodes, i);
    basis[i].DIMS.resize(nodes);
    basis[i].DIML.resize(nodes);
    basis[i].Uo.resize(nodes);
    basis[i].Uc.resize(nodes);
    basis[i].R.resize(nodes);
  }
}

void nbd::evaluateBasis(KerFunc_t ef, Matrix& Base, Cell* cell, const Body* bodies, int64_t nbodies, double epi, int64_t mrank, int64_t sp_pts) {
  int64_t m;
  childMultipoleSize(&m, *cell);

  if (m > 0) {
    std::vector<Body> remote(sp_pts);
    std::vector<Body> close(sp_pts);
    int64_t n1 = remoteBodies(remote.data(), sp_pts, *cell, bodies, nbodies);
    int64_t n2 = closeBodies(close.data(), sp_pts, *cell);

    std::vector<int64_t> cellm(m);
    collectChildMultipoles(*cell, cellm.data());
    
    Matrix work_a, work_b, work_c, work_s;
    int64_t len_s = n1 + (n2 > 0 ? m : 0);
    if (len_s > 0)
      cMatrix(work_s, m, len_s);

    if (n1 > 0) {
      cMatrix(work_a, m, n1);
      gen_matrix(ef, m, n1, cell->BODY, remote.data(), work_a.A.data(), cellm.data(), NULL);
      cpyMatToMat(m, n1, work_a, work_s, 0, 0, 0, 0);
    }

    if (n2 > 0) {
      cMatrix(work_b, m, n2);
      cMatrix(work_c, m, m);
      gen_matrix(ef, m, n2, cell->BODY, close.data(), work_b.A.data(), cellm.data(), NULL);
      mmult('N', 'T', work_b, work_b, work_c, 1., 0.);
      if (n1 > 0)
        normalizeA(work_c, work_a);
      cpyMatToMat(m, m, work_c, work_s, 0, 0, 0, n1);
      cMatrix(work_b, 0, 0);
      cMatrix(work_c, 0, 0);
    }

    if (n1 > 0)
      cMatrix(work_a, 0, 0);

    if (len_s > 0) {
      int64_t rank = m;
      rank = mrank > 0 ? std::min(mrank, rank) : rank;
      std::vector<int64_t> pa(rank);
      cMatrix(Base, m, rank);

      int64_t iters = rank;
      lraID(epi, work_s, Base, pa.data(), &iters);
      cMatrix(work_s, 0, 0);

      int64_t len_m = cell->Multipole.size();
      if (len_m != iters)
        cell->Multipole.resize(iters);
      for (int64_t i = 0; i < iters; i++) {
        int64_t ai = pa[i];
        cell->Multipole[i] = cellm[ai];
      }

      if (iters != rank)
        cMatrix(Base, m, iters);
    }
  }
}

void nbd::evaluateLocal(KerFunc_t ef, Base& basis, Cell* cell, int64_t level, const Body* bodies, int64_t nbodies, double epi, int64_t mrank, int64_t sp_pts) {
  int64_t xlen = basis.DIMS.size();
  int64_t ibegin = 0;
  int64_t iend = xlen;
  selfLocalRange(&ibegin, &iend, level);
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
    iLocal(&box_i, ii, level);

    evaluateBasis(ef, basis.Uo[box_i], ci, bodies, nbodies, epi, mrank, sp_pts);
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
      cMatrix(basis.Uo[i], m, n);
  }
  DistributeMatricesList(basis.Uo.data(), level);
}

void nbd::writeRemoteCoupling(const Base& basis, Cell* cell, int64_t level) {
  int64_t xlen = basis.DIMS.size();
  int64_t ibegin = 0;
  int64_t iend = xlen;
  selfLocalRange(&ibegin, &iend, level);

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
    iLocal(&box_i, ii, level);

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
    iLocal(&box_i, ii, level);

    int64_t offset_i = offsets[box_i];
    int64_t end_i = offset_i + basis.DIML[box_i];
    int64_t len_m = ci->Multipole.size();
    if (len_m != basis.DIML[box_i])
      ci->Multipole.resize(basis.DIML[box_i]);
    std::copy(&mps_comm[offset_i], &mps_comm[end_i], ci->Multipole.begin());
    iter = std::next(iter);
  }
}

void nbd::evaluateBaseAll(KerFunc_t ef, Base basis[], Cells& cells, int64_t levels, const Body* bodies, int64_t nbodies, double epi, int64_t mrank, int64_t sp_pts) {
  int64_t mpi_levels;
  commRank(NULL, NULL, &mpi_levels);

  for (int64_t i = levels; i >= 0; i--) {
    Cell* vlocal = findLocalAtLevelModify(&cells[0], i);
    evaluateLocal(ef, basis[i], vlocal, i, bodies, nbodies, epi, mrank, sp_pts);
    writeRemoteCoupling(basis[i], vlocal, i);
    
    if (i <= mpi_levels && i > 0) {
      int64_t mlen = vlocal->Multipole.size();
      int64_t msib;
      butterflyUpdateDims(mlen, &msib, i);
      Cell* vsib = vlocal->SIBL;
      int64_t len_m = vsib->Multipole.size();
      if (len_m != msib)
        vsib->Multipole.resize(msib);
      butterflyUpdateMultipoles(vlocal->Multipole.data(), mlen, vsib->Multipole.data(), msib, i);
    }
  }
}


void nbd::orth_base_all(Base* basis, int64_t levels) {
  for (int64_t i = levels; i > 0; i--) {
    Base* base_i = basis + i;
    int64_t len = (int64_t)1 << i;
    int64_t lbegin = 0;
    int64_t lend = len;
    selfLocalRange(&lbegin, &lend, i);
    contentLength(&len, i);

#pragma omp parallel for
    for (int64_t j = 0; j < len; j++) {
      int64_t dim = base_i->DIMS[j];
      int64_t dim_l = base_i->DIML[j];
      int64_t dim_c = dim - dim_l;

      Matrix& Uo_i = base_i->Uo[j];
      Matrix& Uc_i = base_i->Uc[j];
      Matrix& Ul_i = base_i->R[j];
      cMatrix(Uc_i, dim, dim_c);
      cMatrix(Ul_i, dim_l, dim_l);

      if (j >= lbegin && j < lend && dim > 0)
        qr_with_complements(Uo_i, Uc_i, Ul_i);
    }

    DistributeMatricesList(base_i->Uc.data(), i);
    DistributeMatricesList(base_i->Uo.data(), i);
    DistributeMatricesList(base_i->R.data(), i);

    int64_t nlevel = i - 1;
    int64_t nbegin = 0;
    int64_t nend = (int64_t)1 << nlevel;
    selfLocalRange(&nbegin, &nend, nlevel);
    int64_t nodes = nend - nbegin;

#pragma omp parallel for
    for (int64_t j = 0; j < nodes; j++) {
      int64_t lj = j + nbegin;
      int64_t gj = j;
      iGlobal(&gj, lj, nlevel);

      int64_t cj0 = (gj << 1);
      int64_t cj1 = (gj << 1) + 1;
      int64_t lj0 = cj0;
      int64_t lj1 = cj1;
      iLocal(&lj0, cj0, i);
      iLocal(&lj1, cj1, i);

      updateSubU(basis[nlevel].Uo[lj], base_i->R[lj0], base_i->R[lj1]);
    }
  }
}
