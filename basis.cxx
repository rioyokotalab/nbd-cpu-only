
#include "basis.h"
#include "dist.h"

#include <numeric>
#include <unordered_set>
#include <iterator>

void allocBasis(Base* basis, int64_t levels) {
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nodes = (int64_t)1 << i;
    contentLength(&nodes, i);
    basis[i].Ulen = nodes;
    basis[i].DIMS.resize(nodes);
    basis[i].DIML.resize(nodes);
    basis[i].Uo.resize(nodes);
    basis[i].Uc.resize(nodes);
    basis[i].R.resize(nodes);
  }
}

void deallocBasis(Base* basis, int64_t levels) {
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nodes = basis[i].Ulen;
    for (int64_t n = 0; n < nodes; n++) {
      matrixDestroy(&basis[i].Uo[n]);
      matrixDestroy(&basis[i].Uc[n]);
      matrixDestroy(&basis[i].R[n]);
    }

    basis[i].Ulen = 0;
    basis[i].DIMS.clear();
    basis[i].DIML.clear();
    basis[i].Uo.clear();
    basis[i].Uc.clear();
    basis[i].R.clear();
  }
}

void basis_mem(int64_t* bytes, const Base* basis, int64_t levels) {
  int64_t count = 0;
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nodes = basis[i].Ulen;
    int64_t bytes_o, bytes_c, bytes_r;
    matrix_mem(&bytes_o, &basis[i].Uo[0], nodes);
    matrix_mem(&bytes_c, &basis[i].Uc[0], nodes);
    matrix_mem(&bytes_r, &basis[i].R[0], nodes);

    count = count + bytes_o + bytes_c + bytes_r;
  }
  *bytes = count;
}

void evaluateBasis(KerFunc_t ef, double epi, int64_t* rank, Matrix& Base, int64_t m, int64_t n1, int64_t n2, 
  int64_t cellm[], const int64_t remote[], const int64_t close[], const Body* bodies) {
  Matrix work_a, work_b, work_c, work_s;
  int64_t len_s = n1 + (n2 > 0 ? m : 0);
  if (len_s > 0)
    matrixCreate(&work_s, m, len_s);

  if (n1 > 0) {
    matrixCreate(&work_a, m, n1);
    gen_matrix(ef, m, n1, bodies, bodies, work_a.A, cellm, remote);
    cpyMatToMat(m, n1, &work_a, &work_s, 0, 0, 0, 0);
  }

  if (n2 > 0) {
    matrixCreate(&work_b, m, n2);
    matrixCreate(&work_c, m, m);
    gen_matrix(ef, m, n2, bodies, bodies, work_b.A, cellm, close);
    mmult('N', 'T', &work_b, &work_b, &work_c, 1., 0.);
    if (n1 > 0)
      normalizeA(&work_c, &work_a);
    cpyMatToMat(m, m, &work_c, &work_s, 0, 0, 0, n1);
    matrixDestroy(&work_b);
    matrixDestroy(&work_c);
  }

  if (n1 > 0)
    matrixDestroy(&work_a);

  if (len_s > 0) {
    int64_t mrank = *rank;
    int64_t n = mrank > 0 ? std::min(mrank, m) : m;
    Matrix work_u;
    std::vector<int64_t> pa(n);
    matrixCreate(&work_u, m, n);

    int64_t iters = n;
    lraID(epi, &work_s, &work_u, pa.data(), &iters);

    matrixCreate(&Base, m, iters);
    cpyMatToMat(m, iters, &work_u, &Base, 0, 0, 0, 0);
    matrixDestroy(&work_s);
    matrixDestroy(&work_u);

    for (int64_t i = 0; i < iters; i++) {
      int64_t piv_i = pa[i] - 1;
      if (piv_i != i) {
        int64_t row_piv = cellm[piv_i];
        cellm[piv_i] = cellm[i];
        cellm[i] = row_piv;
      }
    }
    *rank = iters;
  }
}

void evaluateLocal(KerFunc_t ef, Base& basis, Cell* cell, int64_t level, const Body* bodies, int64_t nbodies, double epi, int64_t mrank, int64_t sp_pts) {
  int64_t xlen = basis.DIMS.size();
  int64_t ibegin = 0;
  int64_t iend = xlen;
  int64_t gbegin = ibegin;
  selfLocalRange(&ibegin, &iend, level);
  iGlobal(&gbegin, ibegin, level);
  int64_t nodes = iend - ibegin;

  int64_t len = (int64_t)1 << level;
  Cell* leaves = &cell[len - 1];

  std::vector<int64_t>& dims = basis.DIMS;
  std::vector<int64_t>& diml = basis.DIML;

#pragma omp parallel for
  for (int64_t i = 0; i < nodes; i++) {
    Cell* ci = &leaves[i + gbegin];
    int64_t ii = ci->ZID;
    int64_t box_i = ii;
    iLocal(&box_i, ii, level);

    int64_t ni = 0;
    std::vector<int64_t> cellm;

    if (ci->NCHILD > 0) {
      for (int64_t j = 0; j < ci->NCHILD; j++)
        ni += (ci->CHILD[j]).Multipole.size();
      cellm.resize(ni);

      int64_t count = 0;
      for (int64_t j = 0; j < ci->NCHILD; j++) {
        int64_t len = (ci->CHILD[j]).Multipole.size();
        std::copy(&(ci->CHILD[j]).Multipole[0], &(ci->CHILD[j]).Multipole[len], &cellm[count]);
        count += len;
      }
    }
    else {
      ni = ci->BODY[1] - ci->BODY[0];
      cellm.resize(ni);
      std::iota(&cellm[0], &cellm[ni], ci->BODY[0]);
    }
    
    if (ni > 0) {
      std::vector<int64_t> remote(sp_pts);
      std::vector<int64_t> close(sp_pts);
      int64_t n1 = remoteBodies(remote.data(), sp_pts, *ci, nbodies);
      int64_t n2 = closeBodies(close.data(), sp_pts, *ci);

      int64_t rank = mrank;
      evaluateBasis(ef, epi, &rank, basis.Uo[box_i], ni, n1, n2, cellm.data(), remote.data(), close.data(), bodies);

      int64_t len_m = ci->Multipole.size();
      if (len_m != rank)
        ci->Multipole.resize(rank);
      std::copy(&cellm[0], &cellm[rank], &ci->Multipole[0]);
    }

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
      matrixCreate(&basis.Uo[i], m, n);
  }
  DistributeMatricesList(basis.Uo.data(), level);
}

void writeRemoteCoupling(const Base& basis, Cell* cell, int64_t level) {
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

  int64_t len = (int64_t)1 << level;
  Cell* leaves = &cell[len - 1];
  std::unordered_set<Cell*> neighbors;

  std::vector<int64_t> mps_comm(count);
  for (int64_t i = 0; i < len; i++) {
    const Cell* ci = &leaves[i];
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

void evaluateBaseAll(KerFunc_t ef, Base basis[], Cell* cells, int64_t levels, const Body* bodies, int64_t nbodies, double epi, int64_t mrank, int64_t sp_pts) {
  for (int64_t i = levels; i >= 0; i--) {
    evaluateLocal(ef, basis[i], cells, i, bodies, nbodies, epi, mrank, sp_pts);
    writeRemoteCoupling(basis[i], cells, i);
    
    int comm_needed;
    butterflyComm(&comm_needed, i);
    if (comm_needed) {
      int64_t ibegin = 0;
      int64_t iend = (int64_t)1 << i;
      int64_t gbegin = ibegin;
      selfLocalRange(&ibegin, &iend, i);
      iGlobal(&gbegin, ibegin, i);
      Cell* mlocal = &cells[((int64_t)1 << i) + gbegin - 1];

      int64_t mlen = mlocal->Multipole.size();
      int64_t msib;
      butterflyUpdateDims(mlen, &msib, i);
      Cell* vsib = mlocal->SIBL;
      int64_t len_m = vsib->Multipole.size();
      if (len_m != msib)
        vsib->Multipole.resize(msib);
      butterflyUpdateMultipoles(mlocal->Multipole.data(), mlen, vsib->Multipole.data(), msib, i);
    }
  }
}


void orth_base_all(Base* basis, int64_t levels) {
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
      matrixCreate(&Uc_i, dim, dim_c);
      matrixCreate(&Ul_i, dim_l, dim_l);

      if (j >= lbegin && j < lend && dim > 0)
        qr_with_complements(&Uo_i, &Uc_i, &Ul_i);
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

      updateSubU(&basis[nlevel].Uo[lj], &base_i->R[lj0], &base_i->R[lj1]);
    }
  }
}
