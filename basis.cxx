
#include "basis.h"
#include "dist.h"

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

void remoteBodies(int64_t* remote, int64_t* close, int64_t size[], const Cell* cell, int64_t nbodies) {
  int64_t len = cell->listNear.size();
  std::vector<int64_t> offsets(len);
  std::vector<int64_t> lens(len);

  int64_t sum_len = 0;
  int64_t cpos = -1;
  for (int64_t i = 0; i < len; i++) {
    const Cell* c = cell->listNear[i];
    offsets[i] = c->BODY[0];
    lens[i] = c->BODY[1] - c->BODY[0];
    sum_len = sum_len + lens[i];
    if (c == cell)
      cpos = i;
  }

  int64_t avail = nbodies - sum_len;
  int64_t msize = size[0];
  msize = msize > avail ? avail : msize;

  int64_t box_i = 0;
  int64_t s_lens = 0;
  for (int64_t i = 0; i < msize; i++) {
    int64_t loc = (int64_t)((double)(avail * i) / msize);
    while (box_i < len && loc + s_lens >= offsets[box_i]) {
      s_lens = s_lens + lens[box_i];
      box_i = box_i + 1;
    }
    remote[i] = loc + s_lens;
  }
  size[0] = msize;

  avail = sum_len - lens[cpos];
  msize = size[1];
  msize = msize > avail ? avail : msize;

  box_i = (int64_t)(cpos == 0);
  s_lens = 0;
  for (int64_t i = 0; i < msize; i++) {
    int64_t loc = (int64_t)((double)(avail * i) / msize);
    while (loc - s_lens >= lens[box_i]) {
      s_lens = s_lens + lens[box_i];
      box_i = box_i + 1;
      if (box_i == cpos)
        box_i = box_i + 1;
    }
    close[i] = loc + offsets[box_i] - s_lens;
  }
  size[1] = msize;
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

#pragma omp parallel for
  for (int64_t i = 0; i < nodes; i++) {
    Cell* ci = &leaves[i + gbegin];
    int64_t box_i = i + ibegin;

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
      int64_t nbegin = ci->BODY[0];
      ni = ci->BODY[1] - nbegin;
      cellm.resize(ni);
      for (int64_t i = 0; i < ni; i++)
        cellm[i] = nbegin + i;
    }
    
    if (ni > 0) {
      std::vector<int64_t> remote(sp_pts);
      std::vector<int64_t> close(sp_pts);
      int64_t n[2] = { sp_pts, sp_pts };
      remoteBodies(remote.data(), close.data(), n, ci, nbodies);

      int64_t rank = mrank;
      evaluateBasis(ef, epi, &rank, basis.Uo[box_i], ni, n[0], n[1], cellm.data(), remote.data(), close.data(), bodies);

      int64_t len_m = ci->Multipole.size();
      if (len_m != rank)
        ci->Multipole.resize(rank);
      std::copy(&cellm[0], &cellm[rank], &ci->Multipole[0]);
    }

    int64_t mi = ci->Multipole.size();
    basis.DIMS[box_i] = ni;
    basis.DIML[box_i] = mi;
  }

  DistributeDims(&basis.DIMS[0], level);
  DistributeDims(&basis.DIML[0], level);

  int64_t count = 0;
  std::vector<int64_t> offsets(xlen);
  for (int64_t i = 0; i < xlen; i++) {
    int64_t m = basis.DIMS[i];
    int64_t n = basis.DIML[i];
    offsets[i] = count;
    count = count + n;

    int64_t msize = m * n;
    if (msize > 0 && (i < ibegin || i >= iend))
      matrixCreate(&basis.Uo[i], m, n);
  }
  DistributeMatricesList(basis.Uo.data(), level);

  std::vector<int64_t> mps_comm(count);
  for (int64_t i = 0; i < nodes; i++) {
    const Cell* ci = &leaves[i + gbegin];
    int64_t box_i = i + ibegin;

    int64_t offset_i = offsets[box_i];
    std::copy(ci->Multipole.begin(), ci->Multipole.end(), &mps_comm[offset_i]);
  }
  DistributeMultipoles(mps_comm.data(), basis.DIML.data(), level);

  for (int64_t i = 0; i < xlen; i++) {
    int64_t gi = i;
    iGlobal(&gi, i, level);
    Cell* ci = &leaves[gi];

    int64_t offset_i = offsets[i];
    int64_t end_i = offset_i + basis.DIML[i];
    int64_t len_m = ci->Multipole.size();
    if (len_m != basis.DIML[i])
      ci->Multipole.resize(basis.DIML[i]);
    std::copy(&mps_comm[offset_i], &mps_comm[end_i], ci->Multipole.begin());
  }
}

void evaluateBaseAll(KerFunc_t ef, Base basis[], Cell* cells, int64_t levels, const Body* bodies, int64_t nbodies, double epi, int64_t mrank, int64_t sp_pts) {
  for (int64_t i = levels; i >= 0; i--)
    evaluateLocal(ef, basis[i], cells, i, bodies, nbodies, epi, mrank, sp_pts);
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
