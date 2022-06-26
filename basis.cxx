
#include "basis.h"
#include "dist.h"

#include "stdlib.h"

void allocBasis(Base* basis, int64_t levels) {
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nodes = (int64_t)1 << i;
    contentLength(&nodes, i);
    basis[i].Ulen = nodes;
    basis[i].DIMS.resize(nodes);
    basis[i].DIML.resize(nodes);
    basis[i].Multipoles.clear();
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
    basis[i].Multipoles.clear();
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

void evaluateBasis(KerFunc_t ef, double epi, int64_t* rank, Matrix* Base, int64_t m, int64_t n1, int64_t n2, 
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

    matrixCreate(Base, m, iters);
    cpyMatToMat(m, iters, &work_u, Base, 0, 0, 0, 0);
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

void remoteBodies(int64_t* remote, int64_t* close, int64_t size[], int64_t cpos, int64_t llen, const int64_t offsets[], const int64_t lens[], const int64_t avail[]) {
  int64_t rm_len = avail[0];
  int64_t rmsize = size[0];
  rmsize = rmsize > rm_len ? rm_len : rmsize;

  int64_t box_i = 0;
  int64_t s_lens = 0;
  for (int64_t i = 0; i < rmsize; i++) {
    int64_t loc = (int64_t)((double)(rm_len * i) / rmsize);
    while (box_i < llen && loc + s_lens >= offsets[box_i]) {
      s_lens = s_lens + lens[box_i];
      box_i = box_i + 1;
    }
    remote[i] = loc + s_lens;
  }
  size[0] = rmsize;

  int64_t cl_len = avail[1];
  int64_t clsize = size[1];
  clsize = clsize > cl_len ? cl_len : clsize;

  box_i = (int64_t)(cpos == 0);
  s_lens = 0;
  for (int64_t i = 0; i < clsize; i++) {
    int64_t loc = (int64_t)((double)(cl_len * i) / clsize);
    while (loc - s_lens >= lens[box_i]) {
      s_lens = s_lens + lens[box_i];
      box_i = box_i + 1;
      box_i = box_i + (int)(box_i == cpos);
    }
    close[i] = loc + offsets[box_i] - s_lens;
  }
  size[1] = clsize;
}

void evaluateBaseAll(KerFunc_t ef, Base basis[], Cell* cells, const CSC* cellsNear, int64_t levels, const Body* bodies, int64_t nbodies, double epi, int64_t mrank, int64_t sp_pts) {
  for (int64_t l = levels; l >= 0; l--) {
    Base* base_i = basis + l;
    int64_t xlen = (int64_t)1 << l;
    int64_t ibegin = 0;
    int64_t iend = xlen;
    int64_t gbegin = ibegin;
    selfLocalRange(&ibegin, &iend, l);
    iGlobal(&gbegin, ibegin, l);
    contentLength(&xlen, l);
    int64_t nodes = iend - ibegin;

    int64_t len = (int64_t)1 << l;
    Cell* leaves = &cells[len - 1];

  #pragma omp parallel for
    for (int64_t i = 0; i < nodes; i++) {
      Cell* ci = &leaves[i + gbegin];
      int64_t box_i = i + ibegin;
      int64_t cic = ci->CHILD;
      int64_t ni = 0;
      int64_t* cellm;

      if (cic >= 0) {
        int64_t len0 = cells[cic].lenMultipole;
        int64_t len1 = cells[cic + 1].lenMultipole;
        ni = len0 + len1;
        cellm = (int64_t*)malloc(sizeof(int64_t) * ni);

        std::copy(&cells[cic].Multipole[0], &cells[cic].Multipole[len0], &cellm[0]);
        std::copy(&cells[cic + 1].Multipole[0], &cells[cic + 1].Multipole[len1], &cellm[len0]);
      }
      else {
        int64_t nbegin = ci->BODY[0];
        ni = ci->BODY[1] - nbegin;
        cellm = (int64_t*)malloc(sizeof(int64_t) * ni);
        for (int64_t i = 0; i < ni; i++)
          cellm[i] = nbegin + i;
      }
      
      if (ni > 0) {
        int64_t ii = ci - cells;
        int64_t lbegin = cellsNear->COL_INDEX[ii];
        int64_t llen = cellsNear->COL_INDEX[ii + 1] - lbegin;

        std::vector<int64_t> offsets(llen);
        std::vector<int64_t> lens(llen);

        int64_t sum_len = 0;
        int64_t cpos = -1;
        for (int64_t j = 0; j < llen; j++) {
          int64_t jc = cellsNear->ROW_INDEX[lbegin + j];
          const Cell* c = &cells[jc];
          offsets[j] = c->BODY[0];
          lens[j] = c->BODY[1] - c->BODY[0];
          sum_len = sum_len + lens[j];
          if (jc == ii)
            cpos = j;
        }

        std::vector<int64_t> remote(sp_pts);
        std::vector<int64_t> close(sp_pts);
        int64_t n[2] = { sp_pts, sp_pts };
        int64_t nmax[2] = { nbodies - sum_len, sum_len - lens[cpos] };

        remoteBodies(remote.data(), close.data(), n, cpos, llen, &offsets[0], &lens[0], nmax);

        int64_t rank = mrank;
        evaluateBasis(ef, epi, &rank, &(base_i->Uo)[box_i], ni, n[0], n[1], cellm, remote.data(), close.data(), bodies);

        matrixCreate(&(base_i->Uc)[box_i], ni, ni - rank);
        matrixCreate(&(base_i->R)[box_i], rank, rank);

        if (rank < ni)
          cellm = (int64_t*)realloc(cellm, sizeof(int64_t) * rank);
        ci->Multipole = cellm;
        ci->lenMultipole = rank;
      }

      int64_t mi = ci->lenMultipole;
      base_i->DIMS[box_i] = ni;
      base_i->DIML[box_i] = mi;
    }

    DistributeDims(&(base_i->DIMS)[0], l);
    DistributeDims(&(base_i->DIML)[0], l);

    int64_t count = 0;
    std::vector<int64_t> offsets(xlen);
    for (int64_t i = 0; i < xlen; i++) {
      int64_t m = base_i->DIMS[i];
      int64_t n = base_i->DIML[i];
      offsets[i] = count;
      count = count + n;

      int64_t msize = m * n;
      if (msize > 0 && (i < ibegin || i >= iend)) {
        matrixCreate(&(base_i->Uo)[i], m, n);
        matrixCreate(&(base_i->Uc)[i], m, m - n);
        matrixCreate(&(base_i->R)[i], n, n);
      }
    }

    base_i->Multipoles.resize(count);
    int64_t* mps_comm = &(base_i->Multipoles)[0];
    for (int64_t i = 0; i < nodes; i++) {
      Cell* ci = &leaves[i + gbegin];
      int64_t offset_i = offsets[i + ibegin];
      int64_t n = base_i->DIML[i];
      std::copy(ci->Multipole, &(ci->Multipole)[n], &mps_comm[offset_i]);
      free(ci->Multipole);
    }
    DistributeMultipoles(mps_comm, &(base_i->DIML)[0], l);

    for (int64_t i = 0; i < xlen; i++) {
      int64_t gi = i;
      iGlobal(&gi, i, l);
      Cell* ci = &leaves[gi];

      int64_t mi = base_i->DIML[i];
      int64_t offset_i = offsets[i];
      ci->Multipole = &mps_comm[offset_i];
      ci->lenMultipole = mi;
    }
  }

  for (int64_t l = levels; l > 0; l--) {
    Base* base_i = basis + l;
    int64_t lbegin = 0;
    int64_t lend = (int64_t)1 << l;
    selfLocalRange(&lbegin, &lend, l);
    int64_t len = lend - lbegin;

#pragma omp parallel for
    for (int64_t i = 0; i < len; i++) {
      int64_t li = i + lbegin;
      if (base_i->DIMS[li] > 0)
        qr_with_complements(&(base_i->Uo)[li], &(base_i->Uc)[li], &(base_i->R)[li]);
    }

    DistributeMatricesList(&(base_i->Uc)[0], l);
    DistributeMatricesList(&(base_i->Uo)[0], l);
    DistributeMatricesList(&(base_i->R)[0], l);

    int64_t nlevel = l - 1;
    int64_t nbegin = 0;
    int64_t nend = (int64_t)1 << nlevel;
    selfLocalRange(&nbegin, &nend, nlevel);
    int64_t nodes = nend - nbegin;

#pragma omp parallel for
    for (int64_t i = 0; i < nodes; i++) {
      int64_t li = i + nbegin;
      int64_t gi = i;
      iGlobal(&gi, li, nlevel);

      int64_t ci0 = (gi << 1);
      int64_t ci1 = (gi << 1) + 1;
      int64_t li0 = ci0;
      int64_t li1 = ci1;
      iLocal(&li0, ci0, l);
      iLocal(&li1, ci1, l);

      updateSubU(&(basis[nlevel].Uo)[li], &(base_i->R)[li0], &(base_i->R)[li1]);
    }
  }
}

