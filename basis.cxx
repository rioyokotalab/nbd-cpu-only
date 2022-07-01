
#include "basis.h"
#include "dist.h"

#include "stdio.h"
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
        int64_t n[3] = { sp_pts, sp_pts, nbodies };
        std::vector<int64_t> remote(sp_pts + sp_pts);

        int64_t ii = ci - cells;
        int64_t lbegin = cellsNear->COL_INDEX[ii];
        int64_t llen = cellsNear->COL_INDEX[ii + 1] - lbegin;
        remoteBodies(remote.data(), n, llen, &cellsNear->ROW_INDEX[lbegin], cells, ii);

        int64_t rank = mrank;
        evaluateBasis(ef, epi, &rank, &(base_i->Uo)[box_i], ni, n, cellm, remote.data(), bodies);

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
      int64_t n = base_i->DIML[i + ibegin];
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

