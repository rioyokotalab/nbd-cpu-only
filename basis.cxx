
#include "nbd.h"
#include "profile.h"

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdio>
#include "stdlib.h"
#include "math.h"
#include "string.h"

int64_t gen_close(int64_t clen, int64_t close[], int64_t ngbs, const int64_t ngbs_body[], const int64_t ngbs_len[]) {
  int64_t avail = std::accumulate(ngbs_len, ngbs_len + ngbs, 0);
  clen = std::min(avail, clen);
  std::iota(close, close + clen, 0);
  std::transform(close, close + clen, close, [avail, clen](int64_t& i)->int64_t { return (double)(i * avail) / clen; });
  int64_t* begin = close;
  for (int64_t i = 0; i < ngbs; i++) {
    int64_t len = std::min(std::distance(begin, close + clen), ngbs_len[i]);
    int64_t slen = std::accumulate(ngbs_len, ngbs_len + i, 0);
    int64_t bound = ngbs_len[i] + slen;
    int64_t body = ngbs_body[i] - slen;
    int64_t* next = std::find_if(begin, begin + len, [bound](int64_t& i)->bool { return i >= bound; });
    std::transform(begin, next, begin, [body](int64_t& i)->int64_t { return i + body; });
    begin = next;
  }
  return clen;
}

int64_t gen_far(int64_t flen, int64_t far[], int64_t ngbs, const int64_t ngbs_body[], const int64_t ngbs_len[], int64_t nbody) {
  int64_t near = std::accumulate(ngbs_len, ngbs_len + ngbs, 0);
  int64_t avail = nbody - near;
  flen = std::min(avail, flen);
  std::iota(far, far + flen, 0);
  std::transform(far, far + flen, far, [avail, flen](int64_t& i)->int64_t { return (double)(i * avail) / flen; });
  int64_t* begin = far;
  for (int64_t i = 0; i < ngbs; i++) {
    int64_t slen = std::accumulate(ngbs_len, ngbs_len + i, 0);
    int64_t bound = ngbs_body[i] - slen;
    int64_t* next = std::find_if(begin, far + flen, [bound](int64_t& i)->bool { return i >= bound; });
    std::transform(begin, next, begin, [slen](int64_t& i)->int64_t { return i + slen; });
    begin = next;
  }
  std::transform(begin, far + flen, begin, [near](int64_t& i)->int64_t { return i + near; });
  return flen;
}

int64_t dist_int_64(int64_t arr[], int64_t blen, const struct CellComm* comm) {
  int64_t plen = comm->Proc[0] == comm->worldRank ? comm->lenTargets : 0;
  const int64_t* row = comm->ProcTargets;
  int64_t lbegin = 0;
  int64_t lmax = 0;
#ifdef _PROF
  double stime = MPI_Wtime();
#endif
  for (int64_t i = 0; i < plen; i++) {
    int64_t p = row[i];
    int64_t len = (comm->ProcBoxesEnd[p] - comm->ProcBoxes[p]) * blen;
    MPI_Bcast(&arr[lbegin], len, MPI_INT64_T, comm->ProcRootI[p], comm->Comm_box[p]);
    lbegin = lbegin + len;
  }

  int64_t xlen = 0;
  content_length(&xlen, comm);
  int64_t alen = xlen * blen;
  if (comm->Proc[1] - comm->Proc[0] > 1)
    MPI_Bcast(arr, alen, MPI_INT64_T, 0, comm->Comm_share);

  for (int64_t i = 0; i < alen; i++)
    lmax = lmax < arr[i] ? arr[i] : lmax;
  MPI_Allreduce(MPI_IN_PLACE, &lmax, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);

#ifdef _PROF
  double etime = MPI_Wtime() - stime;
  recordCommTime(etime);
#endif
  return lmax;
}

void dist_double(double arr[], int64_t blen, const struct CellComm* comm) {
  int64_t plen = comm->Proc[0] == comm->worldRank ? comm->lenTargets : 0;
  const int64_t* row = comm->ProcTargets;
  int64_t lbegin = 0;
#ifdef _PROF
  double stime = MPI_Wtime();
#endif
  for (int64_t i = 0; i < plen; i++) {
    int64_t p = row[i];
    int64_t len = (comm->ProcBoxesEnd[p] - comm->ProcBoxes[p]) * blen;
    MPI_Bcast(&arr[lbegin], len, MPI_DOUBLE, comm->ProcRootI[p], comm->Comm_box[p]);
    lbegin = lbegin + len;
  }

  int64_t xlen = 0;
  content_length(&xlen, comm);
  int64_t alen = xlen * blen;
  if (comm->Proc[1] - comm->Proc[0] > 1)
    MPI_Bcast(arr, alen, MPI_DOUBLE, 0, comm->Comm_share);
#ifdef _PROF
  double etime = MPI_Wtime() - stime;
  recordCommTime(etime);
#endif
}

void buildBasis(void(*ef)(double*), struct Base basis[], int64_t ncells, struct Cell* cells, const struct CSC* rel_near, int64_t levels, 
const struct CellComm* comm, const struct Body* bodies, int64_t nbodies, double epi, int64_t mrank, int64_t sp_pts) {

  for (int64_t l = levels; l >= 0; l--) {
    int64_t xlen = 0;
    content_length(&xlen, &comm[l]);
    basis[l].Ulen = xlen;
    int64_t* arr_i = (int64_t*)malloc(sizeof(int64_t) * xlen * 3);
    basis[l].Lchild = arr_i;
    basis[l].Dims = &arr_i[xlen];
    basis[l].DimsLr = &arr_i[xlen * 2];

    struct Matrix* arr_m = (struct Matrix*)calloc(xlen * 3, sizeof(struct Matrix));
    basis[l].Uo = arr_m;
    basis[l].Uc = &arr_m[xlen];
    basis[l].R = &arr_m[xlen * 2];

    int64_t jbegin = 0, jend = ncells;
    int64_t ibegin = 0, iend = xlen;
    get_level(&jbegin, &jend, cells, l, -1);
    self_local_range(&ibegin, &iend, &comm[l]);
    int64_t nodes = iend - ibegin;
    std::vector<int64_t> cell_ids(xlen);

    for (int64_t i = 0; i < xlen; i++) {
      int64_t gi = i;
      i_global(&gi, &comm[l]);
      int64_t ci = jbegin + gi;
      int64_t lc = cells[ci].Child;
      int64_t ske = cells[ci].Body[1] - cells[ci].Body[0];
      cell_ids[i] = ci;

      if (lc >= 0) {
        lc = lc - jend;
        i_local(&lc, &comm[l + 1]);
        ske = basis[l + 1].DimsLr[lc] + basis[l + 1].DimsLr[lc + 1];
      }
      arr_i[i] = lc;
      arr_i[i + xlen] = ske;
    }

    basis[l].dimN = dist_int_64(basis[l].Dims, 1, &comm[l]);
    int64_t strideU = basis[l].dimN * basis[l].dimN;
    std::vector<int64_t> skeletons(basis[l].dimN * nodes);

    for (int64_t i = 0; i < nodes; i++) {
      int64_t lc = arr_i[i + ibegin];
      int64_t ske = arr_i[i + ibegin + xlen];

      if (lc >= 0) {
        const int64_t* m1 = basis[l + 1].Multipoles + basis[l + 1].dimS * lc;
        const int64_t* m2 = basis[l + 1].Multipoles + basis[l + 1].dimS * (lc + 1);
        int64_t len1 = basis[l + 1].DimsLr[lc];
        int64_t len2 = basis[l + 1].DimsLr[lc + 1];
        std::copy(m1, m1 + len1, skeletons.begin() + i * basis[l].dimN);
        std::copy(m2, m2 + len2, skeletons.begin() + i * basis[l].dimN + len1);
      }
      else {
        int64_t ci = cell_ids[i + ibegin];
        std::iota(skeletons.begin() + i * basis[l].dimN, skeletons.begin() + i * basis[l].dimN + ske, cells[ci].Body[0]);
      }
    }

    std::vector<double> matrix_data(strideU * nodes * 2);

#pragma omp parallel for
    for (int64_t i = 0; i < nodes; i++) {
      int64_t ske_len = basis[l].Dims[i + ibegin];
      int64_t ci = cell_ids[i + ibegin];
      int64_t nbegin = rel_near->ColIndex[ci];
      int64_t nlen = rel_near->ColIndex[ci + 1] - nbegin;
      const int64_t* ngbs = &rel_near->RowIndex[nbegin];
      std::vector<int64_t> close(sp_pts), remote(sp_pts), body, lens;

      for (int64_t j = 0; j < nlen; j++) {
        body.emplace_back(cells[ngbs[j]].Body[0]);
        lens.emplace_back(cells[ngbs[j]].Body[1] - cells[ngbs[j]].Body[0]);
      }
      int64_t len_f = gen_far(sp_pts, remote.data(), body.size(), body.data(), lens.data(), nbodies);
      int64_t cpos = std::distance(ngbs, std::find(ngbs, ngbs + nlen, ci));
      body.erase(body.begin() + cpos);
      lens.erase(lens.begin() + cpos);
      int64_t len_c = gen_close(sp_pts, close.data(), body.size(), body.data(), lens.data());
      int64_t* ske_i = &skeletons[i * basis[l].dimN];
      
      std::vector<double> Smat(ske_len * (ske_len + sp_pts)), Svec(ske_len * 2);
      struct Matrix S_dn = (struct Matrix){ Smat.data(), ske_len, ske_len, ske_len };
      double nrm_dn = 0.;
      double nrm_lr = 0.;
      struct Matrix S_dn_work = (struct Matrix){ &Smat[ske_len * ske_len], ske_len, len_c, ske_len };
      gen_matrix(ef, ske_len, len_c, bodies, bodies, S_dn_work.A, S_dn_work.LDA, ske_i, close.data());
      mmult('N', 'T', &S_dn_work, &S_dn_work, &S_dn, 1., 0.);
      nrm2_A(&S_dn, &nrm_dn);

      struct Matrix S_lr = (struct Matrix){ &Smat[ske_len * ske_len], ske_len, len_f, ske_len };
      gen_matrix(ef, ske_len, len_f, bodies, bodies, S_lr.A, S_lr.LDA, ske_i, remote.data());
      nrm2_A(&S_lr, &nrm_lr);
      double scale = (nrm_dn == 0. || nrm_lr == 0.) ? 1. : nrm_lr / nrm_dn;
      scal_A(&S_dn, scale);

      int64_t rank = mrank > 0 ? (mrank < ske_len ? mrank : ske_len) : ske_len;
      struct Matrix S = (struct Matrix){ Smat.data(), ske_len, ske_len + len_f, ske_len };
      svd_U(&S, Svec.data());

      if (epi > 0.) {
        int64_t r = 0;
        double sepi = Svec[0] * epi;
        while(r < rank && Svec[r] > sepi)
          r += 1;
        rank = r;
      }
      basis[l].DimsLr[i + ibegin] = rank;
      
      std::vector<int32_t> pa(ske_len);
      struct Matrix Qo = (struct Matrix){ &matrix_data[strideU * i * 2], ske_len, rank, ske_len };
      struct Matrix work = (struct Matrix){ Smat.data(), ske_len, rank, ske_len };
      id_row(&Qo, &work, pa.data());
      int64_t lc = basis[l].Lchild[i + ibegin];
      if (lc >= 0)
        upper_tri_reflec_mult('L', 2, &(basis[l + 1].R)[lc], &Qo);

      for (int64_t j = 0; j < rank; j++)
        std::iter_swap(ske_i + (pa[j] - 1), &ske_i[j]);

      if (rank > 0) {
        struct Matrix Q = (struct Matrix){ &matrix_data[strideU * i * 2], ske_len, ske_len, ske_len };
        struct Matrix R = (struct Matrix){ &matrix_data[strideU * i * 2 + ske_len * ske_len], rank, rank, rank };
        qr_full(&Q, &R);
      }
    }

    basis[l].dimS = dist_int_64(basis[l].DimsLr, 1, &comm[l]);
    int64_t strideR = basis[l].dimS * basis[l].dimS;

    basis[l].Multipoles = (int64_t*)malloc(sizeof(int64_t) * basis[l].dimS * xlen);
    for (int64_t i = 0; i < nodes; i++) {
      int64_t offset = basis[l].dimS * (i + ibegin);
      int64_t n = basis[l].DimsLr[i + ibegin];
      if (n > 0)
        memcpy(&basis[l].Multipoles[offset], &skeletons[i * basis[l].dimN], sizeof(int64_t) * n);
    }
    if (basis[l].dimS)
      dist_int_64(basis[l].Multipoles, basis[l].dimS, &comm[l]);

    int64_t strideD = strideU + strideR;
    double* data = (double*)malloc(sizeof(double) * strideD * xlen);

#pragma omp parallel for
    for (int64_t i = 0; i < xlen; i++) {
      int64_t m = basis[l].Dims[i];
      int64_t n = basis[l].DimsLr[i];
      basis[l].Uo[i] = (struct Matrix){ &data[strideD * i], m, n, basis[l].dimN };
      basis[l].Uc[i] = (struct Matrix){ &data[strideD * i + basis[l].dimN * n], m, m - n, basis[l].dimN };
      basis[l].R[i] = (struct Matrix){ &data[strideD * i + basis[l].dimN * m], n, n, basis[l].dimS };
      if (i >= ibegin && i < iend) {
        double* mat = &matrix_data[strideU * (i - ibegin) * 2];
        struct Matrix Qo = (struct Matrix){ mat, m, n, m };
        struct Matrix Qc = (struct Matrix){ &mat[m * n], m, m - n, m };
        struct Matrix R = (struct Matrix){ &mat[m * m], n, n, n };
        mat_cpy(m, n, &Qo, &basis[l].Uo[i], 0, 0, 0, 0);
        mat_cpy(m, m - n, &Qc, &basis[l].Uc[i], 0, 0, 0, 0);
        mat_cpy(n, n, &R, &basis[l].R[i], 0, 0, 0, 0);
      }
    }

    if (strideU)
      dist_double(data, strideD, &comm[l]);
  }
}

void basis_free(struct Base* basis) {
  double* data = basis->Uo[0].A;
  if (data)
    free(data);
  if (basis->Multipoles)
    free(basis->Multipoles);
  free(basis->Lchild);
  free(basis->Uo);
}

void evalS(void(*ef)(double*), struct Matrix* S, const struct Base* basis, const struct Body* bodies, const struct CSC* rels, const struct CellComm* comm) {
  int64_t ibegin = 0, iend = 0;
  self_local_range(&ibegin, &iend, comm);
  int64_t* multipoles = basis->Multipoles;

#pragma omp parallel for
  for (int64_t x = 0; x < rels->N; x++) {
    int64_t n = basis->DimsLr[x + ibegin];
    int64_t off_x = basis->dimS * (x + ibegin);

    for (int64_t yx = rels->ColIndex[x]; yx < rels->ColIndex[x + 1]; yx++) {
      int64_t y = rels->RowIndex[yx];
      int64_t m = basis->DimsLr[y];
      int64_t off_y = basis->dimS * y;
      gen_matrix(ef, m, n, bodies, bodies, S[yx].A, S[yx].LDA, &multipoles[off_y], &multipoles[off_x]);
      upper_tri_reflec_mult('L', 1, &basis->R[y], &S[yx]);
      upper_tri_reflec_mult('R', 1, &basis->R[x + ibegin], &S[yx]);
    }
  }
}
