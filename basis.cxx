
#include "nbd.hxx"
#include "profile.hxx"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include "mkl.h"

struct SampleBodies 
{ int64_t LTlen, *FarLens, *FarAvails, **FarBodies, *CloseLens, *CloseAvails, **CloseBodies, *SkeLens, **Skeletons; };

void buildSampleBodies(struct SampleBodies* sample, int64_t sp_max_far, int64_t sp_max_near, int64_t nbodies, int64_t ncells, const struct Cell* cells, 
const struct CSC* rels, const int64_t* lt_child, const struct Base* basis_lo, int64_t level) {
  int __mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &__mpi_rank);
  int64_t mpi_rank = __mpi_rank;
  const int64_t LEN_CHILD = 2;
  
  int64_t jbegin = 0, jend = ncells;
  get_level(&jbegin, &jend, cells, level, -1);
  int64_t ibegin = jbegin, iend = jend;
  get_level(&ibegin, &iend, cells, level, mpi_rank);
  int64_t nodes = iend - ibegin;
  int64_t* arr_ctrl = (int64_t*)malloc(sizeof(int64_t) * nodes * 5);
  int64_t** arr_list = (int64_t**)malloc(sizeof(int64_t*) * nodes * 3);
  sample->LTlen = nodes;
  sample->FarLens = arr_ctrl;
  sample->CloseLens = &arr_ctrl[nodes];
  sample->SkeLens = &arr_ctrl[nodes * 2];
  sample->FarAvails = &arr_ctrl[nodes * 3];
  sample->CloseAvails = &arr_ctrl[nodes * 4];
  sample->FarBodies = arr_list;
  sample->CloseBodies = &arr_list[nodes];
  sample->Skeletons = &arr_list[nodes * 2];

  int64_t count_f = 0, count_c = 0, count_s = 0;
  for (int64_t i = 0; i < nodes; i++) {
    int64_t li = ibegin + i;
    int64_t nbegin = rels->ColIndex[i];
    int64_t nlen = rels->ColIndex[i + 1] - nbegin;
    const int64_t* ngbs = &rels->RowIndex[nbegin];

    int64_t far_avail = nbodies;
    int64_t close_avail = 0;
    for (int64_t j = 0; j < nlen; j++) {
      int64_t lj = ngbs[j];
      const struct Cell* cj = &cells[lj];
      int64_t len = cj->Body[1] - cj->Body[0];
      far_avail = far_avail - len;
      if (lj != li)
        close_avail = close_avail + len;
    }

    int64_t lc = lt_child[i];
    int64_t ske_len = 0;
    if (basis_lo != NULL && lc >= 0)
      for (int64_t j = 0; j < LEN_CHILD; j++)
        ske_len = ske_len + basis_lo->DimsLr[lc + j];
    else
      ske_len = cells[li].Body[1] - cells[li].Body[0];

    int64_t far_len = sp_max_far < far_avail ? sp_max_far : far_avail;
    int64_t close_len = sp_max_near < close_avail ? sp_max_near : close_avail;
    arr_ctrl[i] = far_len;
    arr_ctrl[i + nodes] = close_len;
    arr_ctrl[i + nodes * 2] = ske_len;
    arr_ctrl[i + nodes * 3] = far_avail;
    arr_ctrl[i + nodes * 4] = close_avail;
    count_f = count_f + far_len;
    count_c = count_c + close_len;
    count_s = count_s + ske_len;
  }

  int64_t* arr_bodies = NULL;
  if ((count_f + count_c + count_s) > 0)
    arr_bodies = (int64_t*)malloc(sizeof(int64_t) * (count_f + count_c + count_s));
  count_s = count_f + count_c;
  count_c = count_f;
  count_f = 0;
  for (int64_t i = 0; i < nodes; i++) {
    int64_t* remote = &arr_bodies[count_f];
    int64_t* close = &arr_bodies[count_c];
    int64_t* skeleton = &arr_bodies[count_s];
    int64_t far_len = arr_ctrl[i];
    int64_t close_len = arr_ctrl[i + nodes];
    int64_t ske_len = arr_ctrl[i + nodes * 2];
    arr_list[i] = remote;
    arr_list[i + nodes] = close;
    arr_list[i + nodes * 2] = skeleton;
    count_f = count_f + far_len;
    count_c = count_c + close_len;
    count_s = count_s + ske_len;
  }

#pragma omp parallel for
  for (int64_t i = 0; i < nodes; i++) {
    int64_t nbegin = rels->ColIndex[i];
    int64_t nlen = rels->ColIndex[i + 1] - nbegin;
    const int64_t* ngbs = &rels->RowIndex[nbegin];

    int64_t* remote = arr_list[i];
    int64_t* close = arr_list[i + nodes];
    int64_t* skeleton = arr_list[i + nodes * 2];
    int64_t far_len = arr_ctrl[i];
    int64_t close_len = arr_ctrl[i + nodes];
    int64_t ske_len = arr_ctrl[i + nodes * 2];
    int64_t far_avail = arr_ctrl[i + nodes * 3];
    int64_t close_avail = arr_ctrl[i + nodes * 4];

    int64_t box_i = 0;
    int64_t s_lens = 0;
    int64_t ic = ngbs[box_i];
    int64_t offset_i = cells[ic].Body[0];
    int64_t len_i = cells[ic].Body[1] - offset_i;

    int64_t li = i + ibegin - jbegin;
    int64_t cpos = 0;
    while (cpos < nlen && ngbs[cpos] != li)
      cpos = cpos + 1;

    for (int64_t j = 0; j < far_len; j++) {
      int64_t loc = (int64_t)((double)(far_avail * j) / far_len);
      while (box_i < nlen && loc + s_lens >= offset_i) {
        s_lens = s_lens + len_i;
        box_i = box_i + 1;
        ic = box_i < nlen ? ngbs[box_i] : ic;
        offset_i = cells[ic].Body[0];
        len_i = cells[ic].Body[1] - offset_i;
      }
      remote[j] = loc + s_lens;
    }

    box_i = (int64_t)(cpos == 0);
    s_lens = 0;
    ic = box_i < nlen ? ngbs[box_i] : ic;
    offset_i = cells[ic].Body[0];
    len_i = cells[ic].Body[1] - offset_i;

    for (int64_t j = 0; j < close_len; j++) {
      int64_t loc = (int64_t)((double)(close_avail * j) / close_len);
      while (loc - s_lens >= len_i) {
        s_lens = s_lens + len_i;
        box_i = box_i + 1;
        box_i = box_i + (int64_t)(box_i == cpos);
        ic = ngbs[box_i];
        offset_i = cells[ic].Body[0];
        len_i = cells[ic].Body[1] - offset_i;
      }
      close[j] = loc + offset_i - s_lens;
    }

    int64_t lc = lt_child[i];
    int64_t sbegin = cells[i + ibegin].Body[0];
    if (basis_lo != NULL && lc >= 0)
      memcpy(skeleton, basis_lo->Multipoles.data() + basis_lo->Offsets[lc], sizeof(int64_t) * ske_len);
    else
      for (int64_t j = 0; j < ske_len; j++)
        skeleton[j] = j + sbegin;
  }
}

void sampleBodies_free(struct SampleBodies* sample) {
  int64_t* data = sample->FarBodies[0];
  if (data)
    free(data);
  free(sample->FarLens);
  free(sample->FarBodies);
}

void nrm2_A(struct Matrix* A, double* nrm) {
  int64_t len_A = A->M * A->N;
  double nrm_A = cblas_dnrm2(len_A, A->A, 1);
  *nrm = nrm_A;
}

void scal_A(struct Matrix* A, double alpha) {
  int64_t len_A = A->M * A->N;
  cblas_dscal(len_A, alpha, A->A, 1);
}

void svd_U(struct Matrix* A, struct Matrix* U, double* S) {
  int64_t rank_a = A->M < A->N ? A->M : A->N;
  LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'N', A->M, A->N, A->A, A->M, S, U->A, A->M, NULL, A->N, &S[rank_a]);
}

void id_row(struct Matrix* U, int32_t arows[], double* work) {
  struct Matrix A = (struct Matrix){ work, U->M, U->N, U->M };
  cblas_dcopy(A.M * A.N, U->A, 1, A.A, 1);
  LAPACKE_dgetrf(LAPACK_COL_MAJOR, A.M, A.N, A.A, A.M, arows);
  cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, A.M, A.N, 1., A.A, A.M, U->A, A.M);
  cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit, A.M, A.N, 1., A.A, A.M, U->A, A.M);
}

void memcpy2d(void* dst, const void* src, int64_t rows, int64_t cols, int64_t ld_dst, int64_t ld_src, size_t elm_size) {
  for (int64_t i = 0; i < cols; i++) {
    unsigned char* _dst = (unsigned char*)dst + i * ld_dst * elm_size;
    const unsigned char* _src = (const unsigned char*)src + i * ld_src * elm_size;
    memcpy(_dst, _src, elm_size * rows);
  }
}

void buildBasis(double(*func)(double), struct Base basis[], int64_t ncells, struct Cell* cells, const struct CSC* rel_near, int64_t levels,
  const struct CellComm* comm, const double* bodies, int64_t nbodies, double epi, int64_t mrank, int64_t sp_pts, int64_t alignment) {

  for (int64_t l = levels; l >= 0; l--) {
    int64_t xlen = 0, ibegin = 0, nodes = 0;
    content_length(&nodes, &xlen, &ibegin, &comm[l]);
    int64_t iend = ibegin + nodes;
    basis[l].Ulen = xlen;
    basis[l].Dims = std::vector<int64_t>(xlen, 0);
    basis[l].DimsLr = std::vector<int64_t>(xlen, 0);
    basis[l].Offsets = std::vector<int64_t>(xlen + 1, 0);

    int64_t jbegin = 0, jend = ncells;
    get_level(&jbegin, &jend, cells, l, -1);
    for (int64_t i = 0; i < xlen; i++) {
      int64_t childi = std::get<0>(comm[l].LocalChild[i]);
      int64_t clen = std::get<1>(comm[l].LocalChild[i]);
      if (childi >= 0)
        for (int64_t j = 0; j < clen; j++)
          basis[l].Dims[i] = basis[l].Dims[i] + basis[l + 1].DimsLr[childi + j];
      else {
        int64_t gi = i;
        i_global(&gi, &comm[l]);
        basis[l].Dims[i] = cells[jbegin + gi].Body[1] - cells[jbegin + gi].Body[0];
      }
    }

    int64_t seg_skeletons = 3 * neighbor_bcast_sizes_cpu(&basis[l].Dims[0], &comm[l]);
    std::vector<double> Skeletons(xlen * seg_skeletons, 0.);
    /*for (int64_t i = 0; i < nodes; i++) {
      int64_t childi = std::get<0>(comm[l].LocalChild[i]);
      int64_t clen = std::get<1>(comm[l].LocalChild[i]);
      if (childi >= 0) {
        int64_t seg = basis[l + 1].dimS * 3;
        int64_t y = 0;
        for (int64_t j = 0; j < clen; j++) {
          int64_t len = 3 * basis[l + 1].DimsLr[childi + j];
          std::copy();
        }
      }
      else {
        int64_t gi = i;
        i_global(&gi, &comm[l]);
        basis[l].Dims[i] = cells[jbegin + gi].Body[1] - cells[jbegin + gi].Body[0];
      }
    }*/

    std::vector<int64_t> lchild_temp(xlen, -1);
    for (int64_t i = 0; i < xlen; i++)
      lchild_temp[i] = std::get<0>(comm[l].LocalChild[i]);

    struct Matrix* arr_m = (struct Matrix*)calloc(xlen * 3, sizeof(struct Matrix));
    basis[l].Uo = arr_m;
    basis[l].Uc = &arr_m[xlen];
    basis[l].R = &arr_m[xlen * 2];

    struct SampleBodies samples;
    buildSampleBodies(&samples, sp_pts, sp_pts, nbodies, ncells, cells, rel_near, &lchild_temp[ibegin], l == levels ? NULL : &basis[l + 1], l);

    int64_t count = 0;
    int64_t count_m = 0;
    for (int64_t i = 0; i < nodes; i++) {
      int64_t ske_len = samples.SkeLens[i];
      int64_t len_m = samples.FarLens[i] < samples.CloseLens[i] ? samples.CloseLens[i] : samples.FarLens[i];
      basis[l].Dims[i + ibegin] = ske_len;
      count = count + ske_len;
      count_m = count_m + ske_len * (ske_len * 2 + len_m + 2);
    }

    int32_t* ipiv_data = (int32_t*)malloc(sizeof(int32_t) * count);
    int32_t** ipiv_ptrs = (int32_t**)malloc(sizeof(int32_t*) * nodes);
    double* matrix_data = (double*)malloc(sizeof(double) * count_m);
    double** matrix_ptrs = (double**)malloc(sizeof(double*) * (xlen + 1));

    count = 0;
    count_m = 0;
    for (int64_t i = 0; i < nodes; i++) {
      int64_t ske_len = samples.SkeLens[i];
      int64_t len_m = samples.FarLens[i] < samples.CloseLens[i] ? samples.CloseLens[i] : samples.FarLens[i];
      ipiv_ptrs[i] = &ipiv_data[count];
      matrix_ptrs[i + ibegin] = &matrix_data[count_m];
      count = count + ske_len;
      count_m = count_m + ske_len * (ske_len * 2 + len_m + 2);
    }

//#pragma omp parallel for
    for (int64_t i = 0; i < nodes; i++) {
      int64_t ske_len = samples.SkeLens[i];
      int64_t len_s = samples.FarLens[i] + (samples.CloseLens[i] > 0 ? ske_len : 0);
      double* mat = matrix_ptrs[i + ibegin];
      struct Matrix S = (struct Matrix){ &mat[ske_len * ske_len], ske_len, len_s, ske_len };
      std::vector<double> Xbodies(3 * ske_len), Cbodies(3 * samples.CloseLens[i]), Fbodies(3 * samples.FarLens[i]);
      for (int64_t j = 0; j < ske_len; j++)
        for (int64_t k = 0; k < 3; k++)
          Xbodies[j * 3 + k] = bodies[samples.Skeletons[i][j] * 3 + k];
      
      for (int64_t j = 0; j < samples.CloseLens[i]; j++)
        for (int64_t k = 0; k < 3; k++)
          Cbodies[j * 3 + k] = bodies[samples.CloseBodies[i][j] * 3 + k];

      for (int64_t j = 0; j < samples.FarLens[i]; j++)
        for (int64_t k = 0; k < 3; k++)
          Fbodies[j * 3 + k] = bodies[samples.FarBodies[i][j] * 3 + k];
      
      //int64_t rank = compute_basis(func, epi, 10, mrank, ske_len, mat, ske_len, &Xbodies[0], Cbodies.size() / 3, &Cbodies[0], Fbodies.size() / 3, &Fbodies[0]);

      if (len_s > 0) {
        struct Matrix S_dn = (struct Matrix){ &mat[ske_len * ske_len], ske_len, ske_len, ske_len };
        double nrm_dn = 0.;
        double nrm_lr = 0.;

        if (samples.CloseLens[i] > 0) {
          struct Matrix S_dn_work = (struct Matrix){ &mat[ske_len * ske_len * 2], ske_len, samples.CloseLens[i], ske_len };
          gen_matrix(func, ske_len, samples.CloseLens[i], Xbodies.data(), Cbodies.data(), S_dn_work.A, ske_len);
          mmult('N', 'T', &S_dn_work, &S_dn_work, &S_dn, 1., 0.);
          nrm2_A(&S_dn, &nrm_dn);
        }

        if (samples.FarLens[i] > 0) {
          struct Matrix S_lr = (struct Matrix){ &mat[ske_len * ske_len * 2], ske_len, samples.FarLens[i], ske_len };
          gen_matrix(func, ske_len, samples.FarLens[i], Xbodies.data(), Fbodies.data(), S_lr.A, ske_len);
          nrm2_A(&S_lr, &nrm_lr);
          if (samples.CloseLens[i] > 0)
            scal_A(&S_dn, nrm_lr / nrm_dn);
        }
      }

      int64_t rank = ske_len < len_s ? ske_len : len_s;
      rank = mrank > 0 ? (mrank < rank ? mrank : rank) : rank;
      if (rank > 0) {
        struct Matrix Q = (struct Matrix){ mat, ske_len, ske_len, ske_len };
        double* Svec = &mat[ske_len * (ske_len + len_s)];
        int32_t* pa = ipiv_ptrs[i];
        svd_U(&S, &Q, Svec);

        if (epi > 0.) {
          int64_t r = 0;
          double sepi = Svec[0] * epi;
          while(r < rank && Svec[r] > sepi)
            r += 1;
          rank = r;
        }

        struct Matrix Qo = (struct Matrix){ mat, ske_len, rank, ske_len };
        struct Matrix R = (struct Matrix){ &mat[ske_len * ske_len], rank, rank, rank };
        id_row(&Qo, pa, S.A);

        int64_t lc = lchild_temp[i + ibegin];
        if (lc >= 0)
          upper_tri_reflec_mult('L', 2, &(basis[l + 1].R)[lc], &Qo);
        qr_full(&Q, &R);

        for (int64_t j = 0; j < rank; j++) {
          int64_t piv = (int64_t)pa[j] - 1;
          if (piv != j) {
            int64_t c = samples.Skeletons[i][piv];
            samples.Skeletons[i][piv] = samples.Skeletons[i][j];
            samples.Skeletons[i][j] = c;
          }
        }
      }

      basis[l].DimsLr[i + ibegin] = rank;
    }
    neighbor_bcast_sizes_cpu(basis[l].DimsLr.data(), &comm[l]);

    count = 0;
    count_m = 0;
    for (int64_t i = 0; i < xlen; i++) {
      int64_t m = basis[l].Dims[i];
      int64_t n = basis[l].DimsLr[i];
      basis[l].Offsets[i] = count;
      count = count + n;
      count_m = count_m + m * m + n * n;
    }
    basis[l].Offsets[xlen] = count;

    basis[l].Multipoles = std::vector<int64_t>(count, 0);
    for (int64_t i = 0; i < nodes; i++) {
      int64_t offset = basis[l].Offsets[i + ibegin];
      int64_t n = basis[l].DimsLr[i + ibegin];
      if (n > 0)
        memcpy(&basis[l].Multipoles[offset], samples.Skeletons[i], sizeof(int64_t) * n);
    }
    dist_int_64(basis[l].Multipoles.data(), basis[l].Offsets.data(), &comm[l]);

    int64_t max[3] = { 0, 0, 0 };
    for (int64_t i = 0; i < xlen; i++) {
      int64_t i1 = basis[l].DimsLr[i];
      int64_t i2 = basis[l].Dims[i] - basis[l].DimsLr[i];
      int64_t rem1 = i1 & (alignment - 1);
      int64_t rem2 = i2 & (alignment - 1);

      i1 = std::max(alignment, i1 - rem1 + (rem1 ? alignment : 0));
      i2 = std::max(alignment, i2 - rem2 + (rem2 ? alignment : 0));
      max[0] = std::max(max[0], i1);
      max[1] = std::max(max[1], i2);
      max[2] = std::max(max[2], std::get<1>(comm[l].LocalChild[i]));
    }
    MPI_Allreduce(MPI_IN_PLACE, max, 3, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);

    int64_t child_s = l < levels ? (basis[l + 1].dimS * max[2]) : 0;
    basis[l].dimS = max[0];
    basis[l].dimR = max[1];
    if (basis[l].dimR + basis[l].dimS < child_s)
      basis[l].dimR = child_s - basis[l].dimS;
    basis[l].dimN = basis[l].dimS + basis[l].dimR;
    int64_t stride = basis[l].dimN * basis[l].dimN;
    int64_t stride_r = basis[l].dimS * basis[l].dimS;
    int64_t LD = basis[l].dimN;

    basis[l].M_cpu = (double*)calloc(basis[l].dimS * xlen * 3, sizeof(double));
    basis[l].U_cpu = (double*)calloc(stride * xlen + nodes * basis[l].dimR, sizeof(double));
    basis[l].R_cpu = (double*)calloc(stride_r * xlen, sizeof(double));
    if (cudaMalloc(&basis[l].M_gpu, sizeof(double) * basis[l].dimS * xlen * 3) != cudaSuccess)
      basis[l].M_gpu = NULL;
    if (cudaMalloc(&basis[l].U_gpu, sizeof(double) * (stride * xlen + nodes * basis[l].dimR)) != cudaSuccess)
      basis[l].U_gpu = NULL;
    if (cudaMalloc(&basis[l].R_gpu, sizeof(double) * stride_r * xlen) != cudaSuccess)
      basis[l].R_gpu = NULL;

    for (int64_t i = 0; i < xlen; i++) {
      double* M_ptr = basis[l].M_cpu + i * basis[l].dimS * 3;
      double* Uc_ptr = basis[l].U_cpu + i * stride;
      double* Uo_ptr = Uc_ptr + basis[l].dimR * basis[l].dimN;
      double* R_ptr = basis[l].R_cpu + i * stride_r;

      int64_t Nc = basis[l].Dims[i] - basis[l].DimsLr[i];
      int64_t No = basis[l].DimsLr[i];

      for (int64_t j = 0; j < No; j++)
        for (int64_t k = 0; k < 3; k++)
          M_ptr[j * 3 + k] = bodies[basis[l].Multipoles[basis[l].Offsets[i] + j] * 3 + k];

      if (ibegin <= i && i < iend) {
        int64_t child = std::get<0>(comm[l].LocalChild[i]);
        int64_t clen = std::get<1>(comm[l].LocalChild[i]);
        int64_t M = basis[l].Dims[i];
        if (child >= 0 && l < levels) {
          int64_t row = 0;
          for (int64_t j = 0; j < clen; j++) {
            int64_t N = basis[l + 1].DimsLr[child + j];
            int64_t Urow = j * basis[l + 1].dimS;
            memcpy2d(&Uc_ptr[Urow], matrix_ptrs[i] + No * M + row, N, Nc, LD, M, sizeof(double));
            memcpy2d(&Uo_ptr[Urow], matrix_ptrs[i] + row, N, No, LD, M, sizeof(double));
            row = row + N;
          }
        }
        else {
          memcpy2d(Uc_ptr, matrix_ptrs[i] + No * M, M, Nc, LD, M, sizeof(double));
          memcpy2d(Uo_ptr, matrix_ptrs[i], M, No, LD, M, sizeof(double));
        }
        memcpy2d(R_ptr, matrix_ptrs[i] + M * M, No, No, basis[l].dimS, No, sizeof(double));

        double* Ui_ptr = basis[l].U_cpu + xlen * stride + (i - ibegin) * basis[l].dimR; 
        for (int64_t j = Nc; j < basis[l].dimR; j++)
          Ui_ptr[j] = 1.;
      }

      basis[l].Uo[i] = (struct Matrix) { Uo_ptr, basis[l].dimN, basis[l].dimS, basis[l].dimN };
      basis[l].Uc[i] = (struct Matrix) { Uc_ptr, basis[l].dimN, basis[l].dimR, basis[l].dimN };
      basis[l].R[i] = (struct Matrix) { R_ptr, No, No, basis[l].dimS };
    }
    neighbor_bcast_cpu(basis[l].U_cpu, stride, &comm[l]);
    dup_bcast_cpu(basis[l].U_cpu, stride * xlen, &comm[l]);
    neighbor_bcast_cpu(basis[l].R_cpu, stride_r, &comm[l]);
    dup_bcast_cpu(basis[l].R_cpu, stride_r * xlen, &comm[l]);

    free(ipiv_data);
    free(ipiv_ptrs);
    free(matrix_data);
    free(matrix_ptrs);
    sampleBodies_free(&samples);

    if (basis[l].M_gpu)
      cudaMemcpy(basis[l].M_gpu, basis[l].M_cpu, sizeof(double) * basis[l].dimS * xlen * 3, cudaMemcpyHostToDevice);
    if (basis[l].U_gpu)
      cudaMemcpy(basis[l].U_gpu, basis[l].U_cpu, sizeof(double) * (stride * xlen + nodes * basis[l].dimR), cudaMemcpyHostToDevice);
    if (basis[l].R_gpu)
      cudaMemcpy(basis[l].R_gpu, basis[l].R_cpu, sizeof(double) * stride_r * xlen, cudaMemcpyHostToDevice);
  }
}
