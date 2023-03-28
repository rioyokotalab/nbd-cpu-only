
#include "nbd.hxx"
#include "profile.hxx"

#include "stdio.h"
#include "string.h"
#include "math.h"

#include <cstdlib>
#include <algorithm>

/*void memcpy2d(void* dst, const void* src, int64_t rows, int64_t cols, int64_t ld_dst, int64_t ld_src, size_t elm_size) {
  for (int64_t i = 0; i < cols; i++) {
    unsigned char* _dst = (unsigned char*)dst + i * ld_dst * elm_size;
    const unsigned char* _src = (const unsigned char*)src + i * ld_src * elm_size;
    memcpy(_dst, _src, elm_size * rows);
  }
}

void buildBasis(int alignment, struct Base basis[], int64_t ncells, struct Cell* cells, struct CellBasis* cell_basis, int64_t levels, const struct CellComm* comm) {
  std::vector<int64_t> dimSmax(levels + 1, 0);
  std::vector<int64_t> dimRmax(levels + 1, 0);
  std::vector<int64_t> nchild(levels + 1, 0);
  
  for (int64_t l = levels; l >= 0; l--) {
    int64_t xlen = 0;
    content_length(NULL, &xlen, NULL, &comm[l]);
    basis[l].Ulen = xlen;
    basis[l].Dims = std::vector<int64_t>(xlen, 0);
    basis[l].DimsLr = std::vector<int64_t>(xlen, 0);

    struct Matrix* arr_m = (struct Matrix*)calloc(xlen * 3, sizeof(struct Matrix));
    basis[l].Uo = arr_m;
    basis[l].Uc = &arr_m[xlen];
    basis[l].R = &arr_m[xlen * 2];

    int64_t lbegin = 0, lend = ncells;
    get_level(&lbegin, &lend, cells, l, -1);

    for (int64_t i = 0; i < xlen; i++) {
      int64_t gi = i;
      i_global(&gi, &comm[l]);
      int64_t ci = lbegin + gi;
      basis[l].Dims[i] = cell_basis[ci].M;
      basis[l].DimsLr[i] = cell_basis[ci].N;

      dimSmax[l] = std::max(dimSmax[l], basis[l].DimsLr[i]);
      dimRmax[l] = std::max(dimRmax[l], basis[l].Dims[i] - basis[l].DimsLr[i]);
      nchild[l] = std::max(nchild[l], std::get<1>(comm[l].LocalChild[i]));
    }
  }

  get_segment_sizes(dimSmax.data(), dimRmax.data(), nchild.data(), alignment, levels);

  for (int64_t l = levels; l >= 0; l--) {
    basis[l].dimS = dimSmax[l];
    basis[l].dimR = dimRmax[l];
    basis[l].dimN = basis[l].dimS + basis[l].dimR;
    int64_t stride = basis[l].dimN * basis[l].dimN;
    int64_t stride_r = basis[l].dimS * basis[l].dimS;
    int64_t LD = basis[l].dimN;

    int64_t ibegin = 0, ilen = 0;
    content_length(&ilen, NULL, &ibegin, &comm[l]);

    basis[l].M_cpu = (double*)calloc(basis[l].dimS * basis[l].Ulen * 3, sizeof(double));
    basis[l].U_cpu = (double*)calloc(stride * basis[l].Ulen + ilen * basis[l].dimR, sizeof(double));
    basis[l].R_cpu = (double*)calloc(stride_r * basis[l].Ulen, sizeof(double));
    if (cudaMalloc(&basis[l].M_gpu, sizeof(double) * basis[l].dimS * basis[l].Ulen * 3) != cudaSuccess)
      basis[l].M_gpu = NULL;
    if (cudaMalloc(&basis[l].U_gpu, sizeof(double) * (stride * basis[l].Ulen + ilen * basis[l].dimR)) != cudaSuccess)
      basis[l].U_gpu = NULL;
    if (cudaMalloc(&basis[l].R_gpu, sizeof(double) * stride_r * basis[l].Ulen) != cudaSuccess)
      basis[l].R_gpu = NULL;

    int64_t lbegin = 0, lend = ncells;
    get_level(&lbegin, &lend, cells, l, -1);

    for (int64_t i = 0; i < basis[l].Ulen; i++) {
      double* M_ptr = basis[l].M_cpu + i * basis[l].dimS * 3;
      double* Uc_ptr = basis[l].U_cpu + i * stride;
      double* Uo_ptr = Uc_ptr + basis[l].dimR * basis[l].dimN;
      double* R_ptr = basis[l].R_cpu + i * stride_r;

      int64_t Nc = basis[l].Dims[i] - basis[l].DimsLr[i];
      int64_t No = basis[l].DimsLr[i];
      basis[l].Uo[i] = (struct Matrix) { Uo_ptr, basis[l].dimN, basis[l].dimS, basis[l].dimN };
      basis[l].Uc[i] = (struct Matrix) { Uc_ptr, basis[l].dimN, basis[l].dimR, basis[l].dimN };
      basis[l].R[i] = (struct Matrix) { R_ptr, No, No, basis[l].dimS };

      int64_t gi = i;
      i_global(&gi, &comm[l]);
      int64_t ci = lbegin + gi;

      memcpy(M_ptr, cell_basis[ci].Multipoles, sizeof(double) * No * 3);
      memcpy2d(R_ptr, cell_basis[ci].R, No, No, basis[l].dimS, No, sizeof(double));
      if (ibegin <= i && i < (ibegin + ilen)) {
        double* Ui_ptr = basis[l].U_cpu + basis[l].Ulen * stride + (i - ibegin) * basis[l].dimR; 
        for (int64_t j = Nc; j < basis[l].dimR; j++)
          Ui_ptr[j] = 1.;
      }

      int64_t child = std::get<0>(comm[l].LocalChild[i]);
      int64_t clen = std::get<1>(comm[l].LocalChild[i]);
      if (child >= 0 && l < levels) {
        int64_t row = 0;

        for (int64_t j = 0; j < clen; j++) {
          int64_t M = basis[l + 1].DimsLr[child + j];
          int64_t Urow = j * basis[l + 1].dimS;
          memcpy2d(&Uc_ptr[Urow], &cell_basis[ci].Uc[row], M, Nc, LD, Nc + No, sizeof(double));
          memcpy2d(&Uo_ptr[Urow], &cell_basis[ci].Uo[row], M, No, LD, Nc + No, sizeof(double));
          row = row + M;
        }
      }
      else {
        int64_t M = basis[l].Dims[i];
        memcpy2d(Uc_ptr, cell_basis[ci].Uc, M, Nc, LD, Nc + No, sizeof(double));
        memcpy2d(Uo_ptr, cell_basis[ci].Uo, M, No, LD, Nc + No, sizeof(double));
      }
    }

    if (basis[l].M_gpu)
      cudaMemcpy(basis[l].M_gpu, basis[l].M_cpu, sizeof(double) * basis[l].dimS * basis[l].Ulen * 3, cudaMemcpyHostToDevice);
    if (basis[l].U_gpu)
      cudaMemcpy(basis[l].U_gpu, basis[l].U_cpu, sizeof(double) * (stride * basis[l].Ulen + ilen * basis[l].dimR), cudaMemcpyHostToDevice);
    if (basis[l].R_gpu)
      cudaMemcpy(basis[l].R_gpu, basis[l].R_cpu, sizeof(double) * stride_r * basis[l].Ulen, cudaMemcpyHostToDevice);
  }
}*/

void basis_free(struct Base* basis) {
  free(basis->Uo);
  if (basis->M_cpu)
    free(basis->M_cpu);
  if (basis->M_gpu)
    cudaFree(basis->M_gpu);
  if (basis->U_cpu)
    free(basis->U_cpu);
  if (basis->U_gpu)
    cudaFree(basis->U_gpu);
  if (basis->R_cpu)
    free(basis->R_cpu);
  if (basis->R_gpu)
    cudaFree(basis->R_gpu);
}

void allocNodes(struct Node A[], double** Workspace, int64_t* Lwork, const struct Base basis[], const struct CSC rels_near[], const struct CSC rels_far[], const struct CellComm comm[], int64_t levels) {
  int64_t work_size = 0;
  int64_t pers_work = 0;

  for (int64_t i = levels; i >= 0; i--) {
    int64_t n_i = 0, ulen = 0, nloc = 0;
    content_length(&n_i, &ulen, &nloc, &comm[i]);
    int64_t nnz = rels_near[i].ColIndex[n_i];
    int64_t nnz_f = rels_far[i].ColIndex[n_i];
    int64_t len_arr = nnz * 4 + nnz_f;

    struct Matrix* arr_m = (struct Matrix*)malloc(sizeof(struct Matrix) * len_arr);
    A[i].A = arr_m;
    A[i].A_cc = &arr_m[nnz];
    A[i].A_oc = &arr_m[nnz * 2];
    A[i].A_oo = &arr_m[nnz * 3];
    A[i].S = &arr_m[nnz * 4];

    A[i].lenA = nnz;
    A[i].lenS = nnz_f;

    int64_t dims = basis[i].dimS;
    int64_t dimr = basis[i].dimR;
    int64_t dimn = basis[i].dimR + basis[i].dimS;
    int64_t dimn_up = i > 0 ? basis[i - 1].dimN : 0;

    int64_t stride = dimn * dimn;
    allocBufferedList((void**)&A[i].A_ptr, (void**)&A[i].A_buf, sizeof(double), stride * nnz);
    allocBufferedList((void**)&A[i].X_ptr, (void**)&A[i].X_buf, sizeof(double), dimn * ulen);
    allocBufferedList((void**)&A[i].M_ptr, (void**)&A[i].M_buf, sizeof(double), dims * ulen * 3);
    allocBufferedList((void**)&A[i].U_ptr, (void**)&A[i].U_buf, sizeof(double), stride * ulen + n_i * dimr);
    allocBufferedList((void**)&A[i].R_ptr, (void**)&A[i].R_buf, sizeof(double), dims * dims * ulen);

    int64_t work_required = n_i * stride * 2;
    int64_t pers_required = dimr * (ulen + n_i * (dimr + 1));
    pers_work = pers_work + pers_required;
    work_size = std::max(work_size, work_required + pers_work);

    for (int64_t x = 0; x < n_i; x++) {
      int64_t box_x = nloc + x;
      int64_t diml_x = basis[i].DimsLr[box_x];

      for (int64_t yx = rels_near[i].ColIndex[x]; yx < rels_near[i].ColIndex[x + 1]; yx++) {
        int64_t y = rels_near[i].RowIndex[yx];
        int64_t diml_y = basis[i].DimsLr[y];

        arr_m[yx] = (struct Matrix) { &A[i].A_buf[yx * stride], dimn, dimn, dimn }; // A
        arr_m[yx + nnz] = (struct Matrix) { &A[i].A_buf[yx * stride], dimr, dimr, dimn }; // A_cc
        arr_m[yx + nnz * 2] = (struct Matrix) { &A[i].A_buf[yx * stride + dimr], dims, dimr, dimn }; // A_oc
        arr_m[yx + nnz * 3] = (struct Matrix) { NULL, diml_y, diml_x, dimn_up }; // A_oo
      }

      for (int64_t yx = rels_far[i].ColIndex[x]; yx < rels_far[i].ColIndex[x + 1]; yx++) {
        int64_t y = rels_far[i].RowIndex[yx];
        int64_t diml_y = basis[i].DimsLr[y];
        arr_m[yx + nnz * 4] = (struct Matrix) { NULL, diml_y, diml_x, dimn_up }; // S
      }
    }

    if (i < levels) {
      int64_t ploc = 0;
      content_length(NULL, NULL, &ploc, &comm[i + 1]);
      int64_t seg = basis[i + 1].dimS;

      for (int64_t j = 0; j < rels_near[i].N; j++) {
        int64_t x0 = std::get<0>(comm[i].LocalChild[j + nloc]) - ploc;
        int64_t lenx = std::get<1>(comm[i].LocalChild[j + nloc]);

        for (int64_t ij = rels_near[i].ColIndex[j]; ij < rels_near[i].ColIndex[j + 1]; ij++) {
          int64_t li = rels_near[i].RowIndex[ij];
          int64_t y0 = std::get<0>(comm[i].LocalChild[li]);
          int64_t leny = std::get<1>(comm[i].LocalChild[li]);

          for (int64_t x = 0; x < lenx; x++)
            if ((x + x0) >= 0 && (x + x0) < rels_near[i + 1].N)
              for (int64_t yx = rels_near[i + 1].ColIndex[x + x0]; yx < rels_near[i + 1].ColIndex[x + x0 + 1]; yx++)
                for (int64_t y = 0; y < leny; y++)
                  if (rels_near[i + 1].RowIndex[yx] == (y + y0))
                    A[i + 1].A_oo[yx].A = &A[i].A[ij].A[(y + dimn * x) * seg];
          
          for (int64_t x = 0; x < lenx; x++)
            if ((x + x0) >= 0 && (x + x0) < rels_far[i + 1].N)
              for (int64_t yx = rels_far[i + 1].ColIndex[x + x0]; yx < rels_far[i + 1].ColIndex[x + x0 + 1]; yx++)
                for (int64_t y = 0; y < leny; y++)
                  if (rels_far[i + 1].RowIndex[yx] == (y + y0))
                    A[i + 1].S[yx].A = &A[i].A[ij].A[(y + dimn * x) * seg];
        }
      }
    }
  }
  
  set_work_size(work_size, Workspace, Lwork);
  for (int64_t i = levels; i > 0; i--) {
    int64_t ibegin = 0;
    content_length(NULL, NULL, &ibegin, &comm[i]);
    int64_t nnz = A[i].lenA;
    int64_t dimc = basis[i].dimR;
    int64_t dimr = basis[i].dimS;

    double** A_next = (double**)malloc(sizeof(double*) * nnz);
    double** X_next = (double**)malloc(sizeof(double*) * basis[i].Ulen);
    int64_t n_next = basis[i - 1].dimR + basis[i - 1].dimS;

    for (int64_t x = 0; x < nnz; x++)
      A_next[x] = A[i - 1].A_ptr + (A[i].A_oo[x].A - A[i - 1].A_buf);

    for (int64_t x = 0; x < basis[i - 1].Ulen; x++) {
      int64_t child = std::get<0>(comm[i - 1].LocalChild[x]);
      int64_t clen = std::get<1>(comm[i - 1].LocalChild[x]);
      
      if (child >= 0)
        for (int64_t j = 0; j < clen; j++)
          X_next[child + j] = &A[i - 1].X_ptr[j * basis[i].dimS + x * n_next];
    }

    int64_t alloc = batchParamsCreate(&A[i].params, dimc, dimr, basis[i].U_gpu, A[i].A_ptr, A[i].X_ptr, n_next, A_next, X_next,
      *Workspace, work_size, basis[i].Ulen, rels_near[i].N, ibegin, rels_near[i].RowIndex, rels_near[i].ColIndex);
    work_size = work_size - alloc;
    free(A_next);
    free(X_next);
  }

  int64_t child = std::get<0>(comm[0].LocalChild[0]);
  int64_t clen = std::get<1>(comm[0].LocalChild[0]);
  std::vector<int64_t> cdims(clen);
  if (child >= 0)
    for (int64_t i = 0; i < clen; i++)
      cdims[i] = basis[1].DimsLr[child + i];
  else
    cdims.emplace_back(basis[0].Dims[0]);
  int64_t low_s = clen > 0 ? basis[1].dimS : 0;
  lastParamsCreate(&A[0].params, A[0].A_ptr, A[0].X_ptr, basis[0].dimN, low_s, cdims.size(), &cdims[0]);
}

void node_free(struct Node* node) {
  freeBufferedList(node->A_ptr, node->A_buf);
  freeBufferedList(node->X_ptr, node->X_buf);
  freeBufferedList(node->M_ptr, node->M_buf);
  freeBufferedList(node->U_ptr, node->U_buf);
  freeBufferedList(node->R_ptr, node->R_buf);
  free(node->A);
  batchParamsDestory(&node->params);
}

void factorA_mov_mem(char dir, struct Node A[], const struct Base basis[], int64_t levels) {
  for (int64_t i = 0; i <= levels; i++) {
    int64_t stride = basis[i].dimN * basis[i].dimN;
    flushBuffer(dir, A[i].A_ptr, A[i].A_buf, sizeof(double), stride * A[i].lenA);
  }
}

void factorA(struct Node A[], const struct Base basis[], const struct CellComm comm[], int64_t levels) {

  for (int64_t i = levels; i > 0; i--) {
    batchCholeskyFactor(&A[i].params, &comm[i]);
#ifdef _PROF
    int64_t ibegin = 0, ilen = 0;
    content_length(&ilen, NULL, &ibegin, &comm[i]);
    int64_t nnz = A[i].lenA;
    record_factor_flops(basis[i].dimR, basis[i].dimS, nnz, ilen);
#endif
  }
  chol_decomp(&A[0].params, &comm[0]);
#ifdef _PROF
  record_factor_flops(0, basis[0].dimN, 1, 1);
#endif
}

void allocRightHandSidesMV(struct RightHandSides rhs[], const struct Base base[], const struct CellComm comm[], int64_t levels) {
  for (int64_t l = levels; l >= 0; l--) {
    int64_t len = base[l].Ulen;
    int64_t len_arr = len * 4;
    struct Matrix* arr_m = (struct Matrix*)malloc(sizeof(struct Matrix) * len_arr);
    rhs[l].Xlen = len;
    rhs[l].X = arr_m;
    rhs[l].B = &arr_m[len];
    rhs[l].Xo = &arr_m[len * 2];
    rhs[l].Xc = &arr_m[len * 3];

    int64_t len_data = len * base[l].dimN * 2;
    double* data = (double*)calloc(len_data, sizeof(double));
    for (int64_t i = 0; i < len; i++) {
      arr_m[i] = (struct Matrix) { &data[i * base[l].dimN], base[l].dimN, 1, base[l].dimN }; // X
      arr_m[i + len] = (struct Matrix) { &data[len * base[l].dimN + i * base[l].dimN], base[l].dimN, 1, base[l].dimN }; // B
      arr_m[i + len * 2] = (struct Matrix) { NULL, base[l].dimS, 1, base[l].dimS }; // Xo
      arr_m[i + len * 3] = (struct Matrix) { NULL, base[l].dimS, 1, base[l].dimS }; // Xc
    }

    for (int64_t i = 0; i < len; i++) {
      int64_t child = std::get<0>(comm[l].LocalChild[i]);
      int64_t clen = std::get<1>(comm[l].LocalChild[i]);
      
      if (child >= 0 && l < levels)
        for (int64_t j = 0; j < clen; j++) {
          rhs[l + 1].Xo[child + j].A = &rhs[l].X[i].A[j * base[l + 1].dimS];
          rhs[l + 1].Xc[child + j].A = &rhs[l].B[i].A[j * base[l + 1].dimS];
        }
    }
  }
}

void rightHandSides_free(struct RightHandSides* rhs) {
  double* data = rhs->X[0].A;
  if (data)
    free(data);
  free(rhs->X);
}

void matVecA(struct RightHandSides rhs[], const struct Node A[], const struct Base basis[], const struct CSC rels_near[], double* X, const struct CellComm comm[], int64_t levels) {
  int64_t lbegin = 0, llen = 0;
  content_length(&llen, NULL, &lbegin, &comm[levels]);
  memcpy(rhs[levels].X[lbegin].A, X, llen * basis[levels].dimN * sizeof(double));

  for (int64_t i = levels; i > 0; i--) {
    int64_t xlen = rhs[i].Xlen;
    level_merge_cpu(rhs[i].X[0].A, xlen * basis[i].dimN, &comm[i]);
    neighbor_bcast_cpu(rhs[i].X[0].A, basis[i].dimN, &comm[i]);
    dup_bcast_cpu(rhs[i].X[0].A, xlen * basis[i].dimN, &comm[i]);

    int64_t ibegin = 0, iboxes = 0;
    content_length(&iboxes, NULL, &ibegin, &comm[i]);
    for (int64_t j = 0; j < iboxes; j++)
      mmult('T', 'N', &basis[i].Uo[j + ibegin], &rhs[i].X[j + ibegin], &rhs[i].Xo[j + ibegin], 1., 0.);
  }

  mmult('T', 'N', &A[0].A[0], &rhs[0].X[0], &rhs[0].B[0], 1., 0.);

  for (int64_t i = 1; i <= levels; i++) {
    int64_t ibegin = 0, iboxes = 0;
    content_length(&iboxes, NULL, &ibegin, &comm[i]);
    for (int64_t j = 0; j < iboxes; j++)
      mmult('N', 'N', &basis[i].Uo[j + ibegin], &rhs[i].Xc[j + ibegin], &rhs[i].B[j + ibegin], 1., 0.);
    for (int64_t y = 0; y < rels_near[i].N; y++)
      for (int64_t xy = rels_near[i].ColIndex[y]; xy < rels_near[i].ColIndex[y + 1]; xy++) {
        int64_t x = rels_near[i].RowIndex[xy];
        mmult('T', 'N', &A[i].A[xy], &rhs[i].X[x], &rhs[i].B[y + ibegin], 1., 1.);
      }
  }
  memcpy(X, rhs[levels].B[lbegin].A, llen * basis[levels].dimN * sizeof(double));
}


void solveRelErr(double* err_out, const double* X, const double* ref, int64_t lenX) {
  double err[2] = { 0., 0. };
  for (int64_t i = 0; i < lenX; i++) {
    double diff = X[i] - ref[i];
    err[0] = err[0] + diff * diff;
    err[1] = err[1] + ref[i] * ref[i];
  }
  MPI_Allreduce(MPI_IN_PLACE, err, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  *err_out = sqrt(err[0] / err[1]);
}

