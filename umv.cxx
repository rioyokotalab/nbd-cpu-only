
#include "nbd.hxx"
#include "profile.hxx"

#include "stdio.h"
#include "string.h"
#include "math.h"

#include <cstdlib>
#include <algorithm>

void memcpy2d(void* dst, const void* src, int64_t rows, int64_t cols, int64_t ld_dst, int64_t ld_src, size_t elm_size) {
  for (int64_t i = 0; i < cols; i++) {
    unsigned char* _dst = (unsigned char*)dst + i * ld_dst * elm_size;
    const unsigned char* _src = (const unsigned char*)src + i * ld_src * elm_size;
    memcpy(_dst, _src, elm_size * rows);
  }
}

void randomize2d(double* dst, int64_t rows, int64_t cols, int64_t ld) {
  for (int64_t j = 0; j < cols; j++)
    for (int64_t i = 0; i < rows; i++)
      dst[i + j * ld] = ((double)std::rand() + 1) / RAND_MAX;
}

void buildBasis(int alignment, struct Base basis[], int64_t ncells, struct Cell* cells, struct CellBasis* cell_basis, int64_t levels, const struct CellComm* comm) {
  std::vector<int64_t> dimSmax(levels + 1, 0);
  std::vector<int64_t> dimRmax(levels + 1, 0);
  std::vector<int64_t> nchild(levels + 1, 0);
  std::srand(999);
  
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
    basis[l].Multipoles = (int64_t**)malloc(sizeof(int64_t*) * xlen);

    int64_t lbegin = 0, lend = ncells;
    get_level(&lbegin, &lend, cells, l, -1);

    for (int64_t i = 0; i < xlen; i++) {
      int64_t gi = i;
      i_global(&gi, &comm[l]);
      int64_t ci = lbegin + gi;
      basis[l].Lchild[i] = std::get<0>(comm[l].LocalChild[i]);
      basis[l].Dims[i] = cell_basis[ci].M;
      basis[l].DimsLr[i] = cell_basis[ci].N;
      basis[l].Uo[i] = (struct Matrix) { cell_basis[ci].Uo, cell_basis[ci].M, cell_basis[ci].N, cell_basis[ci].M };
      basis[l].Uc[i] = (struct Matrix) { cell_basis[ci].Uc, cell_basis[ci].M, cell_basis[ci].M - cell_basis[ci].N, cell_basis[ci].M };
      basis[l].R[i] = (struct Matrix) { cell_basis[ci].R, cell_basis[ci].N, cell_basis[ci].N, cell_basis[ci].N };
      basis[l].Multipoles[i] = cell_basis[ci].Multipoles;

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
    basis[l].padN = nchild[l];
    int64_t stride = basis[l].dimN * basis[l].dimN;
    int64_t stride_r = basis[l].dimS * basis[l].dimS;
    int64_t LD = basis[l].dimN;

    basis[l].U_cpu = (double*)calloc(stride * basis[l].Ulen, sizeof(double));
    basis[l].R_cpu = (double*)calloc(stride_r * basis[l].Ulen, sizeof(double));
    if (cudaMalloc(&basis[l].U_gpu, sizeof(double) * stride * basis[l].Ulen) != cudaSuccess)
      basis[l].U_gpu = NULL;
    if (cudaMalloc(&basis[l].R_gpu, sizeof(double) * stride_r * basis[l].Ulen) != cudaSuccess)
      basis[l].R_gpu = NULL;

    for (int64_t i = 0; i < basis[l].Ulen; i++) {
      double* Uc_ptr = basis[l].U_cpu + i * stride;
      double* Uo_ptr = Uc_ptr + basis[l].dimR * basis[l].dimN;
      double* R_ptr = basis[l].R_cpu + i * stride_r;

      int64_t Nc = basis[l].Dims[i] - basis[l].DimsLr[i];
      int64_t No = basis[l].DimsLr[i];
      memcpy2d(R_ptr, basis[l].R[i].A, No, No, basis[l].dimS, basis[l].R[i].LDA, sizeof(double));

      int64_t child = std::get<0>(comm[l].LocalChild[i]);
      int64_t clen = std::get<1>(comm[l].LocalChild[i]);
      if (child >= 0 && l < levels) {
        int64_t row = 0;

        for (int64_t j = 0; j < clen; j++) {
          int64_t M = basis[l + 1].DimsLr[child + j];
          int64_t Urow = j * basis[l + 1].dimS;
          memcpy2d(&Uc_ptr[Urow], &basis[l].Uc[i].A[row], M, Nc, LD, basis[l].Uc[i].LDA, sizeof(double));
          memcpy2d(&Uo_ptr[Urow], &basis[l].Uo[i].A[row], M, No, LD, basis[l].Uo[i].LDA, sizeof(double));

          randomize2d(&Uc_ptr[Urow + M + Nc * LD], basis[l + 1].dimS - M, basis[l].dimR - Nc, LD);
          randomize2d(&Uo_ptr[Urow + M + No * LD], basis[l + 1].dimS - M, basis[l].dimS - No, LD);
          row = row + M;
        }

        int64_t pad = basis[l].padN;
        randomize2d(&Uc_ptr[LD - pad + Nc * LD], pad, basis[l].dimR - Nc, LD);
        randomize2d(&Uo_ptr[LD - pad + No * LD], pad, basis[l].dimS - No, LD);
      }
      else {
        int64_t M = basis[l].Dims[i];
        memcpy2d(Uc_ptr, basis[l].Uc[i].A, M, Nc, LD, basis[l].Uc[i].LDA, sizeof(double));
        memcpy2d(Uo_ptr, basis[l].Uo[i].A, M, No, LD, basis[l].Uo[i].LDA, sizeof(double));

        randomize2d(&Uc_ptr[M + Nc * LD], LD - M, basis[l].dimR - Nc, LD);
        randomize2d(&Uo_ptr[M + No * LD], LD - M, basis[l].dimS - No, LD);
      }
    }

    if (basis[l].U_gpu)
      cudaMemcpy(basis[l].U_gpu, basis[l].U_cpu, sizeof(double) * stride * basis[l].Ulen, cudaMemcpyHostToDevice);
    if (basis[l].R_gpu)
      cudaMemcpy(basis[l].R_gpu, basis[l].R_cpu, sizeof(double) * stride_r * basis[l].Ulen, cudaMemcpyHostToDevice);
  }
}

void basis_free(struct Base* basis) {
  free(basis->Multipoles);
  free(basis->Lchild);
  free(basis->Uo);
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
  const int64_t NCHILD = 2;

  for (int64_t i = 0; i <= levels; i++) {
    int64_t n_i = rels_near[i].N;
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

    int64_t dimn = basis[i].dimR + basis[i].dimS;
    int64_t stride = dimn * dimn;
    allocBufferedList((void**)&A[i].A_ptr, (void**)&A[i].A_buf, sizeof(double), stride * nnz);
    int64_t work_required = n_i * stride * 2;
    work_size = work_size < work_required ? work_required : work_size;

    int64_t nloc = 0, nend = 0;
    self_local_range(&nloc, &nend, &comm[i]);

    for (int64_t x = 0; x < n_i; x++) {
      int64_t box_x = nloc + x;
      int64_t dim_x = basis[i].Dims[box_x];
      int64_t diml_x = basis[i].DimsLr[box_x];
      int64_t dimc_x = dim_x - diml_x;

      for (int64_t yx = rels_near[i].ColIndex[x]; yx < rels_near[i].ColIndex[x + 1]; yx++) {
        int64_t y = rels_near[i].RowIndex[yx];
        int64_t dim_y = basis[i].Dims[y];
        int64_t diml_y = basis[i].DimsLr[y];
        int64_t dimc_y = dim_y - diml_y;

        arr_m[yx] = (struct Matrix) { A[i].A_buf + yx * stride, dim_y, dim_x, dimn }; // A
        arr_m[yx + nnz] = (struct Matrix) { A[i].A_buf + yx * stride, dimc_y, dimc_x, dimn }; // A_cc
        arr_m[yx + nnz * 2] = (struct Matrix) { A[i].A_buf + yx * stride + basis[i].dimR, diml_y, dimc_x, dimn }; // A_oc
        arr_m[yx + nnz * 3] = (struct Matrix) { NULL, diml_y, diml_x, 0 }; // A_oo

        if (y == box_x)
          for (int64_t z = 0; z < basis[i].padN; z++)
            A[i].A_buf[yx * stride + (z + dimn - basis[i].padN) * (dimn + 1)] = 1.;
      }

      for (int64_t yx = rels_far[i].ColIndex[x]; yx < rels_far[i].ColIndex[x + 1]; yx++) {
        int64_t y = rels_far[i].RowIndex[yx];
        int64_t diml_y = basis[i].DimsLr[y];
        arr_m[yx + nnz * 4] = (struct Matrix) { NULL, diml_y, diml_x, 0 }; // S
      }
    }

    if (i > 0) {
      const struct CSC* rels_up = &rels_near[i - 1];
      const struct Matrix* Mup = A[i - 1].A;
      const int64_t* lchild = basis[i - 1].Lchild;
      int64_t ploc = 0, pend = 0;
      self_local_range(&ploc, &pend, &comm[i - 1]);

      for (int64_t j = 0; j < rels_up->N; j++) {
        int64_t lj = j + ploc;
        int64_t cj = lchild[lj];
        int64_t x0 = cj - nloc;

        for (int64_t ij = rels_up->ColIndex[j]; ij < rels_up->ColIndex[j + 1]; ij++) {
          int64_t li = rels_up->RowIndex[ij];
          int64_t y0 = lchild[li];

          for (int64_t x = 0; x < NCHILD; x++)
            if ((x + x0) >= 0 && (x + x0) < rels_near[i].N)
              for (int64_t yx = rels_near[i].ColIndex[x + x0]; yx < rels_near[i].ColIndex[x + x0 + 1]; yx++)
                for (int64_t y = 0; y < NCHILD; y++)
                  if (rels_near[i].RowIndex[yx] == (y + y0)) {
                    arr_m[yx + nnz * 3].A = Mup[ij].A + Mup[ij].LDA * x * basis[i].dimS + y * basis[i].dimS;
                    arr_m[yx + nnz * 3].LDA = Mup[ij].LDA;
                  }
          
          for (int64_t x = 0; x < NCHILD; x++)
            if ((x + x0) >= 0 && (x + x0) < rels_far[i].N)
              for (int64_t yx = rels_far[i].ColIndex[x + x0]; yx < rels_far[i].ColIndex[x + x0 + 1]; yx++)
                for (int64_t y = 0; y < NCHILD; y++)
                  if (rels_far[i].RowIndex[yx] == (y + y0)) {
                    arr_m[yx + nnz * 4].A = Mup[ij].A + Mup[ij].LDA * x * basis[i].dimS + y * basis[i].dimS;
                    arr_m[yx + nnz * 4].LDA = Mup[ij].LDA;
                  }
        }
      }
    }
  }
  
  set_work_size(work_size, Workspace, Lwork);
  for (int64_t i = 1; i <= levels; i++) {
    int64_t ibegin = 0, iend = 0;
    self_local_range(&ibegin, &iend, &comm[i]);
    int64_t llen = basis[i].Ulen;
    int64_t nnz = A[i].lenA;
    int64_t dimc = basis[i].dimR;
    int64_t dimr = basis[i].dimS;

    double** A_next = (double**)malloc(sizeof(double*) * nnz);
    int64_t* dimc_lis = (int64_t*)malloc(sizeof(int64_t) * llen);
    int64_t n_next = basis[i - 1].dimR + basis[i - 1].dimS;

    for (int64_t x = 0; x < nnz; x++)
      A_next[x] = A[i - 1].A_ptr + (A[i].A_oo[x].A - A[i - 1].A_buf);
    for (int64_t x = 0; x < llen; x++)
      dimc_lis[x] = basis[i].Dims[x] - basis[i].DimsLr[x];

    batchParamsCreate(&A[i].params, dimc, dimr, basis[i].U_cpu, A[i].A_ptr, n_next, A_next, *Workspace, *Lwork,
      rels_near[i].N, ibegin, rels_near[i].RowIndex, rels_near[i].ColIndex, dimc_lis, comm[i].Comm_merge, comm[i].Comm_share);
    free(A_next);
    free(dimc_lis);
  }

  if (levels > 0)
    lastParamsCreate(&A[0].params, A[0].A_ptr, NCHILD, basis[1].dimS, basis[1].DimsLr, comm[0].Comm_merge, comm[0].Comm_share);
  else
    lastParamsCreate(&A[0].params, A[0].A_ptr, 1, basis[0].dimR, basis[0].Dims, comm[0].Comm_merge, comm[0].Comm_share);
}

void node_free(struct Node* node) {
  freeBufferedList(node->A_ptr, node->A_buf);
  free(node->A);
  int is_last = (node->lenA == 1) && (node->lenS == 0);
  if (is_last && node->params != NULL)
    lastParamsDestory(node->params);
  if (!is_last && node->params != NULL)
    batchParamsDestory(node->params);
}

void factorA_mov_mem(char dir, struct Node A[], const struct Base basis[], int64_t levels) {
  for (int64_t i = 0; i <= levels; i++) {
    int64_t stride = (basis[i].dimR + basis[i].dimS) * (basis[i].dimR + basis[i].dimS);
    flushBuffer(dir, A[i].A_ptr, A[i].A_buf, sizeof(double), stride * A[i].lenA);
  }
}

void factorA(struct Node A[], const struct Base basis[], const struct CellComm comm[], int64_t levels) {

  for (int64_t i = levels; i > 0; i--) {
    batchCholeskyFactor(A[i].params);
#ifdef _PROF
    int64_t ibegin = 0, iend = 0;
    self_local_range(&ibegin, &iend, &comm[i]);
    int64_t nnz = A[i].lenA;
    record_factor_flops(basis[i].dimR, basis[i].dimS, nnz, iend - ibegin);
#endif
  }
  chol_decomp(A[0].params);
#ifdef _PROF
  record_factor_flops(0, basis[0].dimR, 1, 1);
#endif
}

void allocRightHandSidesSV(struct RightHandSides rhs[], const struct Base base[], const struct CellComm comm[], int64_t levels) {
  for (int64_t l = levels; l >= 0; l--) {
    int64_t len = base[l].Ulen;
    int64_t len_arr = len * 4;
    struct Matrix* arr_m = (struct Matrix*)malloc(sizeof(struct Matrix) * len_arr);
    rhs[l].Xlen = len;
    rhs[l].X = arr_m;
    rhs[l].B = &arr_m[len];
    rhs[l].Xo = &arr_m[len * 2];
    rhs[l].Xc = &arr_m[len * 3];

    int64_t len_data = len * (base[l].dimN + base[l].dimR * 2);
    double* data = (double*)calloc(len_data, sizeof(double));
    for (int64_t i = 0; i < len; i++) {
      int64_t dim = base[l].Dims[i];
      int64_t diml = base[l].DimsLr[i];
      int64_t dimc = dim - diml;
      int64_t dimb = dimc;

      arr_m[i] = (struct Matrix) { &data[i * base[l].dimN], dim, 1, dim }; // X
      arr_m[i + len] = (struct Matrix) { &data[len * base[l].dimN + i * base[l].dimR], dimb, 1, dimb }; // B
      arr_m[i + len * 2] = (struct Matrix) { NULL, diml, 1, diml }; // Xo
      arr_m[i + len * 3] = (struct Matrix) { &data[len * (base[l].dimR + base[l].dimN) + i * base[l].dimR], dimc, 1, dimc }; // Xc
    }

    for (int64_t i = 0; i < len; i++) {
      int64_t child = std::get<0>(comm[l].LocalChild[i]);
      int64_t clen = std::get<1>(comm[l].LocalChild[i]);
      
      if (child >= 0 && l < levels) {
        int64_t row = 0;
        for (int64_t j = 0; j < clen; j++) {
          rhs[l + 1].Xo[child + j].A = &rhs[l].X[i].A[row];
          row = row + base[l + 1].DimsLr[child + j];
        }
      }
    }
  }
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
      int64_t dim = base[l].Dims[i];
      int64_t diml = base[l].DimsLr[i];
      int64_t dimc = diml;
      int64_t dimb = dim;

      arr_m[i] = (struct Matrix) { &data[i * base[l].dimN], dim, 1, dim }; // X
      arr_m[i + len] = (struct Matrix) { &data[len * base[l].dimN + i * base[l].dimN], dimb, 1, dimb }; // B
      arr_m[i + len * 2] = (struct Matrix) { NULL, diml, 1, diml }; // Xo
      arr_m[i + len * 3] = (struct Matrix) { NULL, dimc, 1, dimc }; // Xc
    }

    for (int64_t i = 0; i < len; i++) {
      int64_t child = std::get<0>(comm[l].LocalChild[i]);
      int64_t clen = std::get<1>(comm[l].LocalChild[i]);
      
      if (child >= 0 && l < levels) {
        int64_t row = 0;
        for (int64_t j = 0; j < clen; j++) {
          rhs[l + 1].Xo[child + j].A = &rhs[l].X[i].A[row];
          rhs[l + 1].Xc[child + j].A = &rhs[l].B[i].A[row];
          row = row + base[l + 1].DimsLr[child + j];
        }
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

void solveA(struct RightHandSides rhs[], const struct Node A[], const struct Base basis[], const struct CSC rels[], double* X, const struct CellComm comm[], int64_t levels) {
  int64_t lbegin = 0, lend = 0;
  self_local_range(&lbegin, &lend, &comm[levels]);
  int64_t row = 0;
  for (int64_t i = 0; i < (lend - lbegin); i++) {
    int64_t m = basis[levels].Dims[i + lbegin];
    memcpy(rhs[levels].X[lbegin + i].A, &X[row], m * sizeof(double));
    row = row + m;
  }

  for (int64_t i = levels; i > 0; i--) {
    int64_t xlen = rhs[i].Xlen;
    int64_t ibegin = 0, iend = 0;
    self_local_range(&ibegin, &iend, &comm[i]);

    level_merge_cpu(rhs[i].X[0].A, xlen * basis[i].dimN, &comm[i]);
    neighbor_reduce_cpu(rhs[i].X[0].A, basis[i].dimN, &comm[i]);
    dup_bcast_cpu(rhs[i].X[0].A, xlen * basis[i].dimN, &comm[i]);

    for (int64_t x = 0; x < rels[i].N; x++) {
      mmult('T', 'N', &basis[i].Uc[x + ibegin], &rhs[i].X[x + ibegin], &rhs[i].Xc[x + ibegin], 1., 1.);
      mmult('T', 'N', &basis[i].Uo[x + ibegin], &rhs[i].X[x + ibegin], &rhs[i].Xo[x + ibegin], 1., 1.);
      int64_t xx;
      lookupIJ(&xx, &rels[i], x + ibegin, x);
      memcpy(rhs[i].B[x + ibegin].A, rhs[i].Xc[x + ibegin].A, sizeof(double) * rhs[i].Xc[x + ibegin].M);
      mat_solve('F', &rhs[i].B[x + ibegin], &A[i].A_cc[xx]);

      for (int64_t yx = rels[i].ColIndex[x]; yx < rels[i].ColIndex[x + 1]; yx++) {
        int64_t y = rels[i].RowIndex[yx];
        if (y > x + ibegin)
          mmult('N', 'N', &A[i].A_cc[yx], &rhs[i].B[x + ibegin], &rhs[i].Xc[y], -1., 1.);
      }
    }

    neighbor_reduce_cpu(rhs[i].Xc[0].A, basis[i].dimR, &comm[i]);
    dup_bcast_cpu(rhs[i].Xc[0].A, xlen * basis[i].dimR, &comm[i]);

    for (int64_t x = 0; x < rels[i].N; x++) {
      int64_t xx;
      lookupIJ(&xx, &rels[i], x + ibegin, x);
      mat_solve('F', &rhs[i].Xc[x + ibegin], &A[i].A_cc[xx]);
      for (int64_t yx = rels[i].ColIndex[x]; yx < rels[i].ColIndex[x + 1]; yx++) {
        int64_t y = rels[i].RowIndex[yx];
        mmult('N', 'N', &A[i].A_oc[yx], &rhs[i].Xc[x + ibegin], &rhs[i].Xo[y], -1., 1.);
      }
    }
  }

  level_merge_cpu(rhs[0].X[0].A, basis[0].dimN, &comm[0]);
  dup_bcast_cpu(rhs[0].X[0].A, basis[0].dimN, &comm[0]);
  mat_solve('A', &rhs[0].X[0], &A[0].A[0]);
  
  for (int64_t i = 1; i <= levels; i++) {
    int64_t ibegin = 0, iend = 0;
    self_local_range(&ibegin, &iend, &comm[i]);
    for (int64_t x = 0; x < rels[i].N; x++) {
      for (int64_t yx = rels[i].ColIndex[x]; yx < rels[i].ColIndex[x + 1]; yx++) {
        int64_t y = rels[i].RowIndex[yx];
        mmult('T', 'N', &A[i].A_oc[yx], &rhs[i].Xo[y], &rhs[i].Xc[x + ibegin], -1., 1.);
      }
      int64_t xx;
      lookupIJ(&xx, &rels[i], x + ibegin, x);
      memcpy(rhs[i].B[x + ibegin].A, rhs[i].Xc[x + ibegin].A, sizeof(double) * rhs[i].Xc[x + ibegin].M);
      mat_solve('B', &rhs[i].B[x + ibegin], &A[i].A_cc[xx]);
    }

    int64_t xlen = rhs[i].Xlen;
    neighbor_bcast_cpu(rhs[i].B[0].A, basis[i].dimR, &comm[i]);
    dup_bcast_cpu(rhs[i].B[0].A, xlen * basis[i].dimR, &comm[i]);

    for (int64_t x = 0; x < rels[i].N; x++) {
      for (int64_t yx = rels[i].ColIndex[x]; yx < rels[i].ColIndex[x + 1]; yx++) {
        int64_t y = rels[i].RowIndex[yx];
        if (y > x + ibegin)
          mmult('T', 'N', &A[i].A_cc[yx], &rhs[i].B[y], &rhs[i].Xc[x + ibegin], -1., 1.);
      }

      int64_t xx;
      lookupIJ(&xx, &rels[i], x + ibegin, x);
      mat_solve('B', &rhs[i].Xc[x + ibegin], &A[i].A_cc[xx]);
      mmult('N', 'N', &basis[i].Uc[x + ibegin], &rhs[i].Xc[x + ibegin], &rhs[i].X[x + ibegin], 1., 0.);
      mmult('N', 'N', &basis[i].Uo[x + ibegin], &rhs[i].Xo[x + ibegin], &rhs[i].X[x + ibegin], 1., 1.);
    }

    neighbor_bcast_cpu(rhs[i].X[0].A, basis[i].dimN, &comm[i]);
    dup_bcast_cpu(rhs[i].X[0].A, xlen * basis[i].dimN, &comm[i]);
  }

  row = 0;
  for (int64_t i = 0; i < (lend - lbegin); i++) {
    int64_t m = basis[levels].Dims[i + lbegin];
    memcpy(&X[row], rhs[levels].X[lbegin + i].A, m * sizeof(double));
    row = row + m;
  }
}

void matVecA(struct RightHandSides rhs[], const struct Node A[], const struct Base basis[], const struct CSC rels_near[], const struct CSC rels_far[], double* X, const struct CellComm comm[], int64_t levels) {
  int64_t lbegin = 0, lend = 0;
  self_local_range(&lbegin, &lend, &comm[levels]);

  int64_t row = 0;
  for (int64_t i = 0; i < (lend - lbegin); i++) {
    int64_t m = basis[levels].Dims[i + lbegin];
    memcpy(rhs[levels].X[lbegin + i].A, &X[row], m * sizeof(double));
    row = row + m;
  }

  for (int64_t i = levels; i > 0; i--) {
    int64_t xlen = rhs[i].Xlen;
    level_merge_cpu(rhs[i].X[0].A, xlen * basis[i].dimN, &comm[i]);
    neighbor_bcast_cpu(rhs[i].X[0].A, basis[i].dimN, &comm[i]);
    dup_bcast_cpu(rhs[i].X[0].A, xlen * basis[i].dimN, &comm[i]);

    int64_t ibegin = 0, iend = 0;
    self_local_range(&ibegin, &iend, &comm[i]);
    int64_t iboxes = iend - ibegin;
    for (int64_t j = 0; j < iboxes; j++)
      mmult('T', 'N', &basis[i].Uo[j + ibegin], &rhs[i].X[j + ibegin], &rhs[i].Xo[j + ibegin], 1., 0.);
  }
  
  for (int64_t i = 1; i <= levels; i++) {
    int64_t ibegin = 0, iend = 0;
    self_local_range(&ibegin, &iend, &comm[i]);
    int64_t iboxes = iend - ibegin;
    for (int64_t y = 0; y < rels_far[i].N; y++)
      for (int64_t xy = rels_far[i].ColIndex[y]; xy < rels_far[i].ColIndex[y + 1]; xy++) {
        int64_t x = rels_far[i].RowIndex[xy];
        mmult('T', 'N', &A[i].S[xy], &rhs[i].Xo[x], &rhs[i].Xc[y + ibegin], 1., 1.);
      }
    for (int64_t j = 0; j < iboxes; j++)
      mmult('N', 'N', &basis[i].Uo[j + ibegin], &rhs[i].Xc[j + ibegin], &rhs[i].B[j + ibegin], 1., 0.);
  }

  for (int64_t y = 0; y < rels_near[levels].N; y++)
    for (int64_t xy = rels_near[levels].ColIndex[y]; xy < rels_near[levels].ColIndex[y + 1]; xy++) {
      int64_t x = rels_near[levels].RowIndex[xy];
      mmult('T', 'N', &A[levels].A[xy], &rhs[levels].X[x], &rhs[levels].B[y + lbegin], 1., 1.);
    }

  row = 0;
  for (int64_t i = 0; i < (lend - lbegin); i++) {
    int64_t m = basis[levels].Dims[i + lbegin];
    memcpy(&X[row], rhs[levels].B[lbegin + i].A, m * sizeof(double));
    row = row + m;
  }
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

