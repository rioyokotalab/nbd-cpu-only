
#pragma once

#include "mpi.h"
#include "cuda_runtime_api.h"
#include "nccl.h"

#include <vector>
#include <utility>
#include <cstdint>
#include <cstddef>

#include "comm.hxx"

struct Matrix { double* A; int64_t M, N, LDA; };

void mat_cpy(int64_t m, int64_t n, const struct Matrix* m1, struct Matrix* m2, int64_t y1, int64_t x1, int64_t y2, int64_t x2);

void mmult(char ta, char tb, const struct Matrix* A, const struct Matrix* B, struct Matrix* C, double alpha, double beta);

void svd_U(struct Matrix* A, double* S);

void id_row(struct Matrix* U, struct Matrix* A, int32_t arows[]);

void upper_tri_reflec_mult(char side, int64_t lenR, const struct Matrix* R, struct Matrix* A);
void qr_full(struct Matrix* Q, struct Matrix* R);

void mat_solve(char type, struct Matrix* X, const struct Matrix* A);

void nrm2_A(struct Matrix* A, double* nrm);
void scal_A(struct Matrix* A, double alpha);

int init_libs(int* argc, char*** argv);
void fin_libs();
void set_work_size(int64_t Lwork, double** D_DATA, int64_t* D_DATA_SIZE);

void batchParamsCreate(void** params, int64_t R_dim, int64_t S_dim, const double* U_ptr, double* A_ptr, double* X_ptr, int64_t N_up, double** A_up, double** X_up,
  double* Workspace, int64_t Lwork, int64_t N_rows, int64_t N_cols, int64_t col_offset, const int64_t row_A[], const int64_t col_A[]);
void batchParamsDestory(void* params);

void lastParamsCreate(void** params, double* A, double* X, int64_t N);
void lastParamsDestory(void* params);

void allocBufferedList(void** A_ptr, void** A_buffer, int64_t element_size, int64_t count);
void flushBuffer(char dir, void* A_ptr, void* A_buffer, int64_t element_size, int64_t count);
void freeBufferedList(void* A_ptr, void* A_buffer);

void batchCholeskyFactor(void* params, const struct CellComm* comm);
void batchForwardULV(void* params_ptr, const struct CellComm* comm);
void batchBackwardULV(void* params_ptr, const struct CellComm* comm);
void chol_decomp(void* params, const struct CellComm* comm);
void chol_solve(void* params_ptr, const struct CellComm* comm);

struct Cell { int64_t Child[2], Body[2], Level, Procs[2]; double R[3], C[3]; };
struct CellBasis { int64_t M, N, *Multipoles; double *Uo, *Uc, *R; };
struct CSC { int64_t M, N, *ColIndex, *RowIndex; };

struct Base { 
  int64_t Ulen, *Dims, *DimsLr, dimR, dimS, dimN, padN, **Multipoles;
  struct Matrix *Uo, *Uc, *R;
  double *U_gpu, *U_cpu, *R_gpu, *R_cpu; 
};

struct Node { int64_t lenA, lenS; struct Matrix *A, *S, *A_cc, *A_oc, *A_oo; double* A_ptr, *A_buf, *X_ptr, *X_buf; void* params; };
struct RightHandSides { int64_t Xlen; struct Matrix *X, *Xc, *Xo, *B; };

void laplace3d(double* r2);

void yukawa3d(double* r2);

void set_kernel_constants(double singularity, double alpha);

void gen_matrix(void(*ef)(double*), int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda, const int64_t sel_i[], const int64_t sel_j[]);

void uniform_unit_cube(double* bodies, int64_t nbodies, int64_t dim, unsigned int seed);

void mesh_unit_sphere(double* bodies, int64_t nbodies);

void mesh_unit_cube(double* bodies, int64_t nbodies);

void magnify_reloc(double* bodies, int64_t nbodies, const double Ccur[], const double Cnew[], const double R[], double alpha);

void body_neutral_charge(double X[], int64_t nbodies, double cmax, unsigned int seed);

void get_bounds(const double* bodies, int64_t nbodies, double R[], double C[]);

void sort_bodies(double* bodies, int64_t nbodies, int64_t sdim);

void read_sorted_bodies(int64_t* nbodies, int64_t lbuckets, double* bodies, int64_t buckets[], const char* fname);

void mat_vec_reference(void(*ef)(double*), int64_t begin, int64_t end, double B[], int64_t nbodies, const double* bodies, const double Xbodies[]);

void buildTree(int64_t* ncells, struct Cell* cells, double* bodies, int64_t nbodies, int64_t levels);

void buildTreeBuckets(struct Cell* cells, const double* bodies, const int64_t buckets[], int64_t levels);

void traverse(char NoF, struct CSC* rels, int64_t ncells, const struct Cell* cells, double theta);

void csc_free(struct CSC* csc);

void get_level(int64_t* begin, int64_t* end, const struct Cell* cells, int64_t level, int64_t mpi_rank);

void buildCellBasis(double epi, int64_t mrank, int64_t sp_pts, void(*ef)(double*), struct CellBasis* basis, int64_t ncells, const struct Cell* cells, 
  int64_t nbodies, const double* bodies, const struct CSC* rels, int64_t levels);

void cellBasis_free(struct CellBasis* basis);

void lookupIJ(int64_t* ij, const struct CSC* rels, int64_t i, int64_t j);

void local_bodies(int64_t body[], int64_t ncells, const struct Cell cells[], int64_t levels);

void loadX(double* X, int64_t body[], const double Xbodies[]);

void relations(struct CSC rels[], int64_t ncells, const struct Cell* cells, const struct CSC* cellRel, int64_t levels, const struct CellComm* comm);

void evalD(void(*ef)(double*), struct Matrix* D, int64_t ncells, const struct Cell* cells, const double* bodies, const struct CSC* csc, int64_t level);

void buildBasis(int alignment, struct Base basis[], int64_t ncells, struct Cell* cells, struct CellBasis* cell_basis, int64_t levels, const struct CellComm* comm);

void basis_free(struct Base* basis);

void evalS(void(*ef)(double*), struct Matrix* S, const struct Base* basis, const double* bodies, const struct CSC* rels, const struct CellComm* comm);

void allocNodes(struct Node A[], double** Workspace, int64_t* Lwork, const struct Base basis[], const struct CSC rels_near[], const struct CSC rels_far[], const struct CellComm comm[], int64_t levels);

void node_free(struct Node* node);

void factorA_mov_mem(char dir, struct Node A[], const struct Base basis[], int64_t levels);

void factorA(struct Node A[], const struct Base B[], const struct CellComm comm[], int64_t levels);

void allocRightHandSidesSV(struct RightHandSides st[], const struct Base base[], const struct CellComm comm[], int64_t levels);

void allocRightHandSidesMV(struct RightHandSides st[], const struct Base base[], const struct CellComm comm[], int64_t levels);

void rightHandSides_free(struct RightHandSides* rhs);

void solveA(struct RightHandSides st[], struct Node A[], const struct Base B[], const struct CSC rels[], double* X, const struct CellComm comm[], int64_t levels);

void matVecA(struct RightHandSides rhs[], const struct Node A[], const struct Base basis[], const struct CSC rels_near[], const struct CSC rels_far[], double* X, const struct CellComm comm[], int64_t levels);

void solveRelErr(double* err_out, const double* X, const double* ref, int64_t lenX);


