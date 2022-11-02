
#pragma once

#include "mpi.h"
#include "stdint.h"
#include "stddef.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Matrix { double* A; int64_t M, N; };

void mat_cpy_batch(int64_t m, int64_t n, const struct Matrix* m1, struct Matrix* m2, int64_t y1, int64_t x1, int64_t y2, int64_t x2);
void mat_cpy_flush();

void mmult(char ta, char tb, const struct Matrix* A, const struct Matrix* B, struct Matrix* C, double alpha, double beta);
void mmult_batch(char ta, char tb, const struct Matrix* A, const struct Matrix* B, struct Matrix* C, double alpha, double beta);
void mmult_flush();

void chol_decomp(struct Matrix* A);
void icmp_chol_decomp_batch(struct Matrix* A_cc, struct Matrix* A_oc, struct Matrix* A_oo);
void icmp_chol_decomp_flush();

void trsm_lowerA_batch(struct Matrix* A, const struct Matrix* L);
void trsm_lowerA_flush();

void svd_U(struct Matrix* A, struct Matrix* U, double* S);

void id_row_batch(struct Matrix* A, int32_t arows[], double* work);
void id_row_flush();

void upper_tri_reflec_mult(char side, const struct Matrix* R, struct Matrix* A);
void qr_full(struct Matrix* Q, struct Matrix* R, double* tau);

void mat_solve(char type, struct Matrix* X, const struct Matrix* A);

void nrm2_A(struct Matrix* A, double* nrm);
void scal_A(struct Matrix* A, double alpha);

void init_batch_lib();
void finalize_batch_lib();
void sync_batch_lib();

void alloc_matrices_aligned(double** A_ptr, int* M_align, int M, int N, int count);
void free_matrices(double* A_ptr);
void copy_basis(char dir, const double* Ur_in, const double* Us_in, double* U_out, int IR_dim, int IS_dim, int OR_dim, int OS_dim, int ldu_in, int ldu_out);
void copy_mat(char dir, const double* A_in, double* A_out, int M_in, int N_in, int lda_in, int M_out, int N_out, int lda_out);
void compute_rs_splits_left(const double* U_ptr, const double* A_ptr, double* out_ptr, const int* row_A, int N, int N_align, int A_count);
void compute_rs_splits_right(const double* V_ptr, const double* A_ptr, double* out_ptr, const int* col_A, int N, int N_align, int A_count);
void factor_diag(int N_diag, double* A_ptr, double* U_ptr, int R_dim, int S_dim, int N_align);
void schur_diag(int N_diag, double* A_ptr, const int* diag_idx, int R_dim, int S_dim, int N_align);

struct Body { double X[3], B; };
struct Cell { int64_t Child, Body[2], Level, Procs[2]; double R[3], C[3]; };
struct CSC { int64_t M, N, *ColIndex, *RowIndex; };
struct CellComm { struct CSC Comms; int64_t Proc[3], *ProcRootI, *ProcBoxes, *ProcBoxesEnd; MPI_Comm Comm_share, Comm_merge, *Comm_box; };
struct Base { int64_t Ulen, *Lchild, *Dims, *DimsLr, *Offsets, *Multipoles; struct Matrix *Uo, *Uc, *R; };
struct Node { int64_t lenA, lenS; struct Matrix *A, *S, *A_cc, *A_oc, *A_oo; };
struct RightHandSides { int64_t Xlen; struct Matrix *X, *XcM, *XoL, *B; };

void laplace3d(double* r2);

void yukawa3d(double* r2);

void set_kernel_constants(double singularity, double alpha);

void gen_matrix(void(*ef)(double*), int64_t m, int64_t n, const struct Body* bi, const struct Body* bj, double Aij[], const int64_t sel_i[], const int64_t sel_j[]);

void uniform_unit_cube(struct Body* bodies, int64_t nbodies, int64_t dim, unsigned int seed);

void mesh_unit_sphere(struct Body* bodies, int64_t nbodies);

void mesh_unit_cube(struct Body* bodies, int64_t nbodies);

void magnify_reloc(struct Body* bodies, int64_t nbodies, const double Ccur[], const double Cnew[], const double R[]);

void body_neutral_charge(struct Body* bodies, int64_t nbodies, double cmax, unsigned int seed);

void get_bounds(const struct Body* bodies, int64_t nbodies, double R[], double C[]);

void sort_bodies(struct Body* bodies, int64_t nbodies, int64_t sdim);

void read_sorted_bodies(int64_t* nbodies, int64_t lbuckets, struct Body* bodies, int64_t buckets[], const char* fname);

void mat_vec_reference(void(*ef)(double*), int64_t begin, int64_t end, double B[], int64_t nbodies, const struct Body* bodies);

void buildTree(int64_t* ncells, struct Cell* cells, struct Body* bodies, int64_t nbodies, int64_t levels);

void buildTreeBuckets(struct Cell* cells, const struct Body* bodies, const int64_t buckets[], int64_t levels);

void traverse(char NoF, struct CSC* rels, int64_t ncells, const struct Cell* cells, double theta);

void csc_free(struct CSC* csc);

void get_level(int64_t* begin, int64_t* end, const struct Cell* cells, int64_t level, int64_t mpi_rank);

void buildComm(struct CellComm* comms, int64_t ncells, const struct Cell* cells, const struct CSC* cellFar, const struct CSC* cellNear, int64_t levels);

void cellComm_free(struct CellComm* comm);

void lookupIJ(int64_t* ij, const struct CSC* rels, int64_t i, int64_t j);

void i_local(int64_t* ilocal, const struct CellComm* comm);

void i_global(int64_t* iglobal, const struct CellComm* comm);

void self_local_range(int64_t* ibegin, int64_t* iend, const struct CellComm* comm);

void content_length(int64_t* len, const struct CellComm* comm);

void local_bodies(int64_t body[], int64_t ncells, const struct Cell cells[], int64_t levels);

void loadX(double* X, int64_t body[], const struct Body* bodies);

void relations(struct CSC rels[], int64_t ncells, const struct Cell* cells, const struct CSC* cellRel, int64_t levels);

void evalD(void(*ef)(double*), struct Matrix* D, int64_t ncells, const struct Cell* cells, const struct Body* bodies, const struct CSC* csc, int64_t level);

void buildBasis(void(*ef)(double*), struct Base basis[], int64_t ncells, struct Cell* cells, const struct CSC* rel_near, int64_t levels, const struct CellComm* comm, const struct Body* bodies, int64_t nbodies, double epi, int64_t mrank, int64_t sp_pts);

void basis_free(struct Base* basis);

void evalS(void(*ef)(double*), struct Matrix* S, const struct Base* basis, const struct Body* bodies, const struct CSC* rels, const struct CellComm* comm);

void allocNodes(struct Node A[], const struct Base basis[], const struct CSC rels_near[], const struct CSC rels_far[], const struct CellComm comm[], int64_t levels);

void node_free(struct Node* node);

void factorA(struct Node A[], const struct Base B[], const struct CSC rels_near[], const struct CSC rels_far[], const struct CellComm comm[], int64_t levels);

void allocRightHandSides(char mvsv, struct RightHandSides st[], const struct Base base[], int64_t levels);

void rightHandSides_free(struct RightHandSides* rhs);

void solveA(struct RightHandSides st[], const struct Node A[], const struct Base B[], const struct CSC rels[], double* X, const struct CellComm comm[], int64_t levels);

void matVecA(struct RightHandSides rhs[], const struct Node A[], const struct Base basis[], const struct CSC rels_near[], const struct CSC rels_far[], double* X, const struct CellComm comm[], int64_t levels);

void solveRelErr(double* err_out, const double* X, const double* ref, int64_t lenX);

#ifdef __cplusplus
}
#endif

