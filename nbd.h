
#pragma once

#include "mpi.h"
#include "stdint.h"
#include "stddef.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Body {
  double X[3];
  double B;
};

struct Matrix {
  double* A;
  int64_t M;
  int64_t N;
};

struct Cell {
  int64_t CHILD;
  int64_t BODY[2];
  double R[3];
  double C[3];

  int64_t LEVEL;
  int64_t Procs[2];
  
  int64_t lenMultipole;
  int64_t* Multipole;
};

struct CSC {
  int64_t M;
  int64_t N;
  int64_t* COL_INDEX;
  int64_t* ROW_INDEX;
};

struct CellComm {
  struct CSC Comms; // P by P sparse. Entry (i, j) proc i comm with j.
  int64_t *ProcRootI, *ProcBoxes, *ProcBoxesEnd, Proc[2]; // len P. The PRI[i]'th element in comms is Entry (i, i) on row/column i.
  // len P. Proc i hold boxes within [PB[i], PBEnd[i]) as LET.
  // len P. Proc i hold boxes within [PB[i], PBEnd[i]) as LET.

  // Local Proc num used by searching. Same content stored in range [P0, P1).
  MPI_Comm Comm_merge, *Comm_box; // len P. CB[i] is used when Proc i broadcasts.
  // Local comm for merged procs.
};

struct Base {
  int64_t Ulen, *Lchild, *Dims, *DimsLr, *Offsets, *Multipoles;
  struct Matrix *Uo, *Uc, *R;
};

struct Node {
  int64_t lenA, lenS;
  struct Matrix *A, *S, *A_cc, *A_oc, *A_oo;
};

struct RightHandSides {
  int64_t Xlen;
  struct Matrix *X, *Xc, *Xo;
};

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

void matrixCreate(struct Matrix* mat, int64_t m, int64_t n);

void matrixDestroy(struct Matrix* mat);

void cpyFromMatrix(const struct Matrix* A, double* v);

void maxpby(struct Matrix* A, const double* v, double alpha, double beta);

void cpyMatToMat(int64_t m, int64_t n, const struct Matrix* m1, struct Matrix* m2, int64_t y1, int64_t x1, int64_t y2, int64_t x2);

void qr_full(struct Matrix* Q, struct Matrix* R);

void updateSubU(struct Matrix* U, const struct Matrix* R1, const struct Matrix* R2);

void lraID(double epi, struct Matrix* A, struct Matrix* U, int32_t arows[], int64_t* rnk_out);

void zeroMatrix(struct Matrix* A);

void mmult(char ta, char tb, const struct Matrix* A, const struct Matrix* B, struct Matrix* C, double alpha, double beta);

void chol_decomp(struct Matrix* A);

void trsm_lowerA(struct Matrix* A, const struct Matrix* L);

void utav(const struct Matrix* U, const struct Matrix* A, const struct Matrix* VT, struct Matrix* C);

void rsr(const struct Matrix* R1, const struct Matrix* R2, struct Matrix* S);

void mat_solve(char type, struct Matrix* X, const struct Matrix* A);

void normalizeA(struct Matrix* A, const struct Matrix* B);

void mnrm2(const struct Matrix* A, double* nrm);

void matrix_mem(int64_t* bytes, const struct Matrix* A, int64_t lenA);

void buildTree(int64_t* ncells, struct Cell* cells, struct Body* bodies, int64_t nbodies, int64_t levels);

void traverse(char NoF, struct CSC* rels, int64_t ncells, const struct Cell* cells, double theta);

void get_level(int64_t* begin, int64_t* end, const struct Cell* cells, int64_t level, int64_t mpi_rank);

void buildComm(struct CellComm* comms, int64_t ncells, const struct Cell* cells, const struct CSC* cellFar, const struct CSC* cellNear, int64_t levels);

void cellComm_free(struct CellComm* comms, int64_t levels);

void lookupIJ(int64_t* ij, const struct CSC* rels, int64_t i, int64_t j);

void i_local(int64_t* ilocal, const struct CellComm* comm);

void i_global(int64_t* iglobal, const struct CellComm* comm);

void self_local_range(int64_t* ibegin, int64_t* iend, const struct CellComm* comm);

void content_length(int64_t* len, const struct CellComm* comm);

void relations(struct CSC rels[], int64_t ncells, const struct Cell* cells, const struct CSC* cellRel, int64_t levels);

void allocBasis(struct Base* basis, int64_t levels, int64_t ncells, const struct Cell* cells, const struct CellComm* comm);

void deallocBasis(struct Base* basis, int64_t levels);

void basis_mem(int64_t* bytes, const struct Base* basis, int64_t levels);

void evaluateBaseAll(void(*ef)(double*), struct Base basis[], int64_t ncells, struct Cell* cells, const struct CSC* rel_near, int64_t levels, const struct CellComm* comm, const struct Body* bodies, int64_t nbodies, double epi, int64_t mrank, int64_t sp_pts);

void evaluate(char NoF, struct Matrix* d, void(*ef)(double*), int64_t ncells, const struct Cell* cells, const struct Body* bodies, const struct CSC* csc, int64_t level);

void solveRelErr(double* err_out, const struct Matrix* X, const struct Matrix* ref, const struct CellComm* comm);

void allocNodes(struct Node* nodes, const struct Base B[], const struct CSC rels_near[], const struct CSC rels_far[], int64_t levels);

void deallocNode(struct Node* node, int64_t levels);

void node_mem(int64_t* bytes, const struct Node* node, int64_t levels);

void factorA(struct Node A[], const struct Base B[], const struct CSC rels_near[], const struct CSC rels_far[], int64_t levels);

void allocRightHandSides(struct RightHandSides st[], const struct Base base[], int64_t levels);

void deallocRightHandSides(struct RightHandSides* st, int64_t levels);

void RightHandSides_mem(int64_t* bytes, const struct RightHandSides* st, int64_t levels);

void solveA(struct RightHandSides st[], const struct Node A[], const struct Base B[], const struct CSC rels[], const struct Matrix* X, int64_t levels);

#ifdef __cplusplus
}
#endif

