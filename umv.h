
#pragma once

#include "build_tree.h"
#include "linalg.h"
#include "kernel.h"
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

struct Node {
  int64_t lenA;
  int64_t lenS;
  std::vector<Matrix> A;
  std::vector<Matrix> S;
  std::vector<Matrix> A_cc;
  std::vector<Matrix> A_oc;
  std::vector<Matrix> A_oo;
  std::vector<Matrix> S_oo;
};

struct RightHandSides {
  int64_t Xlen;
  std::vector<Matrix> X;
  std::vector<Matrix> Xc;
  std::vector<Matrix> Xo;
};

struct SpDense {
  int64_t Levels;
  std::vector<Node> D;
  std::vector<Base> Basis;
  std::vector<CSC> RelsNear;
  std::vector<CSC> RelsFar;
};

void allocNodes(Node* nodes, const CSC rels_near[], const CSC rels_far[], int64_t levels);

void deallocNode(Node* node, int64_t levels);

void node_mem(int64_t* bytes, const Node* node, int64_t levels);

void allocA(Matrix* A, const CSC& rels, const int64_t dims[], int64_t level);

void factorNode(Node& n, const Base& basis, const CSC& rels_near, const CSC& rels_far, int64_t level);

void factorA(Node A[], const Base B[], const CSC rels[], int64_t levels);

void allocSpDense(SpDense& sp, int64_t levels);

void deallocSpDense(SpDense* sp);

void factorSpDense(SpDense& sp);

void basisXoc(char fwbk, RightHandSides& vx, const Base& basis, int64_t level);

void svAccFw(Matrix* Xc, const Matrix* A_cc, const CSC& rels, int64_t level);

void svAccBk(Matrix* Xc, const Matrix* A_cc, const CSC& rels, int64_t level);

void svAocFw(Matrix* Xo, const Matrix* Xc, const Matrix* A_oc, const CSC& rels, int64_t level);

void svAocBk(Matrix* Xc, const Matrix* Xo, const Matrix* A_oc, const CSC& rels, int64_t level);

void allocRightHandSides(RightHandSides st[], const Base base[], int64_t levels);

void deallocRightHandSides(RightHandSides* st, int64_t levels);

void RightHandSides_mem(int64_t* bytes, const RightHandSides* st, int64_t levels);

void solveA(RightHandSides st[], const Node A[], const Base B[], const CSC rels[], const Matrix* X, int64_t levels);

void solveSpDense(RightHandSides st[], const SpDense& sp, const Matrix* X);

void solveRelErr(double* err_out, const Matrix* X, const Matrix* ref, int64_t level);

#ifdef __cplusplus
}
#endif
