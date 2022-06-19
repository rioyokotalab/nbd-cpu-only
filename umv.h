
#pragma once

#include "basis.h"

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

struct SpDense {
  int64_t Levels;
  std::vector<Node> D;
  std::vector<Base> Basis;
  std::vector<CSC> RelsNear;
  std::vector<CSC> RelsFar;
};

void splitA(Matrix* A_out, const CSC& rels, const Matrix* A, const Matrix* U, const Matrix* V, int64_t level);

void splitS(Matrix* S_out, const CSC& rels, const Matrix* S, const Matrix* U, const Matrix* V, int64_t level);

void factorAcc(Matrix* A_cc, const CSC& rels, int64_t level);

void factorAoc(Matrix* A_oc, const Matrix* A_cc, const CSC& rels, int64_t level);

void schurCmplm(Matrix* S, const Matrix* A_oc, const CSC& rels, int64_t level);

void allocNodes(Node* nodes, const CSC rels_near[], const CSC rels_far[], int64_t levels);

void deallocNode(Node* node, int64_t levels);

void node_mem(int64_t* bytes, const Node* node, int64_t levels);

void allocA(Matrix* A, const CSC& rels, const int64_t dims[], int64_t level);

void allocSubMatrices(Node& n, const CSC& rels, const int64_t dims[], const int64_t diml[], int64_t level);

void factorNode(Node& n, const Base& basis, const CSC& rels_near, const CSC& rels_far, int64_t level);

void nextNode(Node& Anext, const CSC& rels_up, const Node& Aprev, const CSC& rels_low, int64_t nlevel);

void factorA(Node A[], const Base B[], const CSC rels[], int64_t levels);

void allocSpDense(SpDense& sp, const Cell* cells, int64_t levels);

void deallocSpDense(SpDense* sp);

void factorSpDense(SpDense& sp);

#ifdef __cplusplus
}
#endif
