
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
};

struct RightHandSides {
  int64_t Xlen;
  std::vector<Matrix> X;
  std::vector<Matrix> Xc;
  std::vector<Matrix> Xo;
};

void allocNodes(Node* nodes, const Base B[], const CSC rels_near[], const CSC rels_far[], int64_t levels);

void deallocNode(Node* node, int64_t levels);

void node_mem(int64_t* bytes, const Node* node, int64_t levels);

void factorA(Node A[], const Base B[], const CSC rels_near[], const CSC rels_far[], int64_t levels);

void allocRightHandSides(RightHandSides st[], const Base base[], int64_t levels);

void deallocRightHandSides(RightHandSides* st, int64_t levels);

void RightHandSides_mem(int64_t* bytes, const RightHandSides* st, int64_t levels);

void solveA(RightHandSides st[], const Node A[], const Base B[], const CSC rels[], const Matrix* X, int64_t levels);

#ifdef __cplusplus
}
#endif
