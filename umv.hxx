
#pragma once

#include "basis.hxx"

namespace nbd {

  struct Node {
    Matrices A;
    Matrices S;
    Matrices A_cc;
    Matrices A_oc;
    Matrices A_oo;
    Matrices S_oo;
  };

  typedef std::vector<Node> Nodes;

  void splitA(Matrices& A_out, const CSC& rels, const Matrices& A, const Matrices& U, const Matrices& V, int64_t level);

  void splitS(Matrices& S_out, const CSC& rels, const Matrices& S, const Matrices& U, const Matrices& V, int64_t level);

  void factorAcc(Matrices& A_cc, const CSC& rels);

  void factorAoc(Matrices& A_oc, const Matrices& A_cc, const CSC& rels);

  void schurCmplm(Matrices& S, const Matrices& A_oc, const CSC& rels);

  void allocNodes(Nodes& nodes, const CSC rels[], int64_t levels);

  void allocA(Matrices& A, const CSC& rels, const int64_t dims[], int64_t level);

  void allocS(Matrices& S, const CSC& rels, const int64_t diml[], int64_t level);

  void allocSubMatrices(Node& n, const CSC& rels, const int64_t dims[], const int64_t dimo[], int64_t level);

  void factorNode(Node& n, Base& basis, const CSC& rels, int64_t level);

  void nextNode(Node& Anext, Base& bsnext, const CSC& rels_up, const Node& Aprev, const Base& bsprev, const CSC& rels_low, int64_t nlevel);

  void factorA(Node A[], Base B[], const CSC rels[], int64_t levels);

};
