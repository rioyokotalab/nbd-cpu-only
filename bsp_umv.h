

#include "nbd.h"

namespace nbd {

  struct Node {
    int M;
    int N;
    Matrices A;

    Matrices A_cc;
    Matrices A_oc;
    Matrices A_co;
    Matrices A_oo;
  };

  struct Base {
    Matrices Uo;
    Matrices Uc;
  };

  Node node(EvalFunc ef, int dim, const Cell* i, const Cell* j);

  // C = A * B^-1;
  void a_inv_b(const Matrix& A, const Matrix& B, Matrix& C);

  int get_rank(real_t repi, const Matrix& A);

  void F_ABBA(const Matrix& A, const Matrix& B, Matrix& F);

  int orth_base(real_t repi, const Matrix& A, Matrix& Us, Matrix& Uc);

  Base base_i(real_t repi, const Node& H);

  // C = UT * A * VT;
  void utav(const Matrix& U, const Matrix& A, const Matrix& VT, Matrix& C);

  void split_A(Node& H, const Base& U, const Base& V);

  void factor_A(Node& H);

  std::vector<real_t*> base_fw(const Base& U, real_t* x);

  void base_bk(const Base& U, real_t* x);

  void A_fw(const Node& H, std::vector<real_t*>& x);

  void A_bk(const Node& H, std::vector<real_t*>& x);

  Node merge_H(const Node& H);

  Matrix merge_D(const Node& H);

  void solve_D(Matrix& D, real_t* x);

  void h2_solve_complete(real_t repi, Node& H, real_t* x);


};