

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
    Matrices F;

    Node (EvalFunc ef, int dim, const Cell* i, const Cell* j);
    Node (const Node& H);
  };

  struct Base {
    Matrices Uo;
    Matrices Uc;

    Base (real_t repi, const Node& H);
  };

  // C = A * B^-1;
  void a_inv_b(const Matrix& A, const Matrix& B, Matrix& C);

  int get_rank(real_t repi, const Matrix& A);

  void F_ABBA(const Matrix& A, const Matrix& B, Matrix& F);

  int orth_base(real_t repi, const Matrix& A, Matrix& Us, Matrix& Uc);

  // C = UT * A * VT;
  void utav(const Matrix& U, const Matrix& A, const Matrix& VT, Matrix& F, Matrix& C);

  void utfv(const Matrix& U, const Matrix& VT, Matrix& F);

  void split_A(Node& H, const Base& U, const Base& V);

  void factor_A(Node& H);

  std::vector<real_t*> base_fw(const Base& U, real_t* x);

  void base_bk(const Base& U, real_t* x);

  void A_fw(const Node& H, std::vector<real_t*>& x);

  void A_bk(const Node& H, std::vector<real_t*>& x);

  void merge_D(const Node& H, Matrix& D, std::vector<int>& ipiv);

  void solve_D(const Matrix& D, const int* ipiv, real_t* x);

  void h2_factor(real_t repi, std::vector<Node>& H, std::vector<Base>& B, Matrix& last, std::vector<int>& ipiv);

  void h2_solve(int lvl, const Node* H, const Base* B, const Matrix& last, const int* ipiv, real_t* x);

  void h2_solve_complete(real_t repi, EvalFunc ef, int dim, const Cell* i, const Cell* j, real_t* x);


};