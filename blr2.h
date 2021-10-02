

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

  // C = A^-1 * B or A * B^-1;
  int a_inv_b(real_t repi, bool inv_A, const Matrix& A, const Matrix& B, Matrix& C);

  void orth_base(Matrix& Us, Matrix& Uc);

  Base base_i(real_t repi, int p, const Node& H);

  // C = UT * A * VT;
  void utav(const Matrix& U, const Matrix& A, const Matrix& VT, Matrix& C);

  void split_A(Node& H, const Base& U, const Base& V);

  void factor_A(Node& H);

  std::vector<real_t*> base_fw(const Base& U, real_t* x);

  void base_bk(const Base& U, real_t* x);

  void A_fw(const Node& H, std::vector<real_t*>& x);

  void A_bk(const Node& H, std::vector<real_t*>& x);

  Matrix merge_D(const Node& H);

  void solve_D(Matrix& D, real_t* x);


};