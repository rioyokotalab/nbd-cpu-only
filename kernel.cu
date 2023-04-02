
#include "kernel.hxx"

#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

BodiesIter::BodiesIter(int64_t m, const double* bi, const double* bj) : M(m), Bodies_i(bi), Bodies_j(bj) {}

__host__ __device__ thrust::pair<const double*, const double*> BodiesIter::operator()(int64_t i) const {
  int64_t x = i / M;
  int64_t y = i - x * M;
  return thrust::make_pair(&Bodies_i[y * 3], &Bodies_j[x * 3]);
}

ArrIter::ArrIter(int64_t m, int64_t lda) : M(m), LDA_M_diff(lda - m) {}

__host__ __device__ int64_t ArrIter::operator()(int64_t x) const
{ return x + (x / M) * LDA_M_diff; }

thrust::transform_iterator<BodiesIter, thrust::counting_iterator<int64_t>>
  createBodiesIter(int64_t m, const double* bi, const double* bj) {
  return thrust::make_transform_iterator(thrust::make_counting_iterator<int64_t>(0), BodiesIter(m, bi, bj));
}

thrust::permutation_iterator<double*, thrust::transform_iterator<ArrIter, thrust::counting_iterator<int64_t>>>
  createArrIter(int64_t m, double A[], int64_t lda) {
  return thrust::make_permutation_iterator(A, thrust::make_transform_iterator(thrust::make_counting_iterator<int64_t>(0), ArrIter(m, lda)));
}

__host__ __device__ double computeR2(const double bi[], const double bj[]) {
  double dX = bi[0] - bj[0];
  double dY = bi[1] - bj[1];
  double dZ = bi[2] - bj[2];
  return dX * dX + dY * dY + dZ * dZ;
}

__host__ __device__ double Laplace3D::operator()(thrust::pair<const double*, const double*> x) const {
  double r2 = computeR2(thrust::get<0>(x), thrust::get<1>(x));
  return r2 == 0. ? singularity : (1. / sqrt(r2));
}

void Laplace3D::genMatrixHost(int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda) const {
  auto id_iter = createBodiesIter(m, bi, bj);
  if (m < lda) {
    auto A_iter = createArrIter(m, Aij, lda);
    thrust::transform(thrust::host, id_iter, id_iter + m * n, A_iter, *this);
  }
  else
    thrust::transform(thrust::host, id_iter, id_iter + m * n, Aij, *this);
}

void Laplace3D::genMatrixDevice(int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda, cudaStream_t stream) const {
  auto id_iter = createBodiesIter(m, bi, bj);
  if (m < lda) {
    auto A_iter = createArrIter(m, Aij, lda);
    thrust::transform(thrust::cuda::par.on(stream), id_iter, id_iter + m * n, A_iter, *this);
  }
  else
    thrust::transform(thrust::cuda::par.on(stream), id_iter, id_iter + m * n, Aij, *this);
}

__host__ __device__ double Yukawa3D::operator()(thrust::pair<const double*, const double*> x) const {
  double r2 = computeR2(thrust::get<0>(x), thrust::get<1>(x));
  if (r2 == 0.)
    return singularity;
  else {
    double r = sqrt(r2);
    return exp(alpha * -r) / r;
  }
}

void Yukawa3D::genMatrixHost(int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda) const {
  auto id_iter = createBodiesIter(m, bi, bj);
  if (m < lda) {
    auto A_iter = createArrIter(m, Aij, lda);
    thrust::transform(thrust::host, id_iter, id_iter + m * n, A_iter, *this);
  }
  else
    thrust::transform(thrust::host, id_iter, id_iter + m * n, Aij, *this);
}

void Yukawa3D::genMatrixDevice(int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda, cudaStream_t stream) const {
  auto id_iter = createBodiesIter(m, bi, bj);
  if (m < lda) {
    auto A_iter = createArrIter(m, Aij, lda);
    thrust::transform(thrust::cuda::par.on(stream), id_iter, id_iter + m * n, A_iter, *this);
  }
  else
    thrust::transform(thrust::cuda::par.on(stream), id_iter, id_iter + m * n, Aij, *this);
}

__host__ __device__ double Gaussian::operator()(thrust::pair<const double*, const double*> x) const {
  double r2 = computeR2(thrust::get<0>(x), thrust::get<1>(x));
  return exp(-r2 * alpha);
}

void Gaussian::genMatrixHost(int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda) const {
  auto id_iter = createBodiesIter(m, bi, bj);
  if (m < lda) {
    auto A_iter = createArrIter(m, Aij, lda);
    thrust::transform(thrust::host, id_iter, id_iter + m * n, A_iter, *this);
  }
  else
    thrust::transform(thrust::host, id_iter, id_iter + m * n, Aij, *this);
}

void Gaussian::genMatrixDevice(int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda, cudaStream_t stream) const {
  auto id_iter = createBodiesIter(m, bi, bj);
  if (m < lda) {
    auto A_iter = createArrIter(m, Aij, lda);
    thrust::transform(thrust::cuda::par.on(stream), id_iter, id_iter + m * n, A_iter, *this);
  }
  else
    thrust::transform(thrust::cuda::par.on(stream), id_iter, id_iter + m * n, Aij, *this);
}

void gen_matrix(const EvalDouble& Eval, int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda) {
  Eval.genMatrixHost(m, n, bi, bj, Aij, lda);
  /*thrust::device_vector<double> bi_dev(m * 3), bj_dev(n * 3), A_dev(lda * n);
  cudaMemcpy(thrust::raw_pointer_cast(&bi_dev[0]), bi, sizeof(double) * m * 3, cudaMemcpyHostToDevice);
  cudaMemcpy(thrust::raw_pointer_cast(&bj_dev[0]), bj, sizeof(double) * n * 3, cudaMemcpyHostToDevice);
  
  Eval.genMatrixDevice(m, n, thrust::raw_pointer_cast(&bi_dev[0]), thrust::raw_pointer_cast(&bj_dev[0]), thrust::raw_pointer_cast(&A_dev[0]), lda, 0);
  cudaMemcpy2D(Aij, sizeof(double) * lda, thrust::raw_pointer_cast(&A_dev[0]), sizeof(double) * lda, sizeof(double) * m, n, cudaMemcpyDeviceToHost);*/
}


/*
#ifdef _GSL
double __matern_h(double r2) {
  double sigma_square = _singularity * _singularity;

  if (r2 > 0.) {
    double con = sigma_square / (pow(2, (_smoothness - 1)) * tgamma(_smoothness));
    double expr = sqrt(2 * _smoothness * r2) / _alpha;
    return con * pow(expr, _smoothness) * gsl_sf_bessel_Knu(_smoothness, expr);
  }
  else
    return sigma_square;
}

#endif

*/

