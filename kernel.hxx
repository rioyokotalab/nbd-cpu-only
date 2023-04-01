#pragma once

#include <cuda_runtime_api.h>
#include <cuComplex.h>
#include <cstdint>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>

struct BodiesIter : public thrust::unary_function<int64_t, thrust::pair<const double*, const double*>> {
  int64_t M;
  const double* Bodies_i, *Bodies_j;
  BodiesIter (int64_t m, const double* bi, const double* bj);
  __host__ __device__ thrust::pair<const double*, const double*> operator()(int64_t i) const;
};

struct ArrIter : public thrust::unary_function<int64_t, int64_t> {
  int64_t M, LDA_M_diff;
  ArrIter (int64_t m, int64_t lda);
  __host__ __device__ int64_t operator()(int64_t x) const;
};

thrust::transform_iterator<BodiesIter, thrust::counting_iterator<int64_t>>
  createBodiesIter(int64_t m, const double* bi, const double* bj);

thrust::permutation_iterator<double*, thrust::transform_iterator<ArrIter, thrust::counting_iterator<int64_t>>>
  createArrIter(int64_t m, double A[], int64_t lda);

__host__ __device__ double computeR2(const double bi[], const double bj[]);

struct EvalDouble : public thrust::unary_function<thrust::pair<const double*, const double*>, double> {
  __host__ __device__ virtual double operator()(thrust::pair<const double*, const double*> i) const = 0;
  virtual void genMatrixHost(int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda) const = 0;
  virtual void genMatrixDevice(int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda, cudaStream_t stream) const = 0;
};

struct Laplace3D : public EvalDouble {
  double singularity;
  Laplace3D (double s) : singularity(1. / s) {}
  __host__ __device__ double operator()(thrust::pair<const double*, const double*> x) const override;
  void genMatrixHost(int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda) const override;
  void genMatrixDevice(int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda, cudaStream_t stream) const override;
};

struct Yukawa3D : public EvalDouble {
  double singularity, alpha;
  Yukawa3D (double s, double a) : singularity(1. / s), alpha(a) {}
  __host__ __device__ double operator()(thrust::pair<const double*, const double*> x) const override;
  void genMatrixHost(int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda) const override;
  void genMatrixDevice(int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda, cudaStream_t stream) const override;
};

struct Gaussian : public EvalDouble {
  double alpha;
  Gaussian (double a) : alpha(1. / (a * a)) {}
  __host__ __device__ double operator()(thrust::pair<const double*, const double*> x) const override;
  void genMatrixHost(int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda) const override;
  void genMatrixDevice(int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda, cudaStream_t stream) const override;
};

void gen_matrix(const EvalDouble& Eval, int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda);

