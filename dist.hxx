
#pragma once

#include "linalg.hxx"

namespace nbd {

  void initComm(int* argc, char** argv[]);

  void closeComm();

  void commRank(int64_t* mpi_rank, int64_t* mpi_size, int64_t* mpi_levels);
  
  void configureComm(int64_t levels, const int64_t ngbs[], int64_t ngs_len);

  void selfLocalRange(int64_t* ibegin, int64_t* iend, int64_t level);

  void iLocal(int64_t* ilocal, int64_t iglobal, int64_t level);

  void iGlobal(int64_t* iglobal, int64_t ilocal, int64_t level);

  void contentLength(int64_t* len, int64_t level);

  void DistributeVectorsList(Vector lis[], int64_t level);

  void DistributeMatricesList(Matrix lis[], int64_t level);

  void DistributeDims(int64_t dims[], int64_t level);

  void DistributeMultipoles(int64_t multipoles[], const int64_t dims[], int64_t level);

  void butterflyUpdateDims(int64_t my_dim, int64_t* rm_dim, int64_t level);

  void butterflyUpdateMultipoles(const int64_t multipoles[], int64_t my_dim, int64_t rm[], int64_t rm_dim, int64_t level);

  void butterflySumA(Matrix A[], int64_t lenA, int64_t level);

  void sendFwSubstituted(const Vector X[], int64_t level);

  void sendBkSubstituted(const Vector X[], int64_t level);

  void recvFwSubstituted(Vector X[], int64_t level);

  void recvBkSubstituted(Vector X[], int64_t level);

  void distributeSubstituted(Vector X[], int64_t level);

  void butterflySumX(Vector X[], int64_t lenX, int64_t level);

  void startTimer(double* wtime, double* cmtime);

  void stopTimer(double* wtime, double* cmtime);

};
