
#pragma once

#include "build_tree.hxx"

namespace nbd {

  void initComm(int* argc, char** argv[]);

  void closeComm();

  void commRank(int64_t* mpi_rank, int64_t* mpi_size, int64_t* mpi_levels);

  void configureComm(int64_t level, const int64_t ngbs[], int64_t ngs_len);

  void selfLocalRange(int64_t& ibegin, int64_t& iend, int64_t level);

  void neighborsILocal(int64_t& ilocal, int64_t iglobal, int64_t level);

  void neighborsIGlobal(int64_t& iglobal, int64_t ilocal, int64_t level);

  void neighborContentLength(int64_t& len, int64_t level);

  void locateCOMM(int64_t level, int64_t* my_ind, int64_t* my_rank, int64_t* nboxes, int64_t** ngbs, int64_t* ngbs_len);

  void locateButterflyCOMM(int64_t level, int64_t* my_ind, int64_t* my_rank, int64_t* my_twi, int64_t* twi_rank);

  void DistributeVectorsList(Vectors& B, int64_t level);

  void DistributeMatricesList(Matrices& lis, int64_t level);

  void DistributeDims(int64_t dims[], int64_t level);

  void axatDistribute(Matrices& A, const CSC& rels, int64_t level);

  void butterflySumA(Matrices& A, int64_t level);

  void sendFwSubstituted(const Vectors& X, int64_t level);

  void sendBkSubstituted(const Vectors& X, int64_t level);

  void recvFwSubstituted(Vectors& X, int64_t level);

  void recvBkSubstituted(Vectors& X, int64_t level);

  void distributeSubstituted(Vectors& X, int64_t level);

  void butterflySumX(Vectors& X, int64_t level);

  void startTimer(double* wtime);

  void stopTimer(double wtime, const char str[]);

};