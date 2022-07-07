
#pragma once

#include "stddef.h"
#include "stdint.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Matrix;

void initComm();

void closeComm();

void commRank(int64_t* mpi_rank, int64_t* mpi_size);

void butterflyComm(int* comm_needed, int64_t level);

void configureComm(int64_t levels, const int64_t ngbs[], int64_t ngs_len);

void selfLocalRange(int64_t* ibegin, int64_t* iend, int64_t level);

void iLocal(int64_t* ilocal, int64_t iglobal, int64_t level);

void iGlobal(int64_t* iglobal, int64_t ilocal, int64_t level);

void contentLength(int64_t* len, int64_t level);

void DistributeMatricesList(struct Matrix lis[], int64_t level);

void DistributeDims(int64_t dims[], int64_t level);

void DistributeMultipoles(int64_t multipoles[], const int64_t dims[], int64_t level);

void butterflySumA(struct Matrix A[], int64_t lenA, int64_t level);

void sendFwSubstituted(const struct Matrix A[], int64_t level);

void sendBkSubstituted(const struct Matrix A[], int64_t level);

void recvFwSubstituted(struct Matrix A[], int64_t level);

void recvBkSubstituted(struct Matrix A[], int64_t level);

void distributeSubstituted(struct Matrix A[], int64_t level);

void startTimer(double* wtime, double* cmtime);

void stopTimer(double* wtime, double* cmtime);

#ifdef __cplusplus
}
#endif
