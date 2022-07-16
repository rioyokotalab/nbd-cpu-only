
#pragma once

#include "mpi.h"
#include "stdint.h"
#include "stddef.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Matrix;
struct Base;
struct Node;
struct RightHandSides;

void matrix_mem(int64_t* bytes, const struct Matrix* A, int64_t lenA);

void basis_mem(int64_t* bytes, const struct Base* basis, int64_t levels);

void node_mem(int64_t* bytes, const struct Node* node, int64_t levels);

void rightHandSides_mem(int64_t* bytes, const struct RightHandSides* st, int64_t levels);

void startTimer(double* wtime, double* cmtime);

void stopTimer(double* wtime, double* cmtime);

void recordCommTime(double cmtime);

void getCommTime(double* cmtime);

#ifdef __cplusplus
}
#endif
