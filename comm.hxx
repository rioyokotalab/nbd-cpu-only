
#pragma once

#include "mpi.h"
#include "cuda_runtime_api.h"
#include "nccl.h"

#include <vector>
#include <utility>
#include <cstdint>
#include <cstddef>

struct CellComm { 
  int64_t Proc[2];
  std::vector<int64_t> ProcTargets;
  std::vector<std::pair<int64_t, int64_t>> LocalChild;
  std::vector<std::pair<int64_t, int64_t>> ProcBoxes;
  std::vector<std::pair<int, MPI_Comm>> Comm_box;
  MPI_Comm Comm_share, Comm_merge;
  std::vector<std::pair<int, ncclComm_t>> NCCL_box;
  ncclComm_t NCCL_merge, NCCL_share;
};

void buildComm(struct CellComm* comms, int64_t ncells, const struct Cell* cells, const struct CSC* cellFar, const struct CSC* cellNear, int64_t levels);

void buildCommGPU(struct CellComm* comms, int64_t levels);

void cellComm_free(struct CellComm* comm);

void i_local(int64_t* ilocal, const struct CellComm* comm);

void i_global(int64_t* iglobal, const struct CellComm* comm);

void self_local_range(int64_t* ibegin, int64_t* iend, const struct CellComm* comm);

void content_length(int64_t* len, const struct CellComm* comm);

void get_segment_sizes(int64_t* dimS, int64_t* dimR, const int64_t* nchild, int64_t alignment, int64_t levels);
