
#pragma once

#include "mpi.h"

#include <vector>
#include <utility>
#include <cstdint>
#include <cstddef>

struct CommTimer {
  std::vector<std::pair<double, double>> timings;

  void record_mpi(double t1, double t2)
  { timings.emplace_back(t1, t2); }

  double get_comm_timing() {
    double sum = 0.;
    for (auto& i : timings)
      sum = sum + i.second - i.first;
    timings.clear();
    return sum;
  }
};

struct CellComm {
  int64_t Proc;
  std::vector<std::pair<int64_t, int64_t>> ProcBoxes;
  std::vector<std::pair<int64_t, int64_t>> LocalChild, LocalParent;
  
  std::vector<std::pair<int, MPI_Comm>> Comm_box;
  MPI_Comm Comm_share, Comm_merge;
  
  CommTimer* timer;
};

void buildComm(struct CellComm* comms, int64_t ncells, const struct Cell* cells, const struct CSC* cellFar, const struct CSC* cellNear, int64_t levels);

void cellComm_free(struct CellComm* comms, int64_t levels);

void relations(struct CSC rels[], const struct CSC* cellRel, int64_t levels, const struct CellComm* comm);

void i_local(int64_t* ilocal, const struct CellComm* comm);

void i_global(int64_t* iglobal, const struct CellComm* comm);

void content_length(int64_t* local, int64_t* neighbors, int64_t* local_off, const struct CellComm* comm);

int64_t neighbor_bcast_sizes_cpu(int64_t* data, const struct CellComm* comm);

void neighbor_bcast_cpu(double* data, int64_t seg, const struct CellComm* comm);

void neighbor_reduce_cpu(double* data, int64_t seg, const struct CellComm* comm);

void level_merge_cpu(double* data, int64_t len, const struct CellComm* comm);

void dup_bcast_cpu(double* data, int64_t len, const struct CellComm* comm);

