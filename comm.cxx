
#include "comm.hxx"
#include "nbd.hxx"

#include <algorithm>
#include <cmath>

void buildComm(struct CellComm* comms, int64_t ncells, const struct Cell* cells, const struct CSC* cellFar, const struct CSC* cellNear, int64_t levels) {
  int __mpi_rank = 0, __mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &__mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &__mpi_size);
  int64_t mpi_rank = __mpi_rank;
  int64_t mpi_size = __mpi_size;

  for (int64_t i = levels; i >= 0; i--) {
    int64_t ibegin = 0, iend = ncells;
    get_level(&ibegin, &iend, cells, i, -1);

    int64_t mbegin = ibegin, mend = iend;
    get_level(&mbegin, &mend, cells, i, mpi_rank);
    int64_t p = cells[mbegin].Procs[0];
    int64_t lenp = cells[mbegin].Procs[1] - p;
    comms[i].Proc[0] = p;
    comms[i].Proc[1] = p + lenp;

    for (int64_t j = 0; j < mpi_size; j++) {
      int is_ngb = 0;
      for (int64_t k = cellNear->ColIndex[mbegin]; k < cellNear->ColIndex[mend]; k++)
        if (cells[cellNear->RowIndex[k]].Procs[0] == j)
          is_ngb = 1;
      for (int64_t k = cellFar->ColIndex[mbegin]; k < cellFar->ColIndex[mend]; k++)
        if (cells[cellFar->RowIndex[k]].Procs[0] == j)
          is_ngb = 1;
      
      int color = (is_ngb && p == mpi_rank) ? 1 : MPI_UNDEFINED;
      MPI_Comm comm = MPI_COMM_NULL;
      MPI_Comm_split(MPI_COMM_WORLD, color, mpi_rank, &comm);

      if (comm != MPI_COMM_NULL) {
        int root = 0;
        if (j == p)
          MPI_Comm_rank(comm, &root);
        comms[i].Comm_box.emplace_back(root, comm);
      }
      if (is_ngb)
        comms[i].ProcTargets.emplace_back(j);
    }

    int color = MPI_UNDEFINED;
    int64_t cc = cells[mbegin].Child[0];
    int64_t clen = cells[mbegin].Child[1] - cc;
    if (lenp > 1 && cc >= 0)
      for (int64_t j = 0; j < clen; j++)
        if (cells[cc + j].Procs[0] == mpi_rank)
          color = p;
    MPI_Comm_split(MPI_COMM_WORLD, color, mpi_rank, &comms[i].Comm_merge);
  
    color = lenp > 1 ? p : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color, mpi_rank, &comms[i].Comm_share);

    for (size_t j = 0; j < comms[i].ProcTargets.size(); j++) {
      int64_t local[2] { mbegin - ibegin, mend - mbegin };
      if (p == mpi_rank) {
        MPI_Allreduce(MPI_IN_PLACE, &std::get<0>(comms[i].Comm_box[j]), 1, MPI_INT, MPI_SUM, std::get<1>(comms[i].Comm_box[j]));
        MPI_Bcast(local, 2, MPI_INT64_T, std::get<0>(comms[i].Comm_box[j]), std::get<1>(comms[i].Comm_box[j]));
      }
      if (comms[i].Comm_share != MPI_COMM_NULL)
        MPI_Bcast(local, 2, MPI_INT64_T, 0, comms[i].Comm_share);
      comms[i].ProcBoxes.emplace_back(local[0], local[1]);

      for (int64_t k = 0; k < local[1]; k++) {
        int64_t ki = k + local[0] + ibegin;
        int64_t lc = cells[ki].Child[0];
        int64_t lclen = cells[ki].Child[1] - lc;
        if (lc >= 0 && i < levels) {
          lc = lc - iend;
          i_local(&lc, &comms[i + 1]);
        }
        comms[i].LocalChild.emplace_back(lc, lclen);
      }
    }
  }
}

void buildCommGPU(struct CellComm* comms, int64_t levels) {
  for (int64_t i = 0; i <= levels; i++) {
    ncclUniqueId id;
    for (size_t j = 0; j < comms[i].Comm_box.size(); j++) {
      int rank, size, root = std::get<0>(comms[i].Comm_box[j]);
      MPI_Comm_rank(std::get<1>(comms[i].Comm_box[j]), &rank);
      MPI_Comm_size(std::get<1>(comms[i].Comm_box[j]), &size);

      if (rank == root) 
        ncclGetUniqueId(&id);

      MPI_Bcast((void*)&id, sizeof(ncclUniqueId), MPI_BYTE, root, std::get<1>(comms[i].Comm_box[j]));
      ncclComm_t comm = NULL;
      ncclResult_t err = ncclCommInitRank(&comm, size, id, rank);
      if (err == ncclSuccess)
        comms[i].NCCL_box.emplace_back(root, comm);
    }

    if (comms[i].Comm_merge != MPI_COMM_NULL) {
      int rank, size;
      MPI_Comm_rank(comms[i].Comm_merge, &rank);
      MPI_Comm_size(comms[i].Comm_merge, &size);

      if (rank == 0) 
        ncclGetUniqueId(&id);

      MPI_Bcast((void*)&id, sizeof(ncclUniqueId), MPI_BYTE, 0, comms[i].Comm_merge);
      ncclComm_t comm = NULL;
      ncclResult_t err = ncclCommInitRank(&comm, size, id, rank);
      comms[i].NCCL_merge = (err == ncclSuccess) ? comm : NULL;
    }

    if (comms[i].Comm_share!= MPI_COMM_NULL) {
      int rank, size;
      MPI_Comm_rank(comms[i].Comm_share, &rank);
      MPI_Comm_size(comms[i].Comm_share, &size);

      if (rank == 0) 
        ncclGetUniqueId(&id);

      MPI_Bcast((void*)&id, sizeof(ncclUniqueId), MPI_BYTE, 0, comms[i].Comm_share);
      ncclComm_t comm = NULL;
      ncclResult_t err = ncclCommInitRank(&comm, size, id, rank);
      comms[i].NCCL_share = (err == ncclSuccess) ? comm : NULL;
    }
  }
}

void cellComm_free(struct CellComm* comm) {
  for (int64_t i = 0; i < (int64_t)comm->Comm_box.size(); i++)
    MPI_Comm_free(&std::get<1>(comm->Comm_box[i]));
  if (comm->Comm_share != MPI_COMM_NULL)
    MPI_Comm_free(&comm->Comm_share);
  if (comm->Comm_merge != MPI_COMM_NULL)
    MPI_Comm_free(&comm->Comm_merge);

  for (int64_t i = 0; i < (int64_t)comm->NCCL_box.size(); i++)
    ncclCommDestroy(std::get<1>(comm->NCCL_box[i]));
  if (comm->NCCL_share != NULL)
    ncclCommDestroy(comm->NCCL_share);
  if (comm->NCCL_merge != NULL)
    ncclCommDestroy(comm->NCCL_merge);
}

void i_local(int64_t* ilocal, const struct CellComm* comm) {
  int64_t iglobal = *ilocal;
  size_t iter = 0;
  int64_t slen = 0;
  while (iter < comm->ProcTargets.size() && 
  (std::get<0>(comm->ProcBoxes[iter]) + std::get<1>(comm->ProcBoxes[iter])) <= iglobal) {
    slen = slen + std::get<1>(comm->ProcBoxes[iter]);
    iter = iter + 1;
  }
  if (iter < comm->ProcTargets.size() && std::get<0>(comm->ProcBoxes[iter]) <= iglobal)
    *ilocal = slen + iglobal - std::get<0>(comm->ProcBoxes[iter]);
  else
    *ilocal = -1;
}

void i_global(int64_t* iglobal, const struct CellComm* comm) {
  int64_t ilocal = *iglobal;
  size_t iter = 0;
  while (iter < comm->ProcTargets.size() && std::get<1>(comm->ProcBoxes[iter]) <= ilocal) {
    ilocal = ilocal - std::get<1>(comm->ProcBoxes[iter]);
    iter = iter + 1;
  }
  if (0 <= ilocal && iter < comm->ProcTargets.size())
    *iglobal = std::get<0>(comm->ProcBoxes[iter]) + ilocal;
  else
    *iglobal = -1;
}

void content_length(int64_t* local, int64_t* neighbors, int64_t* local_off, const struct CellComm* comm) {
  int64_t slen = 0, offset = -1, len_self = -1;
  for (size_t i = 0; i < comm->ProcTargets.size(); i++) {
    if (comm->ProcTargets[i] == comm->Proc[0])
    { offset = slen; len_self = std::get<1>(comm->ProcBoxes[i]); }
    slen = slen + std::get<1>(comm->ProcBoxes[i]);
  }
  if (local)
    *local = len_self;
  if (neighbors)
    *neighbors = slen;
  if (local_off)
    *local_off = offset;
}

int64_t neighbor_bcast_sizes_cpu(int64_t* data, const struct CellComm* comm) {
  int64_t y = 0;
  for (size_t p = 0; p < comm->Comm_box.size(); p++) {
    int64_t llen = std::get<1>(comm->ProcBoxes[p]);
    int64_t* loc = &data[y];
    MPI_Bcast(loc, llen, MPI_INT64_T, std::get<0>(comm->Comm_box[p]), std::get<1>(comm->Comm_box[p]));
    y = y + llen;
  }
  int64_t len = 0;
  content_length(NULL, &len, NULL, comm);
  if (comm->Comm_share != MPI_COMM_NULL)
    MPI_Bcast(data, len, MPI_DOUBLE, 0, comm->Comm_share);

  int64_t max = 0;
  for (int64_t i = 0; i < len; i++)
    max = std::max(max, data[i]);
  MPI_Allreduce(MPI_IN_PLACE, &max, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
  return max;
}

void neighbor_bcast_cpu(double* data, int64_t seg, const struct CellComm* comm) {
  int64_t y = 0;
  for (size_t p = 0; p < comm->Comm_box.size(); p++) {
    int64_t llen = std::get<1>(comm->ProcBoxes[p]) * seg;
    double* loc = &data[y];
    MPI_Bcast(loc, llen, MPI_DOUBLE, std::get<0>(comm->Comm_box[p]), std::get<1>(comm->Comm_box[p]));
    y = y + llen;
  }
}

void neighbor_reduce_cpu(double* data, int64_t seg, const struct CellComm* comm) {
  int64_t y = 0;
  for (size_t p = 0; p < comm->Comm_box.size(); p++) {
    int64_t llen = std::get<1>(comm->ProcBoxes[p]) * seg;
    double* loc = &data[y];
    MPI_Allreduce(MPI_IN_PLACE, loc, llen, MPI_DOUBLE, MPI_SUM, std::get<1>(comm->Comm_box[p]));
    y = y + llen;
  }
}

void level_merge_cpu(double* data, int64_t len, const struct CellComm* comm) {
  if (comm->Comm_merge != MPI_COMM_NULL)
    MPI_Allreduce(MPI_IN_PLACE, data, len, MPI_DOUBLE, MPI_SUM, comm->Comm_merge);
}

void dup_bcast_cpu(double* data, int64_t len, const struct CellComm* comm) {
  if (comm->Comm_share != MPI_COMM_NULL)
    MPI_Bcast(data, len, MPI_DOUBLE, 0, comm->Comm_share);
}

void neighbor_bcast_gpu(double* data, int64_t seg, cudaStream_t stream, const struct CellComm* comm) {
  int64_t y = 0;
  for (size_t p = 0; p < comm->NCCL_box.size(); p++) {
    int64_t llen = std::get<1>(comm->ProcBoxes[p]) * seg;
    double* loc = &data[y];
    ncclBroadcast((const void*)loc, loc, llen, ncclDouble, std::get<0>(comm->NCCL_box[p]), std::get<1>(comm->NCCL_box[p]), stream);
    y = y + llen;
  }
}

void neighbor_reduce_gpu(double* data, int64_t seg, cudaStream_t stream, const struct CellComm* comm) {
  int64_t y = 0;
  for (size_t p = 0; p < comm->NCCL_box.size(); p++) {
    int64_t llen = std::get<1>(comm->ProcBoxes[p]) * seg;
    double* loc = &data[y];
    ncclAllReduce((const void*)loc, loc, llen, ncclDouble, ncclSum, std::get<1>(comm->NCCL_box[p]), stream);
    y = y + llen;
  }
}

void level_merge_gpu(double* data, int64_t len, cudaStream_t stream, const struct CellComm* comm) {
  if (comm->NCCL_merge != NULL)
    ncclAllReduce((const void*)data, data, len, ncclDouble, ncclSum, comm->NCCL_merge, stream);
}

void dup_bcast_gpu(double* data, int64_t len, cudaStream_t stream, const struct CellComm* comm) {
  if (comm->NCCL_share != NULL)
    ncclBroadcast((const void*)data, data, len, ncclDouble, 0, comm->NCCL_share, stream);
}

