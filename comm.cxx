
#include "comm.hxx"
#include "nbd.hxx"

#include <algorithm>
#include <numeric>

MPI_Comm MPI_Comm_split_unique(std::vector<MPI_Comm>& unique_comms, int color, int mpi_rank) {
  MPI_Comm comm = MPI_COMM_NULL;
  MPI_Comm_split(MPI_COMM_WORLD, color, mpi_rank, &comm);

  if (comm != MPI_COMM_NULL) {
    auto iter = std::find_if(unique_comms.begin(), unique_comms.end(), [comm](MPI_Comm c) -> bool { 
      int result; MPI_Comm_compare(comm, c, &result); return result == MPI_IDENT || result == MPI_CONGRUENT; });
    if (iter == unique_comms.end())
      unique_comms.emplace_back(comm);
    else {
      MPI_Comm_free(&comm);
      comm = *iter;
    }
  }
  return comm;
}

std::pair<int64_t, int64_t> local_to_pnx(int64_t ilocal, const std::vector<std::pair<int64_t, int64_t>>& ProcBoxes) {
  int64_t iter = 0;
  while (iter < (int64_t)ProcBoxes.size() && std::get<1>(ProcBoxes[iter]) <= ilocal) {
    ilocal = ilocal - std::get<1>(ProcBoxes[iter]);
    iter = iter + 1;
  }
  if (0 <= ilocal && iter < (int64_t)ProcBoxes.size())
    return std::make_pair(iter, ilocal);
  else
    return std::make_pair(-1, -1);
}

std::pair<int64_t, int64_t> global_to_pnx(int64_t iglobal, const std::vector<std::pair<int64_t, int64_t>>& ProcBoxes) {
  int64_t iter = 0;
  while (iter < (int64_t)ProcBoxes.size() && (std::get<0>(ProcBoxes[iter]) + std::get<1>(ProcBoxes[iter])) <= iglobal)
    iter = iter + 1;
  if (iter < (int64_t)ProcBoxes.size() && std::get<0>(ProcBoxes[iter]) <= iglobal)
    return std::make_pair(iter, iglobal - std::get<0>(ProcBoxes[iter]));
  else
    return std::make_pair(-1, -1);
}

int64_t pnx_to_local(std::pair<int64_t, int64_t> pnx, const std::vector<std::pair<int64_t, int64_t>>& ProcBoxes) {
  if (std::get<0>(pnx) >= 0 && std::get<0>(pnx) < (int64_t)ProcBoxes.size() && std::get<1>(pnx) >= 0) {
    int64_t iter = 0, slen = 0;
    while (iter < std::get<0>(pnx)) {
      slen = slen + std::get<1>(ProcBoxes[iter]);
      iter = iter + 1;
    }
    return std::get<1>(pnx) + slen;
  }
  else
    return -1;
}

int64_t pnx_to_global(std::pair<int64_t, int64_t> pnx, const std::vector<std::pair<int64_t, int64_t>>& ProcBoxes) {
  if (std::get<0>(pnx) >= 0 && std::get<0>(pnx) < (int64_t)ProcBoxes.size() && std::get<1>(pnx) >= 0)
    return std::get<1>(pnx) + std::get<0>(ProcBoxes[std::get<0>(pnx)]);
  else
    return -1;
}

void buildComm(struct CellComm* comms, int64_t ncells, const struct Cell* cells, const struct CSC* cellFar, const struct CSC* cellNear, int64_t levels) {
  int __mpi_rank = 0, __mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &__mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &__mpi_size);
  int64_t mpi_rank = __mpi_rank;
  int64_t mpi_size = __mpi_size;
  std::vector<MPI_Comm> unique_comms;

  for (int64_t i = levels; i >= 0; i--) {
    int64_t ibegin = 0, iend = ncells;
    get_level(&ibegin, &iend, cells, i, -1);

    int64_t mbegin = ibegin, mend = iend;
    get_level(&mbegin, &mend, cells, i, mpi_rank);
    int64_t p = cells[mbegin].Procs[0];
    int64_t lenp = cells[mbegin].Procs[1] - p;

    std::vector<int64_t> ProcTargets;
    for (int64_t j = 0; j < mpi_size; j++) {
      int is_ngb = 0;
      for (int64_t k = cellNear->ColIndex[mbegin]; k < cellNear->ColIndex[mend]; k++)
        if (cells[cellNear->RowIndex[k]].Procs[0] == j)
          is_ngb = 1;
      for (int64_t k = cellFar->ColIndex[mbegin]; k < cellFar->ColIndex[mend]; k++)
        if (cells[cellFar->RowIndex[k]].Procs[0] == j)
          is_ngb = 1;
      
      int color = (is_ngb && p == mpi_rank) ? 1 : MPI_UNDEFINED;
      MPI_Comm comm = MPI_Comm_split_unique(unique_comms, color, mpi_rank);

      if (comm != MPI_COMM_NULL) {
        int root = 0;
        if (j == p)
          MPI_Comm_rank(comm, &root);
        MPI_Allreduce(MPI_IN_PLACE, &root, 1, MPI_INT, MPI_SUM, comm);
        comms[i].Comm_box.emplace_back(root, comm);
      }
      if (is_ngb)
        ProcTargets.emplace_back(j);
    }
    comms[i].Proc = std::distance(ProcTargets.begin(), std::find(ProcTargets.begin(), ProcTargets.end(), p));

    int color = MPI_UNDEFINED;
    int64_t cc = cells[mbegin].Child[0];
    int64_t clen = cells[mbegin].Child[1] - cc;
    if (lenp > 1 && cc >= 0)
      for (int64_t j = 0; j < clen; j++)
        if (cells[cc + j].Procs[0] == mpi_rank)
          color = p;
    comms[i].Comm_merge = MPI_Comm_split_unique(unique_comms, color, mpi_rank);
  
    color = lenp > 1 ? p : MPI_UNDEFINED;
    comms[i].Comm_share = MPI_Comm_split_unique(unique_comms, color, mpi_rank);

    std::pair<int64_t, int64_t> local = std::make_pair(mbegin, mend - mbegin);
    comms[i].ProcBoxes = std::vector<std::pair<int64_t, int64_t>>(ProcTargets.size(), local);

    for (size_t j = 0; j < comms[i].Comm_box.size(); j++)
      MPI_Bcast(&comms[i].ProcBoxes[j], sizeof(std::pair<int64_t, int64_t>), MPI_BYTE, std::get<0>(comms[i].Comm_box[j]), std::get<1>(comms[i].Comm_box[j]));
    if (comms[i].Comm_share != MPI_COMM_NULL)
      MPI_Bcast(&comms[i].ProcBoxes[0], sizeof(std::pair<int64_t, int64_t>) * comms[i].ProcBoxes.size(), MPI_BYTE, 0, comms[i].Comm_share);

    for (size_t j = 0; j < comms[i].ProcBoxes.size(); j++)
      for (int64_t k = 0; k < std::get<1>(comms[i].ProcBoxes[j]); k++) {
        int64_t ki = k + std::get<0>(comms[i].ProcBoxes[j]);
        int64_t li = pnx_to_local(std::make_pair(j, k), comms[i].ProcBoxes);
        int64_t lc = cells[ki].Child[0];
        int64_t lclen = cells[ki].Child[1] - lc;
        if (i < levels) {
          std::pair<int64_t, int64_t> pnx = global_to_pnx(lc, comms[i + 1].ProcBoxes);
          lc = pnx_to_local(pnx, comms[i + 1].ProcBoxes);
          if (lc >= 0)
            std::for_each(comms[i + 1].LocalParent.begin() + lc, comms[i + 1].LocalParent.begin() + (lc + lclen), 
              [&](std::pair<int64_t, int64_t>& x) { std::get<0>(x) = li; std::get<1>(x) = std::distance(&comms[i + 1].LocalParent[lc], &x); });
          else
            lclen = 0;
        }
        comms[i].LocalChild.emplace_back(lc, lclen);
      }
    comms[i].LocalParent = std::vector<std::pair<int64_t, int64_t>>(comms[i].LocalChild.size(), std::make_pair(-1, -1));
  }

  std::vector<ncclUniqueId> nccl_ids(unique_comms.size());
  std::vector<ncclComm_t> nccl_comms(unique_comms.size());
  ncclGroupStart();
  for (int64_t i = 0; i < (int64_t)unique_comms.size(); i++) {
    int rank, size;
    MPI_Comm_rank(unique_comms[i], &rank);
    MPI_Comm_size(unique_comms[i], &size);
    if (rank == 0)
      ncclGetUniqueId(&nccl_ids[i]);
    MPI_Bcast((void*)&nccl_ids[i], sizeof(ncclUniqueId), MPI_BYTE, 0, unique_comms[i]);
    ncclCommInitRank(&nccl_comms[i], size, nccl_ids[i], rank);
  }
  ncclGroupEnd();

  for (int64_t i = levels; i >= 0; i--) {
    comms[i].NCCL_box = std::vector<std::pair<int, ncclComm_t>>(comms[i].Comm_box.size(), std::make_pair(0, (ncclComm_t)NULL));
    comms[i].NCCL_merge = NULL;
    comms[i].NCCL_share = NULL; 

    for (size_t j = 0; j < comms[i].Comm_box.size(); j++) {
      int64_t k = std::distance(unique_comms.begin(), std::find(unique_comms.begin(), unique_comms.end(), std::get<1>(comms[i].Comm_box[j])));
      comms[i].NCCL_box[j] = std::make_pair(std::get<0>(comms[i].Comm_box[j]), nccl_comms[k]);
    }
    if (comms[i].Comm_merge != MPI_COMM_NULL) {
      int64_t k = std::distance(unique_comms.begin(), std::find(unique_comms.begin(), unique_comms.end(), comms[i].Comm_merge));
      comms[i].NCCL_merge = nccl_comms[k];
    }
    if (comms[i].Comm_share != MPI_COMM_NULL) {
      int64_t k = std::distance(unique_comms.begin(), std::find(unique_comms.begin(), unique_comms.end(), comms[i].Comm_share));
      comms[i].NCCL_share = nccl_comms[k];
    }
  }
}

void cellComm_free(struct CellComm* comms, int64_t levels) {
  std::vector<MPI_Comm> mpi_comms;
  std::vector<ncclComm_t> nccl_comms;

  for (int64_t i = 0; i <= levels; i++) {
    for (size_t j = 0; j < comms[i].Comm_box.size(); j++)
      mpi_comms.emplace_back(std::get<1>(comms[i].Comm_box[j]));
    if (comms[i].Comm_merge != MPI_COMM_NULL)
      mpi_comms.emplace_back(comms[i].Comm_merge);
    if (comms[i].Comm_share != MPI_COMM_NULL)
      mpi_comms.emplace_back(comms[i].Comm_share);

    for (size_t j = 0; j < comms[i].NCCL_box.size(); j++)
      nccl_comms.emplace_back(std::get<1>(comms[i].NCCL_box[j]));
    if (comms[i].NCCL_merge != NULL)
      nccl_comms.emplace_back(comms[i].NCCL_merge);
    if (comms[i].NCCL_share != NULL)
      nccl_comms.emplace_back(comms[i].NCCL_share);
    
    comms[i].Comm_box.clear();
    comms[i].Comm_merge = MPI_COMM_NULL;
    comms[i].Comm_share = MPI_COMM_NULL;
    comms[i].NCCL_box.clear();
    comms[i].NCCL_merge = NULL;
    comms[i].NCCL_share = NULL;
  }

  std::sort(mpi_comms.begin(), mpi_comms.end());
  std::sort(nccl_comms.begin(), nccl_comms.end());
  mpi_comms.erase(std::unique(mpi_comms.begin(), mpi_comms.end()), mpi_comms.end());
  nccl_comms.erase(std::unique(nccl_comms.begin(), nccl_comms.end()), nccl_comms.end());

  for (int64_t i = 0; i < (int64_t)mpi_comms.size(); i++)
    MPI_Comm_free(&mpi_comms[i]);
  for (int64_t i = 0; i < (int64_t)nccl_comms.size(); i++)
    ncclCommDestroy(nccl_comms[i]);
}

void i_local(int64_t* ilocal, const struct CellComm* comm) {
  int64_t iglobal = *ilocal;
  *ilocal = pnx_to_local(global_to_pnx(iglobal, comm->ProcBoxes), comm->ProcBoxes);
}

void i_global(int64_t* iglobal, const struct CellComm* comm) {
  int64_t ilocal = *iglobal;
  *iglobal = pnx_to_global(local_to_pnx(ilocal, comm->ProcBoxes), comm->ProcBoxes);
}

void content_length(int64_t* local, int64_t* neighbors, int64_t* local_off, const struct CellComm* comm) {
  int64_t slen = 0, offset = -1, len_self = -1;
  for (int64_t i = 0; i < (int64_t)comm->ProcBoxes.size(); i++) {
    if (i == comm->Proc)
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

