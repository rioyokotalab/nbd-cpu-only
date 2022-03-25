
#include "dist.hxx"

#include "mpi.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

using namespace nbd;

struct Communicator {
  int64_t SELF_I;
  int64_t TWIN_I;
  std::vector<int64_t> NGB_RNKS;
  std::vector<int64_t> COMM_RNKS;
};

int64_t MPI_RANK = 0;
int64_t MPI_SIZE = 1;
int64_t MPI_LEVELS = 0;
std::vector<Communicator> COMMS;

double prog_time = 0.;
double tot_time = 0.;
double last_mark = 0.;

void nbd::initComm(int* argc, char** argv[]) {
  MPI_Init(argc, argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_RANK = rank;
  MPI_SIZE = size;
  MPI_LEVELS = (int64_t)std::log2(size);
  COMMS.resize(MPI_LEVELS + 1);

  prog_time = MPI_Wtime();
  tot_time = 0.;
  last_mark = 0.;
}

void nbd::closeComm() {
  if (MPI_RANK == 0) {
    double end = MPI_Wtime();
    double prog = end - prog_time;
    printf("Program %f s. Time in communication: %f s.\n", prog, tot_time);
  }

  for (int64_t i = 0; i <= MPI_LEVELS; i++) {
    COMMS[i].NGB_RNKS.clear();
    COMMS[i].COMM_RNKS.clear();
  }
  COMMS.clear();
  MPI_Finalize();
}

void nbd::commRank(int64_t* mpi_rank, int64_t* mpi_size, int64_t* mpi_levels) {
  if (mpi_rank)
    *mpi_rank = MPI_RANK;
  if (mpi_size)
    *mpi_size = MPI_SIZE;
  if (mpi_levels)
    *mpi_levels = MPI_LEVELS;
}

void nbd::configureComm(int64_t level, const int64_t ngbs[], int64_t ngbs_len) {
  if (level <= MPI_LEVELS && level >= 0) {
    int64_t lvl_diff = MPI_LEVELS - level;
    int64_t my_rank = MPI_RANK >> lvl_diff;
    Communicator& gi = COMMS[level];

    int64_t self = std::distance(ngbs, std::find(ngbs, ngbs + ngbs_len, my_rank));
    if (self < ngbs_len) {
      int64_t my_twin = my_rank ^ (int64_t)1;
      int64_t mask = MPI_RANK - (my_rank << lvl_diff);
      int64_t twi = std::distance(ngbs, std::find(ngbs, ngbs + ngbs_len, my_twin));
      gi.SELF_I = self;
      gi.TWIN_I = twi == ngbs_len ? -1 : twi;
      gi.NGB_RNKS.resize(ngbs_len);
      gi.COMM_RNKS.resize(ngbs_len);
      for (int64_t i = 0; i < ngbs_len; i++) {
        gi.NGB_RNKS[i] = ngbs[i];
        gi.COMM_RNKS[i] = (ngbs[i] << lvl_diff) | mask;
      }
    }
  }
}

void nbd::selfLocalRange(int64_t& ibegin, int64_t& iend, int64_t level) {
  if (level >= 0) {
    int64_t lvl_diff = level - MPI_LEVELS;
    int64_t boxes = lvl_diff > 0 ? ((int64_t)1 << lvl_diff) : 1;
    const Communicator& gi = lvl_diff > 0 ? COMMS[MPI_LEVELS] : COMMS[level];
    ibegin = gi.SELF_I * boxes;
    iend = gi.SELF_I * boxes + boxes;
  }
}

void nbd::neighborsILocal(int64_t& ilocal, int64_t iglobal, int64_t level) {
  if (level >= 0) {
    int64_t lvl_diff = level - MPI_LEVELS;
    int64_t boxes = lvl_diff > 0 ? ((int64_t)1 << lvl_diff) : 1;
    int64_t box_rank = iglobal / boxes;
    int64_t i_rank = iglobal - box_rank * boxes;

    const Communicator& gi = lvl_diff > 0 ? COMMS[MPI_LEVELS] : COMMS[level];
    const int64_t* ngbs = &gi.NGB_RNKS[0];
    int64_t ngbs_len = gi.NGB_RNKS.size();

    int64_t box_ind = std::distance(ngbs, std::find(ngbs, ngbs + ngbs_len, box_rank));
    ilocal = (box_ind < ngbs_len) ? (box_ind * boxes + i_rank) : -1;
  }
}

void nbd::neighborsIGlobal(int64_t& iglobal, int64_t ilocal, int64_t level) {
  if (level >= 0) {
    int64_t lvl_diff = level - MPI_LEVELS;
    int64_t boxes = lvl_diff > 0 ? ((int64_t)1 << lvl_diff) : 1;
    int64_t box_ind = ilocal / boxes;
    int64_t i_rank = ilocal - box_ind * boxes;

    const Communicator& gi = lvl_diff > 0 ? COMMS[MPI_LEVELS] : COMMS[level];
    const int64_t* ngbs = &gi.NGB_RNKS[0];
    int64_t ngbs_len = gi.NGB_RNKS.size();

    iglobal = (box_ind < ngbs_len) ? (ngbs[box_ind] * boxes + i_rank) : -1;
  }
}

void nbd::neighborContentLength(int64_t& len, int64_t level) {
  if (level >= 0) {
    int64_t lvl_diff = level - MPI_LEVELS;
    int64_t boxes = lvl_diff > 0 ? ((int64_t)1 << lvl_diff) : 1;
    const Communicator& gi = lvl_diff > 0 ? COMMS[MPI_LEVELS] : COMMS[level];
    int64_t ngbs_len = gi.NGB_RNKS.size();
    len = boxes * ngbs_len;
  }
}

void nbd::locateCOMM(int64_t level, int64_t* my_ind, int64_t* my_rank, int64_t* nboxes, int64_t** ngbs, int64_t* ngbs_len) {
  if (level >= 0) {
    int64_t lvl_diff = level - MPI_LEVELS;
    int64_t boxes = lvl_diff > 0 ? ((int64_t)1 << lvl_diff) : 1;
    Communicator& gi = lvl_diff > 0 ? COMMS[MPI_LEVELS] : COMMS[level];
    
    if (my_ind)
      *my_ind = gi.SELF_I;
    if (my_rank)
      *my_rank = gi.COMM_RNKS[gi.SELF_I];
    if (nboxes)
      *nboxes = boxes;
    if (ngbs)
      *ngbs = &gi.COMM_RNKS[0];
    if (ngbs_len)
      *ngbs_len = gi.COMM_RNKS.size();
  }
}

void nbd::locateButterflyCOMM(int64_t level, int64_t* my_ind, int64_t* my_rank, int64_t* my_twi, int64_t* twi_rank) {
  if (level >= 0) {
    Communicator& gi = level > MPI_LEVELS ? COMMS[MPI_LEVELS] : COMMS[level];
    
    if (my_ind)
      *my_ind = gi.SELF_I;
    if (my_rank)
      *my_rank = gi.COMM_RNKS[gi.SELF_I];
    if (my_twi)
      *my_twi = gi.TWIN_I;
    if (twi_rank)
      *twi_rank = gi.TWIN_I == -1 ? -1 : gi.COMM_RNKS[gi.TWIN_I];
  }
}

void nbd::DistributeVectorsList(Vectors& lis, int64_t level) {
  int64_t my_ind, my_rank, nboxes, *ngbs, ngbs_len;
  locateCOMM(level, &my_ind, &my_rank, &nboxes, &ngbs, &ngbs_len);
  std::vector<MPI_Request> requests(ngbs_len);

  std::vector<int64_t> LENS(ngbs_len);
  std::vector<double*> DATA(ngbs_len);

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    int64_t tot_len = 0;
    for (int64_t n = 0; n < nboxes; n++) {
      int64_t rm_i = i * nboxes + n;
      const Vector& B_i = lis[rm_i];
      int64_t len = B_i.N;
      tot_len = tot_len + len;
    }
    LENS[i] = tot_len;
    DATA[i] = (double*)malloc(sizeof(double) * tot_len);
  }

  int64_t offset = 0;
  double* my_data = DATA[my_ind];
  int64_t my_len = LENS[my_ind];
  for (int64_t n = 0; n < nboxes; n++) {
    int64_t my_i = my_ind * nboxes + n;
    const Vector& B_i = lis[my_i];
    int64_t len = B_i.N;
    cpyFromVector(B_i, my_data + offset);
    offset = offset + len;
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank)
      MPI_Isend(my_data, (int)my_len, MPI_DOUBLE, (int)rm_rank, 0, MPI_COMM_WORLD, &requests[i]);
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank) {
      double* data = DATA[i];
      int64_t len = LENS[i];
      MPI_Recv(data, (int)len, MPI_DOUBLE, (int)rm_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank)
      MPI_Wait(&requests[i], MPI_STATUS_IGNORE);
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank) {
      offset = 0;
      const double* rm_v = DATA[i];
      for (int64_t n = 0; n < nboxes; n++) {
        int64_t rm_i = i * nboxes + n;
        Vector& B_i = lis[rm_i];
        int64_t len = B_i.N;
        vaxpby(B_i, rm_v + offset, 1., 0.);
        offset = offset + len;
      }
    }
  }

  for (int64_t i = 0; i < ngbs_len; i++)
    free(DATA[i]);
}

void nbd::DistributeMatricesList(Matrices& lis, int64_t level) {
  int64_t my_ind, my_rank, nboxes, *ngbs, ngbs_len;
  locateCOMM(level, &my_ind, &my_rank, &nboxes, &ngbs, &ngbs_len);
  std::vector<MPI_Request> requests(ngbs_len);

  std::vector<int64_t> LENS(ngbs_len);
  std::vector<double*> DATA(ngbs_len);

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    int64_t tot_len = 0;
    for (int64_t n = 0; n < nboxes; n++) {
      int64_t rm_i = i * nboxes + n;
      const Matrix& A_i = lis[rm_i];
      int64_t len = A_i.M * A_i.N;
      tot_len = tot_len + len;
    }
    LENS[i] = tot_len;
    DATA[i] = (double*)malloc(sizeof(double) * tot_len);
  }

  int64_t offset = 0;
  double* my_data = DATA[my_ind];
  int64_t my_len = LENS[my_ind];
  for (int64_t n = 0; n < nboxes; n++) {
    int64_t my_i = my_ind * nboxes + n;
    const Matrix& A_i = lis[my_i];
    int64_t len = A_i.M * A_i.N;
    cpyFromMatrix('N', A_i, my_data + offset);
    offset = offset + len;
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank)
      MPI_Isend(my_data, (int)my_len, MPI_DOUBLE, (int)rm_rank, 0, MPI_COMM_WORLD, &requests[i]);
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank) {
      double* data = DATA[i];
      int64_t len = LENS[i];
      MPI_Recv(data, (int)len, MPI_DOUBLE, (int)rm_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank)
      MPI_Wait(&requests[i], MPI_STATUS_IGNORE);
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank) {
      offset = 0;
      const double* rm_v = DATA[i];
      for (int64_t n = 0; n < nboxes; n++) {
        int64_t rm_i = i * nboxes + n;
        Matrix& A_i = lis[rm_i];
        int64_t len = A_i.M * A_i.N;
        maxpby(A_i, rm_v + offset, 1., 0.);
        offset = offset + len;
      }
    }
  }

  for (int64_t i = 0; i < ngbs_len; i++)
    free(DATA[i]);
}


void nbd::DistributeDims(int64_t dims[], int64_t level) {
  int64_t my_ind, my_rank, nboxes, *ngbs, ngbs_len;
  locateCOMM(level, &my_ind, &my_rank, &nboxes, &ngbs, &ngbs_len);
  std::vector<MPI_Request> requests(ngbs_len);

  const int64_t* my_data = &dims[my_ind * nboxes];

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank)
      MPI_Isend(my_data, (int)nboxes, MPI_INT64_T, (int)rm_rank, 0, MPI_COMM_WORLD, &requests[i]);
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank) {
      int64_t* data = &dims[i * nboxes];
      MPI_Recv(data, (int)nboxes, MPI_INT64_T, (int)rm_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank)
      MPI_Wait(&requests[i], MPI_STATUS_IGNORE);
  }
}

void constructCOMM_AXAT(double** DATA, int64_t* LEN, int64_t RM_BOX, const Matrices& A, const CSC& rels) {
  int64_t nboxes = rels.N;
  std::vector<int64_t> lens(nboxes);
  std::fill(lens.begin(), lens.end(), 0);

  for (int64_t i = 0; i < rels.N; i++)
    for (int64_t ji = rels.CSC_COLS[i]; ji < rels.CSC_COLS[i + 1]; ji++) {
      int64_t j = rels.CSC_ROWS[ji];
      int64_t rm_box = j / nboxes;
      if (rm_box == RM_BOX) {
        const Matrix& A_ji = A[ji];
        int64_t len = A_ji.M * A_ji.N;
        int64_t box_i = j - rm_box * nboxes;
        lens[box_i] = lens[box_i] + len;
      }
    }

  std::vector<int64_t> offsets(nboxes + 1);
  offsets[0] = 0;
  for (int64_t i = 1; i <= nboxes; i++)
    offsets[i] = offsets[i - 1] + lens[i - 1];

  int64_t tot_len = offsets[nboxes];
  double* data = (double*)malloc(sizeof(double) * tot_len);
  
  for (int64_t i = 0; i < rels.N; i++)
    for (int64_t ji = rels.CSC_COLS[i]; ji < rels.CSC_COLS[i + 1]; ji++) {
      int64_t j = rels.CSC_ROWS[ji];
      int64_t rm_box = j / nboxes;
      if (rm_box == RM_BOX) {
        const Matrix& A_ji = A[ji];
        int64_t len = A_ji.M * A_ji.N;
        int64_t box_i = j - rm_box * nboxes;
        double* tar = data + offsets[box_i];
        cpyFromMatrix('T', A_ji, tar);
        offsets[box_i] = offsets[box_i] + len;
      }
    }

  *LEN = tot_len;
  *DATA = data;
}

void axRemoteV(int64_t RM_BOX, Matrices& A, const CSC& rels, const double* rmv) {
  int64_t nboxes = rels.N;
  int64_t offset = 0;
  for (int64_t i = 0; i < rels.N; i++)
    for (int64_t ji = rels.CSC_COLS[i]; ji < rels.CSC_COLS[i + 1]; ji++) {
      int64_t j = rels.CSC_ROWS[ji];
      int64_t rm_box = j / nboxes;
      if (rm_box == RM_BOX) {
        Matrix& A_ji = A[ji];
        int64_t len = A_ji.M * A_ji.N;
        const double* rmv_i = rmv + offset;
        maxpby(A_ji, rmv_i, 1., 1.);
        offset = offset + len;
      }
    }
}

void nbd::axatDistribute(Matrices& A, const CSC& rels, int64_t level) {
  int64_t my_ind, my_rank, nboxes, *ngbs, ngbs_len;
  locateCOMM(level, &my_ind, &my_rank, &nboxes, &ngbs, &ngbs_len);
  std::vector<MPI_Request> requests(ngbs_len);

  std::vector<int64_t> LENS(ngbs_len);
  std::vector<double*> SRC_DATA(ngbs_len);
  std::vector<double*> RM_DATA(ngbs_len);

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    int64_t rm_box = level < MPI_LEVELS ? (rm_rank >> (MPI_LEVELS - level)) : rm_rank;
    if (rm_rank != my_rank) {
      constructCOMM_AXAT(&SRC_DATA[i], &LENS[i], rm_box, A, rels);
      RM_DATA[i] = (double*)malloc(sizeof(double) * LENS[i]);
    }
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank) {
      const double* my_data = SRC_DATA[i];
      int64_t my_len = LENS[i];
      MPI_Isend(my_data, (int)my_len, MPI_DOUBLE, (int)rm_rank, 0, MPI_COMM_WORLD, &requests[i]);
    }
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank) {
      double* data = RM_DATA[i];
      int64_t len = LENS[i];
      MPI_Recv(data, (int)len, MPI_DOUBLE, (int)rm_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank)
      MPI_Wait(&requests[i], MPI_STATUS_IGNORE);
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    int64_t rm_box = level < MPI_LEVELS ? (rm_rank >> (MPI_LEVELS - level)) : rm_rank;
    if (rm_rank != my_rank)
      axRemoteV(rm_box, A, rels, RM_DATA[i]);
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank) {
      free(SRC_DATA[i]);
      free(RM_DATA[i]);
    }
  }
}


void nbd::butterflySumA(Matrices& A, int64_t level) {
  int64_t my_ind, my_rank, my_twi, rm_rank;
  locateButterflyCOMM(level, &my_ind, &my_rank, &my_twi, &rm_rank);

  MPI_Request request;
  int64_t LEN = 0;
  int64_t alen = A.size();
  double* SRC_DATA, *RM_DATA;

  for (int64_t i = 0; i < alen; i++) {
    const Matrix& A_i = A[i];
    int64_t len = A_i.M * A_i.N;
    LEN = LEN + len;
  }

  SRC_DATA = (double*)malloc(sizeof(double) * LEN);
  RM_DATA = (double*)malloc(sizeof(double) * LEN);

  int64_t offset = 0;
  for (int64_t i = 0; i < alen; i++) {
    const Matrix& A_i = A[i];
    int64_t len = A_i.M * A_i.N;
    cpyFromMatrix('N', A_i, SRC_DATA + offset);
    offset = offset + len;
  }

  MPI_Isend(SRC_DATA, (int)LEN, MPI_DOUBLE, (int)rm_rank, 0, MPI_COMM_WORLD, &request);
  MPI_Recv(RM_DATA, (int)LEN, MPI_DOUBLE, (int)rm_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Wait(&request, MPI_STATUS_IGNORE);

  offset = 0;
  for (int64_t i = 0; i < alen; i++) {
    Matrix& A_i = A[i];
    int64_t len = A_i.M * A_i.N;
    maxpby(A_i, RM_DATA + offset, 1., 1.);
    offset = offset + len;
  }

  free(SRC_DATA);
  free(RM_DATA);
}

void nbd::sendFwSubstituted(const Vectors& X, int64_t level) {
  int64_t my_ind, my_rank, nboxes, *ngbs, ngbs_len;
  locateCOMM(level, &my_ind, &my_rank, &nboxes, &ngbs, &ngbs_len);

  std::vector<MPI_Request> requests(ngbs_len);
  std::vector<double*> DATA(ngbs_len);
  std::vector<int64_t> LENS(ngbs_len);

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank > my_rank) {
      int64_t len_i = 0;
      int64_t ibegin = i * nboxes;
      int64_t iend = ibegin + nboxes;
      for (int64_t n = ibegin; n < iend; n++)
        len_i = len_i + X[n].N;
      
      LENS[i] = len_i;
      DATA[i] = (double*)malloc(sizeof(double) * len_i);

      len_i = 0;
      for (int64_t n = ibegin; n < iend; n++) {
        cpyFromVector(X[n], DATA[i] + len_i);
        len_i = len_i + X[n].N;
      }
      MPI_Isend(DATA[i], (int)LENS[i], MPI_DOUBLE, (int)rm_rank, 0, MPI_COMM_WORLD, &requests[i]);
    }
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank > my_rank)
      MPI_Wait(&requests[i], MPI_STATUS_IGNORE);
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank > my_rank)
      free(DATA[i]);
  }
}

void nbd::sendBkSubstituted(const Vectors& X, int64_t level) {
  int64_t my_ind, my_rank, nboxes, *ngbs, ngbs_len;
  locateCOMM(level, &my_ind, &my_rank, &nboxes, &ngbs, &ngbs_len);
  int64_t LEN = 0;

  int64_t lbegin = my_ind * nboxes;
  int64_t lend = lbegin + nboxes;
  for (int64_t i = lbegin; i < lend; i++)
    LEN = LEN + X[i].N;
  double* DATA = (double*)malloc(sizeof(double) * LEN);

  LEN = 0;
  for (int64_t i = lbegin; i < lend; i++) {
    cpyFromVector(X[i], DATA + LEN);
    LEN = LEN + X[i].N;
  }
  
  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank < my_rank)
      MPI_Send(DATA, (int)LEN, MPI_DOUBLE, (int)rm_rank, 0, MPI_COMM_WORLD);
  }

  free(DATA);
}

void nbd::recvFwSubstituted(Vectors& X, int64_t level) {
  int64_t my_ind, my_rank, nboxes, *ngbs, ngbs_len;
  locateCOMM(level, &my_ind, &my_rank, &nboxes, &ngbs, &ngbs_len);
  int64_t LEN = 0;

  int64_t lbegin = my_ind * nboxes;
  int64_t lend = lbegin + nboxes;
  for (int64_t i = lbegin; i < lend; i++)
    LEN = LEN + X[i].N;
  double* DATA = (double*)malloc(sizeof(double) * LEN);
  
  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank < my_rank) {
      MPI_Recv(DATA, (int)LEN, MPI_DOUBLE, (int)rm_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      int64_t offset = 0;
      for (int64_t i = lbegin; i < lend; i++) {
        vaxpby(X[i], DATA + offset, 1., 1.);
        offset = offset + X[i].N;
      }
    }
  }

  free(DATA);
}

void nbd::recvBkSubstituted(Vectors& X, int64_t level) {
  int64_t my_ind, my_rank, nboxes, *ngbs, ngbs_len;
  locateCOMM(level, &my_ind, &my_rank, &nboxes, &ngbs, &ngbs_len);

  std::vector<MPI_Request> requests(ngbs_len);
  std::vector<double*> DATA(ngbs_len);
  std::vector<int64_t> LENS(ngbs_len);

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank > my_rank) {
      int64_t len_i = 0;
      int64_t ibegin = i * nboxes;
      int64_t iend = ibegin + nboxes;
      for (int64_t n = ibegin; n < iend; n++)
        len_i = len_i + X[n].N;
      
      LENS[i] = len_i;
      DATA[i] = (double*)malloc(sizeof(double) * len_i);
    }
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank > my_rank)
      MPI_Irecv(DATA[i], (int)LENS[i], MPI_DOUBLE, (int)rm_rank, 0, MPI_COMM_WORLD, &requests[i]);
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank > my_rank)
      MPI_Wait(&requests[i], MPI_STATUS_IGNORE);
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank > my_rank) {
      int64_t len_i = 0;
      int64_t ibegin = i * nboxes;
      int64_t iend = ibegin + nboxes;
      for (int64_t n = ibegin; n < iend; n++) {
        vaxpby(X[n], DATA[i] + len_i, 1., 0.);
        len_i = len_i + X[n].N;
      }
      free(DATA[i]);
    }
  }
}

void nbd::distributeSubstituted(Vectors& X, int64_t level) {
  int64_t my_ind, my_rank, nboxes, *ngbs, ngbs_len;
  locateCOMM(level, &my_ind, &my_rank, &nboxes, &ngbs, &ngbs_len);
  std::vector<MPI_Request> requests(ngbs_len);

  std::vector<int64_t> LENS(ngbs_len);
  std::vector<double*> SRC_DATA(ngbs_len);
  std::vector<double*> RM_DATA(ngbs_len);

  int64_t LEN = 0;
  int64_t lbegin = my_ind * nboxes;
  int64_t lend = lbegin + nboxes;
  for (int64_t i = lbegin; i < lend; i++)
    LEN = LEN + X[i].N;

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank) {
      int64_t len_i = 0;
      int64_t ibegin = i * nboxes;
      int64_t iend = ibegin + nboxes;
      for (int64_t n = ibegin; n < iend; n++)
        len_i = len_i + X[n].N;
      
      LENS[i] = len_i;
      SRC_DATA[i] = (double*)malloc(sizeof(double) * len_i);
      RM_DATA[i] = (double*)malloc(sizeof(double) * LEN);

      len_i = 0;
      for (int64_t n = ibegin; n < iend; n++) {
        cpyFromVector(X[n], SRC_DATA[i] + len_i);
        len_i = len_i + X[n].N;
      }
    }
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank) {
      const double* my_data = SRC_DATA[i];
      int64_t my_len = LENS[i];
      MPI_Isend(my_data, (int)my_len, MPI_DOUBLE, (int)rm_rank, 0, MPI_COMM_WORLD, &requests[i]);
    }
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank) {
      double* data = RM_DATA[i];
      MPI_Recv(data, (int)LEN, MPI_DOUBLE, (int)rm_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank)
      MPI_Wait(&requests[i], MPI_STATUS_IGNORE);
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank) {
      int64_t offset = 0;
      const double* data = RM_DATA[i];
      for (int64_t n = lbegin; n < lend; n++) {
        vaxpby(X[n], data + offset, 1., 1.);
        offset = offset + X[n].N;
      }
    }
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank) {
      free(SRC_DATA[i]);
      free(RM_DATA[i]);
    }
  }
}

void nbd::butterflySumX(Vectors& X, int64_t level) {
  int64_t my_ind, my_rank, my_twi, rm_rank;
  locateButterflyCOMM(level, &my_ind, &my_rank, &my_twi, &rm_rank);

  int64_t xlen = X.size();
  MPI_Request request;
  int64_t LEN = 0;
  double* SRC_DATA, *RM_DATA;

  for (int64_t i = 0; i < xlen; i++)
    LEN = LEN + X[i].N;

  SRC_DATA = (double*)malloc(sizeof(double) * LEN);
  RM_DATA = (double*)malloc(sizeof(double) * LEN);

  int64_t offset = 0;
  for (int64_t i = 0; i < xlen; i++) {
    cpyFromVector(X[i], SRC_DATA + offset);
    offset = offset + X[i].N;
  }

  MPI_Isend(SRC_DATA, (int)LEN, MPI_DOUBLE, (int)rm_rank, 0, MPI_COMM_WORLD, &request);
  MPI_Recv(RM_DATA, (int)LEN, MPI_DOUBLE, (int)rm_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Wait(&request, MPI_STATUS_IGNORE);

  offset = 0;
  for (int64_t i = 0; i < xlen; i++) {
    vaxpby(X[i], RM_DATA + offset, 1., 1.);
    offset = offset + X[i].N;
  }

  free(SRC_DATA);
  free(RM_DATA);
}

void nbd::startTimer(double* wtime) {
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0)
    *wtime = MPI_Wtime();
}

void nbd::stopTimer(double wtime, const char str[]) {
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0) {
    double etime = MPI_Wtime();
    printf("%-20s : %f s\n", str, etime - wtime);
  }
}

/*void nbd::DistributeBodies(LocalBodies& bodies, int64_t level) {
  int64_t my_ind, my_rank, nboxes, *ngbs, ngbs_len;
  locateCOMM(level, &my_ind, &my_rank, &nboxes, &ngbs, &ngbs_len);
  int64_t dim = bodies.DIM;

  int64_t my_nbody = bodies.NBODIES[my_ind] * dim;
  int64_t my_offset = bodies.OFFSETS[my_ind * nboxes] * dim;
  const double* my_bodies = &bodies.BODIES[my_offset];
  const int64_t* my_lens = &bodies.LENS[my_ind * nboxes];

  std::vector<MPI_Request> requests1(ngbs_len);
  std::vector<MPI_Request> requests2(ngbs_len);

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank) {
      MPI_Isend(my_bodies, my_nbody, MPI_DOUBLE, rm_rank, 0, MPI_COMM_WORLD, &requests1[i]);
      MPI_Isend(my_lens, nboxes, MPI_INT64_T, rm_rank, 0, MPI_COMM_WORLD, &requests2[i]);
    }
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank) {
      int64_t rm_nbody = bodies.NBODIES[i] * dim;
      int64_t rm_offset = bodies.OFFSETS[i * nboxes] * dim;
      double* rm_bodies = &bodies.BODIES[rm_offset];
      int64_t* rm_lens = &bodies.LENS[i * nboxes];
      
      MPI_Recv(rm_bodies, rm_nbody, MPI_DOUBLE, rm_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(rm_lens, nboxes, MPI_INT64_T, rm_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank) {
      MPI_Wait(&requests1[i], MPI_STATUS_IGNORE);
      MPI_Wait(&requests2[i], MPI_STATUS_IGNORE);
    }
  }

  bodies.OFFSETS[0] = 0;
  for (int64_t i = 1; i < bodies.OFFSETS.size(); i++)
    bodies.OFFSETS[i] = bodies.OFFSETS[i - 1] + bodies.LENS[i - 1];
}*/
