
#include "dist.h"

#include "mpi.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

struct Communicator {
  int64_t SELF_I;
  int64_t TWIN_I;
  int64_t MY_RANK;
  int64_t TWIN_RANK;
  int64_t N_BOXES;

  std::vector<int64_t> NGB;
  std::vector<int64_t> NGB_COMM;
};

int64_t MPI_RANK = 0;
int64_t MPI_SIZE = 1;
int64_t MPI_LEVELS = 0;
std::vector<Communicator> COMMS;

double prog_time = 0.;
double tot_time = 0.;

void initComm(int* argc, char** argv[]) {
  MPI_Init(argc, argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_RANK = rank;
  MPI_SIZE = size;
  MPI_LEVELS = (int64_t)std::log2(size);

  prog_time = MPI_Wtime();
  tot_time = 0.;
}

void closeComm() {
  if (MPI_RANK == 0) {
    double end = MPI_Wtime();
    double prog = end - prog_time;
    printf("Program: %f s. COMM: %f s.\n", prog, tot_time);
  }

  int64_t len = COMMS.size();
  for (int64_t i = 0; i < len; i++) {
    COMMS[i].NGB.clear();
    COMMS[i].NGB_COMM.clear();
  }
  COMMS.clear();
  MPI_Finalize();
}

void commRank(int64_t* mpi_rank, int64_t* mpi_level) {
  if (mpi_rank)
    *mpi_rank = MPI_RANK;
  if (mpi_level)
    *mpi_level = MPI_LEVELS;
}

void butterflyComm(int* comm_needed, int64_t level) {
  *comm_needed = (int)(level > 0 && level <= MPI_LEVELS);
}

void configureComm(int64_t level, const int64_t ngbs[], int64_t ngbs_len) {
  int64_t csize = COMMS.size();
  if (level >= csize) {
    COMMS.resize(level + 1);
  }
  if (ngbs) {
    int64_t lvl_diff = level < MPI_LEVELS ? MPI_LEVELS - level : 0;
    int64_t my_rank = MPI_RANK >> lvl_diff;
    Communicator& gi = COMMS[level];

    int64_t self = std::distance(ngbs, std::find(ngbs, ngbs + ngbs_len, my_rank));
    if (self < ngbs_len) {
      int64_t my_twin = my_rank ^ (int64_t)1;
      int64_t mask = MPI_RANK - (my_rank << lvl_diff);
      int64_t twi = std::distance(ngbs, std::find(ngbs, ngbs + ngbs_len, my_twin));
      gi.SELF_I = self;
      gi.TWIN_I = twi == ngbs_len ? -1 : twi;
      gi.NGB.resize(ngbs_len);
      gi.NGB_COMM.resize(ngbs_len);
      for (int64_t i = 0; i < ngbs_len; i++) {
        gi.NGB[i] = ngbs[i];
        gi.NGB_COMM[i] = (ngbs[i] << lvl_diff) | mask;
      }

      gi.MY_RANK = gi.NGB_COMM[self];
      gi.TWIN_RANK = gi.TWIN_I >= 0 ? gi.NGB_COMM[twi] : -1;
      lvl_diff = level - MPI_LEVELS;
      gi.N_BOXES = lvl_diff > 0 ? ((int64_t)1 << lvl_diff) : 1;
    }
  }
}

void selfLocalRange(int64_t* ibegin, int64_t* iend, int64_t level) {
  if (level >= 0) {
    const Communicator& gi = COMMS[level];
    int64_t boxes = gi.N_BOXES;
    *ibegin = gi.SELF_I * boxes;
    *iend = gi.SELF_I * boxes + boxes;
  }
}

void iLocal(int64_t* ilocal, int64_t iglobal, int64_t level) {
  if (level >= 0) {
    const Communicator& gi = COMMS[level];
    int64_t boxes = gi.N_BOXES;
    const int64_t* list = gi.NGB.data();
    int64_t list_len = gi.NGB.size();
    int64_t box_rank = iglobal / boxes;
    int64_t i_rank = iglobal - box_rank * boxes;
    int64_t box_ind = std::distance(list, std::find(list, list + list_len, box_rank));
    
    *ilocal = (box_ind < list_len) ? (box_ind * boxes + i_rank) : -1;
  }
}

void iGlobal(int64_t* iglobal, int64_t ilocal, int64_t level) {
  if (level >= 0) {
    const Communicator& gi = COMMS[level];
    int64_t boxes = gi.N_BOXES;
    const int64_t* list = gi.NGB.data();
    int64_t list_len = gi.NGB.size();
    int64_t box_ind = ilocal / boxes;
    int64_t i_rank = ilocal - box_ind * boxes;

    *iglobal = (box_ind < list_len) ? (list[box_ind] * boxes + i_rank) : -1;
  }
}

void contentLength(int64_t* len, int64_t level) {
  if (level >= 0) {
    const Communicator& gi = COMMS[level];
    int64_t boxes = gi.N_BOXES;
    int64_t list_len = gi.NGB.size();
    *len = boxes * list_len;
  }
}

void DistributeVectorsList(Vector lis[], int64_t level) {
  int64_t my_ind = COMMS[level].SELF_I;
  int64_t my_rank = COMMS[level].MY_RANK;
  int64_t nboxes = COMMS[level].N_BOXES;
  int64_t *ngbs = COMMS[level].NGB_COMM.data();
  int64_t ngbs_len = COMMS[level].NGB_COMM.size();
  std::vector<MPI_Request> requests(ngbs_len);

  std::vector<int64_t> LENS(ngbs_len);
  std::vector<double*> DATA(ngbs_len);

  for (int64_t i = 0; i < ngbs_len; i++) {
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
    cpyFromVector(&B_i, my_data + offset);
    offset = offset + len;
  }

  int tag = 0;
  double stime = MPI_Wtime();
  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank)
      MPI_Isend(my_data, (int)my_len, MPI_DOUBLE, (int)rm_rank, tag, MPI_COMM_WORLD, &requests[i]);
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank) {
      double* data = DATA[i];
      int64_t len = LENS[i];
      MPI_Recv(data, (int)len, MPI_DOUBLE, (int)rm_rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank)
      MPI_Wait(&requests[i], MPI_STATUS_IGNORE);
  }
  double etime = MPI_Wtime() - stime;
  tot_time = tot_time + etime;

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank) {
      offset = 0;
      const double* rm_v = DATA[i];
      for (int64_t n = 0; n < nboxes; n++) {
        int64_t rm_i = i * nboxes + n;
        Vector& B_i = lis[rm_i];
        int64_t len = B_i.N;
        vaxpby(&B_i, &rm_v[offset], 1., 0.);
        offset = offset + len;
      }
    }
  }

  for (int64_t i = 0; i < ngbs_len; i++)
    free(DATA[i]);
}

void DistributeMatricesList(Matrix lis[], int64_t level) {
  int64_t my_ind = COMMS[level].SELF_I;
  int64_t my_rank = COMMS[level].MY_RANK;
  int64_t nboxes = COMMS[level].N_BOXES;
  int64_t *ngbs = COMMS[level].NGB_COMM.data();
  int64_t ngbs_len = COMMS[level].NGB_COMM.size();
  std::vector<MPI_Request> requests(ngbs_len);

  std::vector<int64_t> LENS(ngbs_len);
  std::vector<double*> DATA(ngbs_len);

  for (int64_t i = 0; i < ngbs_len; i++) {
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
    cpyFromMatrix(&A_i, my_data + offset);
    offset = offset + len;
  }

  int tag = 1;
  double stime = MPI_Wtime();
  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank)
      MPI_Isend(my_data, (int)my_len, MPI_DOUBLE, (int)rm_rank, tag, MPI_COMM_WORLD, &requests[i]);
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank) {
      double* data = DATA[i];
      int64_t len = LENS[i];
      MPI_Recv(data, (int)len, MPI_DOUBLE, (int)rm_rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank)
      MPI_Wait(&requests[i], MPI_STATUS_IGNORE);
  }
  double etime = MPI_Wtime() - stime;
  tot_time = tot_time + etime;

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank) {
      offset = 0;
      const double* rm_v = DATA[i];
      for (int64_t n = 0; n < nboxes; n++) {
        int64_t rm_i = i * nboxes + n;
        Matrix& A_i = lis[rm_i];
        int64_t len = A_i.M * A_i.N;
        maxpby(&A_i, rm_v + offset, 1., 0.);
        offset = offset + len;
      }
    }
  }

  for (int64_t i = 0; i < ngbs_len; i++)
    free(DATA[i]);
}


void DistributeDims(int64_t dims[], int64_t level) {
  int64_t my_ind = COMMS[level].SELF_I;
  int64_t my_rank = COMMS[level].MY_RANK;
  int64_t nboxes = COMMS[level].N_BOXES;
  int64_t *ngbs = COMMS[level].NGB_COMM.data();
  int64_t ngbs_len = COMMS[level].NGB_COMM.size();
  std::vector<MPI_Request> requests(ngbs_len);

  const int64_t* my_data = &dims[my_ind * nboxes];
  int tag = 2;
  double stime = MPI_Wtime();
  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank)
      MPI_Isend(my_data, (int)nboxes, MPI_INT64_T, (int)rm_rank, tag, MPI_COMM_WORLD, &requests[i]);
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank) {
      int64_t* data = &dims[i * nboxes];
      MPI_Recv(data, (int)nboxes, MPI_INT64_T, (int)rm_rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank)
      MPI_Wait(&requests[i], MPI_STATUS_IGNORE);
  }
  double etime = MPI_Wtime() - stime;
  tot_time = tot_time + etime;
}

void DistributeMultipoles(int64_t multipoles[], const int64_t dims[], int64_t level) {
  int64_t my_ind = COMMS[level].SELF_I;
  int64_t my_rank = COMMS[level].MY_RANK;
  int64_t nboxes = COMMS[level].N_BOXES;
  int64_t *ngbs = COMMS[level].NGB_COMM.data();
  int64_t ngbs_len = COMMS[level].NGB_COMM.size();
  std::vector<int64_t> sizes(ngbs_len);
  std::vector<int64_t> offsets(ngbs_len);

  int64_t tot_len = 0;
  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t len = 0;
    for (int64_t n = 0; n < nboxes; n++) {
      int64_t rm_i = i * nboxes + n;
      len = len + dims[rm_i];
    }
    sizes[i] = len;
    offsets[i] = tot_len;
    tot_len = tot_len + len;
  }

  int64_t my_len = sizes[my_ind];
  int64_t my_offset = offsets[my_ind];
  const int64_t* my_mps = &multipoles[my_offset];

  std::vector<MPI_Request> requests(ngbs_len);
  int tag = 3;
  double stime = MPI_Wtime();
  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank)
      MPI_Isend(my_mps, (int)my_len, MPI_INT64_T, (int)rm_rank, tag, MPI_COMM_WORLD, &requests[i]);
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank) {
      int64_t rm_len = sizes[i];
      int64_t rm_offset = offsets[i];
      int64_t* rm_mps = &multipoles[rm_offset];

      MPI_Recv(rm_mps, (int)rm_len , MPI_INT64_T, (int)rm_rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank)
      MPI_Wait(&requests[i], MPI_STATUS_IGNORE);
  }
  double etime = MPI_Wtime() - stime;
  tot_time = tot_time + etime;
}

void butterflyUpdateDims(int64_t my_dim, int64_t* rm_dim, int64_t level) {
  int64_t rm_rank = COMMS[level].TWIN_RANK;
  MPI_Request request;

  double stime = MPI_Wtime();
  int tag = 4;
  MPI_Isend(&my_dim, 1, MPI_INT64_T, (int)rm_rank, tag, MPI_COMM_WORLD, &request);
  MPI_Recv(rm_dim, 1, MPI_INT64_T, (int)rm_rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Wait(&request, MPI_STATUS_IGNORE);
  double etime = MPI_Wtime() - stime;
  tot_time = tot_time + etime;
}

void butterflyUpdateMultipoles(const int64_t multipoles[], int64_t my_dim, int64_t rm[], int64_t rm_dim, int64_t level) {
  int64_t rm_rank = COMMS[level].TWIN_RANK;
  MPI_Request request;

  double stime = MPI_Wtime();
  int tag = 5;
  MPI_Isend(multipoles, (int)my_dim, MPI_INT64_T, (int)rm_rank, tag, MPI_COMM_WORLD, &request);
  MPI_Recv(rm, (int)rm_dim, MPI_INT64_T, (int)rm_rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Wait(&request, MPI_STATUS_IGNORE);
  double etime = MPI_Wtime() - stime;
  tot_time = tot_time + etime;
}


void butterflySumA(Matrix A[], int64_t lenA, int64_t level) {
  int64_t rm_rank = COMMS[level].TWIN_RANK;

  MPI_Request request;
  int64_t LEN = 0;
  double* SRC_DATA, *RM_DATA;

  for (int64_t i = 0; i < lenA; i++) {
    const Matrix& A_i = A[i];
    int64_t len = A_i.M * A_i.N;
    LEN = LEN + len;
  }

  SRC_DATA = (double*)malloc(sizeof(double) * LEN);
  RM_DATA = (double*)malloc(sizeof(double) * LEN);

  int64_t offset = 0;
  for (int64_t i = 0; i < lenA; i++) {
    const Matrix& A_i = A[i];
    int64_t len = A_i.M * A_i.N;
    cpyFromMatrix(&A_i, SRC_DATA + offset);
    offset = offset + len;
  }

  double stime = MPI_Wtime();
  int tag = 6;
  MPI_Isend(SRC_DATA, (int)LEN, MPI_DOUBLE, (int)rm_rank, tag, MPI_COMM_WORLD, &request);
  MPI_Recv(RM_DATA, (int)LEN, MPI_DOUBLE, (int)rm_rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Wait(&request, MPI_STATUS_IGNORE);
  double etime = MPI_Wtime() - stime;
  tot_time = tot_time + etime;

  offset = 0;
  for (int64_t i = 0; i < lenA; i++) {
    Matrix& A_i = A[i];
    int64_t len = A_i.M * A_i.N;
    maxpby(&A_i, RM_DATA + offset, 1., 1.);
    offset = offset + len;
  }

  free(SRC_DATA);
  free(RM_DATA);
}

void sendFwSubstituted(const Vector X[], int64_t level) {
  int64_t my_rank = COMMS[level].MY_RANK;
  int64_t nboxes = COMMS[level].N_BOXES;
  int64_t *ngbs = COMMS[level].NGB_COMM.data();
  int64_t ngbs_len = COMMS[level].NGB_COMM.size();

  std::vector<MPI_Request> requests(ngbs_len);
  std::vector<double*> DATA(ngbs_len);
  std::vector<int64_t> LENS(ngbs_len);

  int tag = 7;
  double stime = MPI_Wtime();
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
        cpyFromVector(&X[n], DATA[i] + len_i);
        len_i = len_i + X[n].N;
      }
      MPI_Isend(DATA[i], (int)LENS[i], MPI_DOUBLE, (int)rm_rank, tag, MPI_COMM_WORLD, &requests[i]);
    }
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank > my_rank)
      MPI_Wait(&requests[i], MPI_STATUS_IGNORE);
  }
  double etime = MPI_Wtime() - stime;
  tot_time = tot_time + etime;

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank > my_rank)
      free(DATA[i]);
  }
}

void sendBkSubstituted(const Vector X[], int64_t level) {
  int64_t my_ind = COMMS[level].SELF_I;
  int64_t my_rank = COMMS[level].MY_RANK;
  int64_t nboxes = COMMS[level].N_BOXES;
  int64_t *ngbs = COMMS[level].NGB_COMM.data();
  int64_t ngbs_len = COMMS[level].NGB_COMM.size();
  int64_t LEN = 0;

  int64_t lbegin = my_ind * nboxes;
  int64_t lend = lbegin + nboxes;
  for (int64_t i = lbegin; i < lend; i++)
    LEN = LEN + X[i].N;
  double* DATA = (double*)malloc(sizeof(double) * LEN);

  LEN = 0;
  for (int64_t i = lbegin; i < lend; i++) {
    cpyFromVector(&X[i], DATA + LEN);
    LEN = LEN + X[i].N;
  }
  
  int tag = 8;
  double stime = MPI_Wtime();
  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank < my_rank)
      MPI_Send(DATA, (int)LEN, MPI_DOUBLE, (int)rm_rank, tag, MPI_COMM_WORLD);
  }
  double etime = MPI_Wtime() - stime;
  tot_time = tot_time + etime;

  free(DATA);
}

void recvFwSubstituted(Vector X[], int64_t level) {
  int64_t my_ind = COMMS[level].SELF_I;
  int64_t my_rank = COMMS[level].MY_RANK;
  int64_t nboxes = COMMS[level].N_BOXES;
  int64_t *ngbs = COMMS[level].NGB_COMM.data();
  int64_t ngbs_len = COMMS[level].NGB_COMM.size();
  int64_t LEN = 0;

  int64_t lbegin = my_ind * nboxes;
  int64_t lend = lbegin + nboxes;
  for (int64_t i = lbegin; i < lend; i++)
    LEN = LEN + X[i].N;
  double* DATA = (double*)malloc(sizeof(double) * LEN);
  
  int tag = 7;
  double stime = MPI_Wtime();
  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank < my_rank) {
      MPI_Recv(DATA, (int)LEN, MPI_DOUBLE, (int)rm_rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      int64_t offset = 0;
      for (int64_t i = lbegin; i < lend; i++) {
        vaxpby(&X[i], DATA + offset, 1., 1.);
        offset = offset + X[i].N;
      }
    }
  }
  double etime = MPI_Wtime() - stime;
  tot_time = tot_time + etime;

  free(DATA);
}

void recvBkSubstituted(Vector X[], int64_t level) {
  int64_t my_rank = COMMS[level].MY_RANK;
  int64_t nboxes = COMMS[level].N_BOXES;
  int64_t *ngbs = COMMS[level].NGB_COMM.data();
  int64_t ngbs_len = COMMS[level].NGB_COMM.size();

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

  int tag = 8;
  double stime = MPI_Wtime();
  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank > my_rank)
      MPI_Irecv(DATA[i], (int)LENS[i], MPI_DOUBLE, (int)rm_rank, tag, MPI_COMM_WORLD, &requests[i]);
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank > my_rank)
      MPI_Wait(&requests[i], MPI_STATUS_IGNORE);
  }
  double etime = MPI_Wtime() - stime;
  tot_time = tot_time + etime;

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank > my_rank) {
      int64_t len_i = 0;
      int64_t ibegin = i * nboxes;
      int64_t iend = ibegin + nboxes;
      for (int64_t n = ibegin; n < iend; n++) {
        vaxpby(&X[n], DATA[i] + len_i, 1., 0.);
        len_i = len_i + X[n].N;
      }
      free(DATA[i]);
    }
  }
}

void distributeSubstituted(Vector X[], int64_t level) {
  int64_t my_ind = COMMS[level].SELF_I;
  int64_t my_rank = COMMS[level].MY_RANK;
  int64_t nboxes = COMMS[level].N_BOXES;
  int64_t *ngbs = COMMS[level].NGB_COMM.data();
  int64_t ngbs_len = COMMS[level].NGB_COMM.size();
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
        cpyFromVector(&X[n], SRC_DATA[i] + len_i);
        len_i = len_i + X[n].N;
      }
    }
  }

  int tag = 9;
  double stime = MPI_Wtime();
  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank) {
      const double* my_data = SRC_DATA[i];
      int64_t my_len = LENS[i];
      MPI_Isend(my_data, (int)my_len, MPI_DOUBLE, (int)rm_rank, tag, MPI_COMM_WORLD, &requests[i]);
    }
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank) {
      double* data = RM_DATA[i];
      MPI_Recv(data, (int)LEN, MPI_DOUBLE, (int)rm_rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank)
      MPI_Wait(&requests[i], MPI_STATUS_IGNORE);
  }
  double etime = MPI_Wtime() - stime;
  tot_time = tot_time + etime;

  for (int64_t i = 0; i < ngbs_len; i++) {
    int64_t rm_rank = ngbs[i];
    if (rm_rank != my_rank) {
      int64_t offset = 0;
      const double* data = RM_DATA[i];
      for (int64_t n = lbegin; n < lend; n++) {
        vaxpby(&X[n], data + offset, 1., 1.);
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

void butterflySumX(Vector X[], int64_t lenX, int64_t level) {
  int64_t rm_rank = COMMS[level].TWIN_RANK;
  MPI_Request request;
  int64_t LEN = 0;
  double* SRC_DATA, *RM_DATA;

  for (int64_t i = 0; i < lenX; i++)
    LEN = LEN + X[i].N;

  SRC_DATA = (double*)malloc(sizeof(double) * LEN);
  RM_DATA = (double*)malloc(sizeof(double) * LEN);

  int64_t offset = 0;
  for (int64_t i = 0; i < lenX; i++) {
    cpyFromVector(&X[i], SRC_DATA + offset);
    offset = offset + X[i].N;
  }

  int tag = 10;
  double stime = MPI_Wtime();
  MPI_Isend(SRC_DATA, (int)LEN, MPI_DOUBLE, (int)rm_rank, tag, MPI_COMM_WORLD, &request);
  MPI_Recv(RM_DATA, (int)LEN, MPI_DOUBLE, (int)rm_rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Wait(&request, MPI_STATUS_IGNORE);
  double etime = MPI_Wtime() - stime;
  tot_time = tot_time + etime;

  offset = 0;
  for (int64_t i = 0; i < lenX; i++) {
    vaxpby(&X[i], RM_DATA + offset, 1., 1.);
    offset = offset + X[i].N;
  }

  free(SRC_DATA);
  free(RM_DATA);
}

void startTimer(double* wtime, double* cmtime) {
  MPI_Barrier(MPI_COMM_WORLD);
  *wtime = MPI_Wtime();
  *cmtime = tot_time;
}

void stopTimer(double* wtime, double* cmtime) {
  MPI_Barrier(MPI_COMM_WORLD);
  double stime = *wtime;
  double scmtime = *cmtime;
  double etime = MPI_Wtime();
  *wtime = etime - stime;
  *cmtime = tot_time - scmtime;
}
