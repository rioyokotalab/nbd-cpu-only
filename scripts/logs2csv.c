
#include "stdio.h"
#include "stdlib.h"

int main(int argc, char* argv[]) {
  FILE* log = fopen(argv[1], "r");
  FILE* csv = fopen(argv[2], "r+");
  if (csv == NULL) {
    csv = fopen(argv[2], "w");
    fprintf(csv, "N,Leaf,Admis,Dim,Nprocs,COMPRESS_TIME,COMPRESS_COMM,FAC_TIME,FAC_COMM,SOLV_TIME,SOLV_COMM,BASIS_MEM,MAT_MEM,VEC_MEM,SOLV_ERR,PROC_TIME,PROC_COMM\n");
  }
  else
    fseek(csv, 0, SEEK_END);
  
  while (log != NULL) {
    int N, leaf, dim, procs;
    double admis, ctime, ctime_comm, ftime, ftime_comm, stime, stime_comm, b_mem, m_mem, v_mem, err, ptime, ptime_comm;

    int ret = fscanf(log, "LORASP: %d,%d,%lf,%d,%d\n\
      Construct: %lf s. COMM: %lf s.\n\
      Factorize: %lf s. COMM: %lf s.\n\
      Solution: %lf s. COMM: %lf s.\n\
      Basis Memory: %lf GiB.\n\
      Matrix Memory: %lf GiB.\n\
      Vector Memory: %lf GiB.\n\
      Err: %lf\n\
      Program: %lf s. COMM: %lf s.\n", 
      &N, &leaf, &admis, &dim, &procs, &ctime, &ctime_comm, &ftime, &ftime_comm, &stime, &stime_comm, &b_mem, &m_mem, &v_mem, &err, &ptime, &ptime_comm);
    
    if (ret != EOF) {
      fprintf(csv, "%d,%d,%lf,%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%e,%lf,%lf\n",
        N, leaf, admis, dim, procs, ctime, ctime_comm, ftime, ftime_comm, stime, stime_comm, b_mem, m_mem, v_mem, err, ptime, ptime_comm);
    }
    else
      break;
  }

  fclose(log);
  fclose(csv);

  return 0;
}
