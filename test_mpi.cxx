#include <mpi.h>
#include <iostream>
#include <string>
#include <unistd.h>

int main(int argc, char **argv) {
  char hostname[256];                                           // Define hostname
  int size,rank,len;                                            // Define MPI size and rank
  MPI_Init(&argc,&argv);                                        // Initialize MPI communicator
  MPI_Comm_size(MPI_COMM_WORLD,&size);                          // Get number of MPI processes
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);                          // Get rank of current MPI process
  MPI_Get_processor_name(hostname,&len);                        // Get hostname
  const std::string Nbody_str = argc > 1 ? std::string(argv[1]) : "1024";
  int64_t Nbody = std::stoll(Nbody_str, nullptr, 10);
  for( int irank=0; irank!=size; ++irank ) {                    // Loop over MPI ranks
    MPI_Barrier(MPI_COMM_WORLD);                                //  Synchronize processes
    if( rank == irank ) {                                       //  If loop counter matches MPI rank
      double* arr = (double*)malloc(sizeof(double) * Nbody * 3);
      arr[0] = 100.; arr[Nbody-1] = 100.;
      std::cout << hostname << " " << rank << " / " << size     // Print hostname, rank, and size
                << ": allocated " << Nbody << " doubles"
                << std::endl;
      free(arr);
    }                                                           //  Endif for loop counter
    usleep(100);                                                //  Wait for 100 microseconds
  }                                                             // End loop over MPI ranks
  MPI_Finalize();                                               // Finalize MPI communicator
}
