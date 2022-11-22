
CC 			= mpicc -std=c99
CXX			= mpicxx -std=c++11
CFLAGS		= -O3 -m64 -Wall -Wextra -I. -I"${MKLROOT}/include" -fopenmp
LDFLAGS 	= -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

NVCC		= ${CUDADIR}/bin/nvcc -ccbin=mpicxx
NVCCFLAGS	= -O3 -m64 -I. -I"${MKLROOT}/include" --Werror=all-warnings -Xcompiler=-fopenmp
NVCCLIBS	= -L${CUDADIR}/lib64 -lcublas -lcudart -lcusolver

all:
	$(CC) $(CFLAGS) -c linalg.c -o linalg.o
	$(CC) $(CFLAGS) -c batch.c -o batch.o
	$(CC) $(CFLAGS) -c kernel.c -o kernel.o
	$(CC) $(CFLAGS) -c build_tree.c -o build_tree.o
	$(CXX) $(CFLAGS) -D_PROF -c basis.cxx -o basis.o
	$(CC) $(CFLAGS) -D_PROF -c umv.c -o umv.o
	$(CC) $(CFLAGS) -D_PROF -c profile.c -o profile.o
	$(CC) $(CFLAGS) -c lorasp.c -o lorasp.o
	$(NVCC) $(NVCCFLAGS) -c batch_gpu.cu -o batch_gpu.o

	$(CXX) $(CFLAGS) linalg.o batch.o kernel.o build_tree.o basis.o umv.o profile.o \
	  lorasp.o $(LDFLAGS) -o lorasp
	$(NVCC) $(NVCCFLAGS) linalg.o batch_gpu.o kernel.o build_tree.o basis.o umv.o profile.o \
	  lorasp.o $(LDFLAGS) $(NVCCLIBS) -o gpu
	
clean:
	rm -rf *.a *.o lorasp gpu
