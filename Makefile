
CXX			= mpicxx -std=c++14
CFLAGS		= -O3 -m64 -Wall -Wextra -I. -I"${MKLROOT}/include" -fopenmp
LDFLAGS 	= -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

NVCC		= ${CUDADIR}/bin/nvcc -ccbin=mpicxx
NVCCFLAGS	= -O3 -m64 -I. -I"${MKLROOT}/include" --Werror=all-warnings -Xcompiler=-fopenmp
NVCCLIBS	= -L${CUDADIR}/lib64 -lcublas -lcudart -lcusolver -lcurand -lnccl

all:
	$(CXX) $(CFLAGS) -c linalg.cxx -o linalg.o
	$(CXX) $(CFLAGS) -c batch.cxx -o batch.o
	$(CXX) $(CFLAGS) -c kernel.cxx -o kernel.o
	$(CXX) $(CFLAGS) -D_PROF -c build_tree.cxx -o build_tree.o
	$(CXX) $(CFLAGS) -D_PROF -c umv.cxx -o umv.o
	$(CXX) $(CFLAGS) -D_PROF -c profile.cxx -o profile.o
	$(CXX) $(CFLAGS) -c lorasp.cxx -o lorasp.o
	$(NVCC) $(NVCCFLAGS) -c batch_gpu.cu -o batch_gpu.o

	$(CXX) $(CFLAGS) linalg.o batch.o kernel.o build_tree.o umv.o profile.o \
	  lorasp.o $(LDFLAGS) -o lorasp
	$(NVCC) $(NVCCFLAGS) linalg.o batch_gpu.o kernel.o build_tree.o umv.o profile.o \
	  lorasp.o $(LDFLAGS) $(NVCCLIBS) -o gpu
	
clean:
	rm -rf *.a *.o lorasp gpu
