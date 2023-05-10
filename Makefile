
CXX			= mpicxx -std=c++14
CFLAGS		= -O3 -m64 -Wall -Wextra -I. -I"${MKLROOT}/include" -fopenmp
LDFLAGS 	= -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

NVCC		= ${CUDADIR}/bin/nvcc -ccbin=mpicxx -std=c++14
NVCCFLAGS	= -O3 -m64 -I. -I"${MKLROOT}/include" --Werror=all-warnings -Xcompiler=-fopenmp
NVCCLIBS	= -L${CUDADIR}/lib64 -lcublas -lcudart -lcusolver -lnccl

all:
	$(CXX) $(CFLAGS) -c linalg.cxx -o linalg.o
	$(CXX) $(CFLAGS) -c basis.cxx -o basis.o
	$(CXX) $(CFLAGS) -c build_tree.cxx -o build_tree.o
	$(CXX) $(CFLAGS) -c comm.cxx -o comm.o
	$(CXX) $(CFLAGS) -c umv.cxx -o umv.o
	$(CXX) $(CFLAGS) -c lorasp.cxx -o lorasp.o
	$(CXX) $(CFLAGS) -c eigen.cxx -o eigen.o

	$(CXX) $(CFLAGS) linalg.o basis.o build_tree.o comm.o umv.o \
	  lorasp.o $(LDFLAGS) -o lorasp
	$(CXX) $(CFLAGS) linalg.o basis.o build_tree.o comm.o umv.o \
	  eigen.o $(LDFLAGS) -o eigen

clean:
	rm -rf *.a *.o lorasp eigen
