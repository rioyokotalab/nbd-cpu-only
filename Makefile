
CC	= gcc
CXX	= g++
MPICC	= mpicc
MPICXX	= mpicxx

CCFLAGS	= -std=c99 -O3 -Wall -Wextra -fopenmp -I. -DMKL_ILP64 -I"${MKLROOT}/include"
CXXFLAGS	= -std=c++11 -O3 -Wall -Wextra -fopenmp -I. -DMKL_ILP64 -I"${MKLROOT}/include"

LDFLAGS	= -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.so ${MKLROOT}/lib/intel64/libmkl_sequential.so ${MKLROOT}/lib/intel64/libmkl_core.so -Wl,--end-group -lpthread -lm -ldl

all:
	make lorasp

linalg: linalg.c linalg.h
	$(CC) $(CCFLAGS) -DUSE_MKL -c linalg.c

kernel: kernel.c kernel.h
	$(CC) $(CCFLAGS) -c kernel.c

build_tree: build_tree.cxx build_tree.h
	$(CXX) $(CXXFLAGS) -c build_tree.cxx

basis: basis.cxx basis.h
	$(CXX) $(CXXFLAGS) -c basis.cxx

umv: umv.cxx umv.h
	$(CXX) $(CXXFLAGS) -c umv.cxx

solver: solver.cxx solver.h
	$(CXX) $(CXXFLAGS) -c solver.cxx

dist: dist.cxx dist.h
	$(MPICXX) $(CXXFLAGS) -c dist.cxx

lib: linalg kernel build_tree basis umv solver dist
	ar rcs libnbd.a linalg.o kernel.o build_tree.o basis.o umv.o solver.o dist.o

lorasp: lorasp.cxx lib
	$(CXX) $(CXXFLAGS) -c lorasp.cxx
	$(MPICXX) $(CXXFLAGS) -o lorasp lorasp.o -L. -lnbd $(LDFLAGS)

clean:
	rm -f *.o *.a a.out lorasp
