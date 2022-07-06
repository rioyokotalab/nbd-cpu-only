
CC	= gcc
CXX	= g++
MPICC	= mpicc
MPICXX	= mpicxx

CCFLAGS	= -std=c99 -O3 -m64 -Wall -Wextra -fopenmp -I.
CXXFLAGS	= -std=c++11 -O3 -m64 -Wall -Wextra -fopenmp -I.
LDFLAGS	= -lpthread -lm -ldl

ifneq (${MKLROOT},)
	MKLFLAG	= -DUSE_MKL
	CCFLAGS	+=  -I"${MKLROOT}/include"
	CXXFLAGS	+=  -I"${MKLROOT}/include"
	LDFLAGS	+= -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.so \
	  ${MKLROOT}/lib/intel64/libmkl_sequential.so \
	  ${MKLROOT}/lib/intel64/libmkl_core.so -Wl,--end-group
	MSG	= *** Successfully found and linking with Intel MKL! ***
else
	LDFLAGS	+= -lblas -llapacke
	MSG	= *** ENV MKLROOT not set, linking with netlib BLAS. ***
endif

all: lorasp
	$(info $(MSG))

linalg: linalg.c linalg.h
	$(CC) $(CCFLAGS) $(MKLFLAG) -c linalg.c

kernel: kernel.c kernel.h
	$(CC) $(CCFLAGS) -c kernel.c

build_tree: build_tree.c build_tree.h
	$(CC) $(CCFLAGS) -c build_tree.c

umv: umv.cxx umv.h
	$(CXX) $(CXXFLAGS) -c umv.cxx

solver: solver.cxx solver.h
	$(CXX) $(CXXFLAGS) -c solver.cxx

dist: dist.cxx dist.h
	$(MPICXX) $(CXXFLAGS) -c dist.cxx

lib: linalg kernel build_tree umv solver dist
	ar rcs libnbd.a linalg.o kernel.o build_tree.o umv.o solver.o dist.o

lorasp: lorasp.cxx lib
	$(CXX) $(CXXFLAGS) -c lorasp.cxx
	$(MPICXX) $(CXXFLAGS) -o lorasp lorasp.o -L. -lnbd $(LDFLAGS)

clean:
	rm -f *.o *.a a.out lorasp
