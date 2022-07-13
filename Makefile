
CC	= mpicc
CXX	= mpicxx

CCFLAGS	= -std=c99 -O3 -m64 -Wall -Wextra -I.
CXXFLAGS	= -std=c++11 -O3 -m64 -Wall -Wextra -I.
LDFLAGS	= -lpthread -lm -ldl

ifneq (${MKLROOT},)
	MKLFLAG	= -DUSE_MKL
	CCFLAGS	+=  -I"${MKLROOT}/include"
	CXXFLAGS	+=  -I"${MKLROOT}/include"
	LDFLAGS	+= -fopenmp -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core 
	MSG	= *** Successfully found and linking with Intel MKL! ***
else
	LDFLAGS	+= -lblas -llapacke
	MSG	= *** ENV MKLROOT not set, linking with netlib BLAS. ***
endif

all: lorasp
	$(info $(MSG))

linalg: linalg.c
	$(CC) $(CCFLAGS) $(MKLFLAG) -c linalg.c

kernel: kernel.c
	$(CC) $(CCFLAGS) -c kernel.c

build_tree: build_tree.c
	$(CC) $(CCFLAGS) -c build_tree.c

umv: umv.c
	$(CC) $(CCFLAGS) -c umv.c

dist: dist.cxx dist.h
	$(CXX) $(CXXFLAGS) -c dist.cxx

lib: linalg kernel build_tree umv dist
	ar rcs libnbd.a linalg.o kernel.o build_tree.o umv.o dist.o

lorasp: lorasp.cxx lib
	$(CXX) $(CXXFLAGS) -o lorasp lorasp.cxx -L. -lnbd $(LDFLAGS)

clean:
	rm -f *.o *.a a.out lorasp
