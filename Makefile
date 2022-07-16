
CC	= mpicc
CXX	= mpicxx

CCFLAGS	= -std=c99 -O3 -m64 -Wall -Wextra -I. -fopenmp 
CXXFLAGS	= -std=c++11 -O3 -m64 -Wall -Wextra -I. -fopenmp 
LDFLAGS	= -lpthread -lm -ldl
PROF_FLAG = -D_PROF

ifneq (${MKLROOT},)
	MKLFLAG	= -DUSE_MKL
	CCFLAGS	+=  -I"${MKLROOT}/include"
	CXXFLAGS	+=  -I"${MKLROOT}/include"
	LDFLAGS	+= -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core 
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

basis: basis.c
	$(CC) $(CCFLAGS) $(PROF_FLAG) -c basis.c

umv: umv.c
	$(CC) $(CCFLAGS) $(PROF_FLAG) -c umv.c

profile: profile.c
	$(CC) $(CCFLAGS) $(PROF_FLAG) -c profile.c

lib: linalg kernel build_tree basis umv profile
	ar rcs libnbd.a linalg.o kernel.o build_tree.o basis.o umv.o profile.o

lorasp: lorasp.cxx lib
	$(CXX) $(CXXFLAGS) -o lorasp lorasp.cxx -L. -lnbd $(LDFLAGS)

clean:
	rm -f *.o *.a a.out lorasp
