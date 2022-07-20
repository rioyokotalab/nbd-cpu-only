
CC	= mpicc
CCFLAGS	= -std=c99 -O3 -m64 -Wall -Wextra -I. -fopenmp 
LDFLAGS	= -lpthread -lm -ldl
PROF_FLAG = -D_PROF

ifneq (${MKLROOT},)
	MKLFLAG	= -DUSE_MKL
	CCFLAGS	+=  -I"${MKLROOT}/include"
	LDFLAGS	+= -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core 
else ifneq (${OPENBLAS_DIR},)
	CCFLAGS +=	-I"${OPENBLAS_DIR}/include"
	LDFLAGS	+= -L"${OPENBLAS_DIR}/lib" -lopenblas
else
	LDFLAGS += -lblas -llapacke
endif

all:
	mkdir -p build
	$(CC) $(CCFLAGS) $(MKLFLAG) -c linalg.c -o build/linalg.o
	$(CC) $(CCFLAGS) -c kernel.c -o build/kernel.o
	$(CC) $(CCFLAGS) -c build_tree.c -o build/build_tree.o
	$(CC) $(CCFLAGS) $(PROF_FLAG) -c basis.c -o build/basis.o
	$(CC) $(CCFLAGS) $(PROF_FLAG) -c umv.c -o build/umv.o
	$(CC) $(CCFLAGS) $(PROF_FLAG) -c profile.c -o build/profile.o
	ar rcs build/libnbd.a build/linalg.o build/kernel.o build/build_tree.o build/basis.o build/umv.o build/profile.o
	$(CC) $(CCFLAGS) lorasp.c -L./build -lnbd $(LDFLAGS) -o build/lorasp

clean:
	rm -rf build
