
CC = mpicc -std=c99 -O3 -m64 -Wall -Wextra -I. -I"${MKLROOT}/include" -fopenmp -D_PROF
LDFLAGS = -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

all:
	mkdir -p build
	$(CC) $(CCFLAGS) $(PROF_FLAG) linalg.c kernel.c build_tree.c basis.c umv.c profile.c \
	lorasp.c $(LDFLAGS) -o build/lorasp

clean:
	rm -rf build
