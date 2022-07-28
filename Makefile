
CC = mpicc -std=c99 -O3 -m64 -Wall -Wextra -I. -I"${MKLROOT}/include" -fopenmp -D_PROF -D_MKL
LDFLAGS = -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -ldl

all:
	mkdir -p build
	$(CC) linalg.c kernel.c build_tree.c basis.c umv.c profile.c \
	lorasp.c $(LDFLAGS) -o build/lorasp

clean:
	rm -rf build
