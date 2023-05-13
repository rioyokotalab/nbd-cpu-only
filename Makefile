
CXX			= mpicxx -std=c++14
CFLAGS		= -DUSE_MKL -O3 -m64 -Wall -Wextra -I. -I"${MKLROOT}/include" -fopenmp
LDFLAGS 	= -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
LAPACK_LDFLAGS			= -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed \
											-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
SCALAPACK_LDFLAGS	= -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_scalapack_lp64 \
											-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 \
											-liomp5 -lpthread -lm -ldl

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
	$(CXX) $(CFLAGS) -c pdsyev.cxx -o pdsyev.o
	$(CXX) $(CFLAGS) -c pdsyevx.cxx -o pdsyevx.o
	$(CXX) $(CFLAGS) -c dsyev.cxx -o dsyev.o
	$(CXX) $(CFLAGS) -c dsyevx.cxx -o dsyevx.o

	$(CXX) $(CFLAGS) linalg.o basis.o build_tree.o comm.o umv.o \
	  lorasp.o $(LDFLAGS) -o lorasp
	$(CXX) $(CFLAGS) linalg.o basis.o build_tree.o comm.o umv.o \
	  eigen.o $(LDFLAGS) -o eigen
	$(CXX) $(CFLAGS) pdsyev.o $(SCALAPACK_LDFLAGS) -o pdsyev
	$(CXX) $(CFLAGS) pdsyevx.o $(SCALAPACK_LDFLAGS) -o pdsyevx
	$(CXX) $(CFLAGS) linalg.o basis.o build_tree.o comm.o umv.o \
	  dsyev.o $(LAPACK_LDFLAGS) -o dsyev
	$(CXX) $(CFLAGS) linalg.o basis.o build_tree.o comm.o umv.o \
	  dsyevx.o $(LAPACK_LDFLAGS) -o dsyevx

clean:
	rm -rf *.a *.o lorasp eigen pdsyev pdsyevx dsyev dsyevx
