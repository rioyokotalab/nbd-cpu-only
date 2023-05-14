include make.inc

%.o : %.cxx
	$(CXX) $(CXXFLAGS) $< -c -o $@

all: lorasp eigen dsyev dsyevx pdsyev pdsyevx test_mpi

lorasp: lorasp.o linalg.o basis.o build_tree.o comm.o umv.o
	$(CXX) $(CXXFLAGS) linalg.o basis.o build_tree.o comm.o umv.o \
	  lorasp.o $(LAPACK_SEQ_LDFLAGS) -o lorasp

eigen: eigen.o linalg.o basis.o build_tree.o comm.o umv.o
	$(CXX) $(CXXFLAGS) linalg.o basis.o build_tree.o comm.o umv.o \
	  eigen.o $(LAPACK_SEQ_LDFLAGS) -o eigen

dsyev: dsyev.o linalg.o basis.o build_tree.o comm.o umv.o
	$(CXX) $(CXXFLAGS) linalg.o basis.o build_tree.o comm.o umv.o \
	  dsyev.o $(LAPACK_PAR_LDFLAGS) -o dsyev

dsyevx: dsyevx.o linalg.o basis.o build_tree.o comm.o umv.o
	$(CXX) $(CXXFLAGS) linalg.o basis.o build_tree.o comm.o umv.o \
	  dsyevx.o $(LAPACK_PAR_LDFLAGS) -o dsyevx

pdsyev: pdsyev.o
	$(CXX) $(CXXFLAGS) pdsyev.o $(SCALAPACK_LDFLAGS) -o pdsyev

pdsyevx: pdsyevx.o
	$(CXX) $(CXXFLAGS) pdsyevx.o $(SCALAPACK_LDFLAGS) -o pdsyevx

test_mpi: test_mpi.o
	$(CXX) $(CXXFLAGS) test_mpi.o -o test_mpi

clean:
	rm -rf *.a *.o lorasp eigen pdsyev pdsyevx dsyev dsyevx test_mpi
