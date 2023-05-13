include make.inc

all:
	$(CXX) $(CXXFLAGS) -c linalg.cxx -o linalg.o
	$(CXX) $(CXXFLAGS) -c basis.cxx -o basis.o
	$(CXX) $(CXXFLAGS) -c build_tree.cxx -o build_tree.o
	$(CXX) $(CXXFLAGS) -c comm.cxx -o comm.o
	$(CXX) $(CXXFLAGS) -c umv.cxx -o umv.o
	$(CXX) $(CXXFLAGS) -c lorasp.cxx -o lorasp.o
	$(CXX) $(CXXFLAGS) -c eigen.cxx -o eigen.o
	$(CXX) $(CXXFLAGS) -c pdsyev.cxx -o pdsyev.o
	$(CXX) $(CXXFLAGS) -c pdsyevx.cxx -o pdsyevx.o
	$(CXX) $(CXXFLAGS) -c dsyev.cxx -o dsyev.o
	$(CXX) $(CXXFLAGS) -c dsyevx.cxx -o dsyevx.o

	$(CXX) $(CXXFLAGS) linalg.o basis.o build_tree.o comm.o umv.o \
	  lorasp.o $(LAPACK_SEQ_LDFLAGS) -o lorasp
	$(CXX) $(CXXFLAGS) linalg.o basis.o build_tree.o comm.o umv.o \
	  eigen.o $(LAPACK_SEQ_LDFLAGS) -o eigen
	$(CXX) $(CXXFLAGS) pdsyev.o $(SCALAPACK_LDFLAGS) -o pdsyev
	$(CXX) $(CXXFLAGS) pdsyevx.o $(SCALAPACK_LDFLAGS) -o pdsyevx
	$(CXX) $(CXXFLAGS) linalg.o basis.o build_tree.o comm.o umv.o \
	  dsyev.o $(LAPACK_PAR_LDFLAGS) -o dsyev
	$(CXX) $(CXXFLAGS) linalg.o basis.o build_tree.o comm.o umv.o \
	  dsyevx.o $(LAPACK_PAR_LDFLAGS) -o dsyevx

clean:
	rm -rf *.a *.o lorasp eigen pdsyev pdsyevx dsyev dsyevx
