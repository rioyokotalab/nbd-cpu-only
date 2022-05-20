
CXX = FCC -std=c++11 -O3 -I. -Kfast,openmp -SSL2
MPICXX  = mpiFCC -std=c++11 -O3 -I. -Kfast,openmp -SSL2
LDFLAGS	= -lm

all:
	make lorasp

linalg: linalg.cxx linalg.hxx
	$(CXX) -c linalg.cxx

kernel: kernel.cxx kernel.hxx
	$(CXX) -c kernel.cxx

build_tree: build_tree.cxx build_tree.hxx
	$(CXX) -c build_tree.cxx

basis: basis.cxx basis.hxx
	$(CXX) -c basis.cxx

umv: umv.cxx umv.hxx
	$(CXX) -c umv.cxx

solver: solver.cxx solver.hxx
	$(CXX) -c solver.cxx

dist: dist.cxx dist.hxx
	$(MPICXX) -c dist.cxx

lib: linalg kernel build_tree basis umv solver dist
	ar rcs libnbd.a linalg.o kernel.o build_tree.o basis.o umv.o solver.o dist.o

lorasp: lorasp.cxx lib
	$(CXX) -c lorasp.cxx
	$(MPICXX) -o lorasp lorasp.o -L. -lnbd $(LDFLAGS)

clean:
	rm -f *.o *.a a.out lorasp
