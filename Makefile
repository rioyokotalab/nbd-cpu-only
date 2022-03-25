
CC	= mpifcc -O3 -I.
CXX	= mpiFCC -std=c++11 -O3 -I.
MPICXX	= mpiFCC -O3 -I.
LC	= -lm

all:
	make lorasp h2 lra

minblas: minblas.c minblas.h
	$(CC) -c minblas.c

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

h2mv: h2mv.cxx h2mv.hxx
	$(CXX) -c h2mv.cxx

solver: solver.cxx solver.hxx
	$(CXX) -c solver.cxx

dist: dist.cxx dist.hxx
	$(MPICXX) -c dist.cxx

lib: minblas linalg kernel build_tree basis umv h2mv solver dist
	ar rcs libnbd.a minblas.o linalg.o kernel.o build_tree.o basis.o umv.o h2mv.o solver.o dist.o

lorasp: lorasp.cxx lib
	$(CXX) -c lorasp.cxx
	$(MPICXX) -o lorasp lorasp.o -L. -lnbd

h2: h2_example.cxx lib
	$(CXX) -c h2_example.cxx
	$(MPICXX) -o h2_example h2_example.o -L. -lnbd

lra: lra_example.cxx lib
	$(CXX) -c lra_example.cxx
	$(CXX) -o lra_example lra_example.o -L. -lnbd

clean:
	rm -f *.o *.a a.out lorasp h2_example lra_example
