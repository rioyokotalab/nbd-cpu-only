CXX                 = mpiicpc -std=c++14
CXXFLAGS            = -DUSE_MKL -axCORE-AVX512 -O3 -m64 -Wall -Wextra -I. -I"${MKLROOT}/include" -qopenmp
LAPACK_SEQ_LDFLAGS  = -qmkl=sequential
LAPACK_PAR_LDFLAGS  = -qmkl=parallel
SCALAPACK_LDFLAGS   = -qmkl=cluster
