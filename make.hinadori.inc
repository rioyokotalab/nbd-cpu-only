CXX                 = mpicxx -std=c++14
CXXFLAGS            = -DUSE_MKL -O3 -m64 -Wall -Wextra -I. -I"${MKLROOT}/include" -fopenmp
LAPACK_SEQ_LDFLAGS  = -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
LAPACK_PAR_LDFLAGS  = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed \
                      -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
SCALAPACK_LDFLAGS   = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_scalapack_lp64 \
                      -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 \
                      -liomp5 -lpthread -lm -ldl
