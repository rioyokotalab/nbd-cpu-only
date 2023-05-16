CXX                 = mpiFCCpx -std=c++14
CXXFLAGS            = -Nclang -Kfast -Kopenmp -Wall -Wextra -I. -I"${FJSVXTCLANGA}/include"
LAPACK_SEQ_LDFLAGS  = -SSL2
LAPACK_PAR_LDFLAGS  = -SSL2BLAMP
SCALAPACK_LDFLAGS   = -SCALAPACK -SSL2BLAMP
