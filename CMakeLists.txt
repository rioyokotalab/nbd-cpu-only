
cmake_minimum_required (VERSION 3.9)

project ("shimo" LANGUAGES CXX)

set(BLA_VENDOR Intel10_64lp_seq)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(MPI REQUIRED)

add_library(nbd STATIC 
  build_tree.cxx build_tree.hxx
  kernel.cxx kernel.hxx
  linalg.cxx linalg.hxx
  basis.cxx basis.hxx
  umv.cxx umv.hxx
  solver.cxx solver.hxx
  dist.cxx dist.hxx
)
target_include_directories(nbd PRIVATE "./" ${MPI_INCLUDE_PATH})
target_compile_options(nbd PRIVATE "-DUSE_MKL=true" ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(nbd
  ${BLAS_LIBRARIES} ${BLAS_LINKER_FLAGS}
  ${LAPACK_LIBRARIES} ${LAPACK_LINKER_FLAGS}
  ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS}
)

add_executable(lorasp
  lorasp.cxx
)
target_include_directories(lorasp PRIVATE "./")
target_compile_options(lorasp PRIVATE ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(lorasp nbd 
  ${BLAS_LIBRARIES} ${BLAS_LINKER_FLAGS}
  ${LAPACK_LIBRARIES} ${LAPACK_LINKER_FLAGS}
  ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS}
)

