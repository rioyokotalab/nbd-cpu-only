
cmake_minimum_required (VERSION 3.13)

project ("shimo" LANGUAGES CXX)

set(MKL_ARCH intel64)
set(MKL_LINK dynamic)
set(MKL_INTERFACE_FULL intel_lp64)
set(MKL_THREADING sequential)
set(MKL_MPI intelmpi)

find_package(MKL REQUIRED)
message(STATUS "${MKL_IMPORTED_TARGETS}")
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_library(nbd STATIC 
  build_tree.cxx build_tree.hxx
  kernel.cxx kernel.hxx
  linalg.cxx linalg.hxx
  basis.cxx basis.hxx
  umv.cxx umv.hxx
  solver.cxx solver.hxx
  dist.cxx dist.hxx
)
target_include_directories(nbd PRIVATE "./")
target_compile_options(nbd PRIVATE "-DUSE_MKL=true")
target_link_libraries(nbd
  MKL::MKL
  OpenMP::OpenMP_CXX
  MPI::MPI_CXX
)

add_executable(lorasp
  lorasp.cxx
)
target_include_directories(lorasp PRIVATE "./")
target_compile_options(lorasp PRIVATE)
target_link_libraries(lorasp nbd
  MKL::MKL
  OpenMP::OpenMP_CXX
  MPI::MPI_CXX
)

