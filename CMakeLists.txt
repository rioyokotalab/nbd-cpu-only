
cmake_minimum_required (VERSION 3.13)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "-O3 -Wall -Wextra")

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")

set(MKL_ARCH intel64)
set(MKL_LINK dynamic)
set(MKL_INTERFACE_FULL intel_lp64)
set(MKL_THREADING sequential)
set(MKL_MPI intelmpi)

project ("shimo" LANGUAGES C CXX)

find_package(MKL REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_library(nbd STATIC 
  kernel.c kernel.h
  linalg.c linalg.h
  build_tree.cxx build_tree.hxx
  basis.cxx basis.hxx
  umv.cxx umv.hxx
  solver.cxx solver.hxx
  dist.cxx dist.hxx
)
target_include_directories(nbd PRIVATE "./")
target_compile_options(nbd PRIVATE "-DUSE_MKL")
target_link_libraries(nbd MKL::MKL OpenMP::OpenMP_C OpenMP::OpenMP_CXX MPI::MPI_C MPI::MPI_CXX)

add_executable(lorasp lorasp.cxx)
target_include_directories(lorasp PRIVATE "./")
target_compile_options(lorasp PRIVATE)
target_link_libraries(lorasp nbd)

