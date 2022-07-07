
cmake_minimum_required (VERSION 3.13)

set(CMAKE_C_COMPILER "icc")
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "-O3 -m64 -Wall -Wextra")

set(CMAKE_CXX_COMPILER "icpc")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-O3 -m64 -Wall -Wextra")

set(MKL_ARCH intel64)
set(MKL_LINK dynamic)
set(MKL_INTERFACE_FULL intel_ilp64)
set(MKL_THREADING intel_thread)
set(MKL_MPI openmpi)

project ("shimo" LANGUAGES C CXX)

find_package(MKL REQUIRED)
find_package(MPI REQUIRED)

add_library(nbd STATIC 
  kernel.c kernel.h
  linalg.c linalg.h
  build_tree.c build_tree.h
  umv.cxx umv.h
  solver.cxx solver.h
  dist.cxx dist.h
)
target_include_directories(nbd PRIVATE "./")
target_compile_options(nbd PRIVATE "-DUSE_MKL")
target_link_libraries(nbd MKL::MKL MPI::MPI_C MPI::MPI_CXX)

add_executable(lorasp lorasp.cxx)
target_include_directories(lorasp PRIVATE "./")
target_compile_options(lorasp PRIVATE)
target_link_libraries(lorasp nbd)

