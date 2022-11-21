
cmake_minimum_required (VERSION 3.18)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "-O3 -m64 -Wall -Wextra")

project ("shimo" LANGUAGES C CUDA)

set(MKL_ARCH intel64)
set(MKL_LINK dynamic)
set(MKL_INTERFACE_FULL intel_lp64)
set(MKL_THREADING sequential)
set(MKL_MPI intelmpi)
find_package(MKL REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(lorasp lorasp.c kernel.c linalg.c batch.c basis.c build_tree.c umv.c profile.c)
target_compile_options(lorasp PRIVATE -D_PROF)
target_include_directories(lorasp PRIVATE ./)
target_link_libraries(lorasp MPI::MPI_C OpenMP::OpenMP_C MKL::MKL)

if(DEFINED ENV{CUDADIR})
  add_executable(gpu lorasp.c kernel.c linalg.c batch_gpu.cu basis.c build_tree.c umv.c profile.c)
  target_compile_options(gpu PRIVATE -D_PROF)
  target_include_directories(gpu PRIVATE ./)
  target_link_libraries(gpu MPI::MPI_C OpenMP::OpenMP_C MKL::MKL -L$ENV{CUDADIR}/lib64 -lcublas -lcudart -lcusolver)
endif()

