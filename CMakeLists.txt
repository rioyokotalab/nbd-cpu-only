
cmake_minimum_required (VERSION 3.18)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "-O3 -m64 -Wall -Wextra")

project ("shimo" LANGUAGES C)

if(DEFINED ENV{MKLROOT})
set(MKL_ARCH intel64)
set(MKL_LINK dynamic)
set(MKL_INTERFACE_FULL intel_lp64)
set(MKL_THREADING gnu_thread)
set(MKL_MPI intelmpi)
find_package(MKL)
elseif(DEFINED ENV{OPENBLAS_DIR})
set(BLA_VENDOR OpenBLAS)
find_package(BLAS)
find_package(LAPACK)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(lorasp lorasp.c kernel.c linalg.c basis.c build_tree.c umv.c profile.c)
target_include_directories(lorasp PRIVATE "./")
target_link_libraries(lorasp MPI::MPI_C OpenMP::OpenMP_C)

if(MKL_FOUND)
target_compile_options(lorasp PRIVATE -D_PROF -D_MKL)
target_link_libraries(lorasp MKL::MKL)
elseif(BLAS_FOUND)
target_include_directories(lorasp PRIVATE "$ENV{OPENBLAS_DIR}/include")
target_compile_options(lorasp PRIVATE -D_PROF)
target_link_libraries(lorasp m dl BLAS::BLAS LAPACK::LAPACK)
endif()
