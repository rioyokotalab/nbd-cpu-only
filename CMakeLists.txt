﻿
cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-O3 -m64 -Wall -Wextra")

project ("shimo" LANGUAGES CXX)

set(MKL_ARCH intel64)
set(MKL_LINK dynamic)
set(MKL_INTERFACE_FULL intel_lp64)
set(MKL_THREADING sequential)
set(MKL_MPI intelmpi)
find_package(MKL REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PkgConfig REQUIRED)

add_executable(lorasp lorasp.cxx linalg.cxx basis.cxx comm.cxx build_tree.cxx umv.cxx)
target_include_directories(lorasp PRIVATE ./)
target_link_libraries(lorasp MPI::MPI_CXX OpenMP::OpenMP_CXX MKL::MKL)

