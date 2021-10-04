
cmake_minimum_required (VERSION 3.9)

project ("nbd" LANGUAGES CXX)

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(BLAS)
find_package(LAPACK)

add_library(nbd STATIC 
  nbd.h
  kernel.cpp kernel.h
  build_tree.cpp build_tree.h 
  lra.cpp lra.h
  h2mv.cpp h2mv.h
  bsp_umv.cpp bsp_umv.h
)
target_include_directories(nbd PRIVATE "./" )
target_link_libraries(nbd blas lapacke)

add_subdirectory(examples)

