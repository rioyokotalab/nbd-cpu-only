
cmake_minimum_required (VERSION 3.9)

project ("nbd" LANGUAGES CXX)

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_library(nbd STATIC 
  nbd.h
  kernel.cpp kernel.h
  build_tree.cpp build_tree.h 
  aca.cpp aca.h
  min_blas.cpp min_blas.h)
target_include_directories(nbd PRIVATE "./" )

add_executable(main
  main.cpp
)
target_link_libraries(main nbd)