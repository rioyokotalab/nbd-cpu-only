
cmake_minimum_required (VERSION 3.9)

project ("qsparse" LANGUAGES CXX)

add_library(qs STATIC 
  factorize.cpp factorize.h
  solve.cpp solve.h
  block_matrix.cpp block_matrix.h
)
target_include_directories(qs PRIVATE "./" )
target_link_libraries(qs nbd)
