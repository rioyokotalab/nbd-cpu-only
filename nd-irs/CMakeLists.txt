
cmake_minimum_required (VERSION 3.9)

project ("nd-irs" LANGUAGES CXX)

add_library(nd-irs STATIC
  block_matrix.cpp block_matrix.h
  solve.cpp solve.h
)

target_include_directories(nd-irs PRIVATE "./" )
target_link_libraries(nd-irs nbd)
