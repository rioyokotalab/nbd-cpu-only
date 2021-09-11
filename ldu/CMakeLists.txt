
cmake_minimum_required (VERSION 3.9)

project ("ldu" LANGUAGES CXX)

add_library(ldu STATIC
  block_matrix.cpp block_matrix.h
)

target_include_directories(ldu PRIVATE "./" )
target_link_libraries(ldu nbd)
