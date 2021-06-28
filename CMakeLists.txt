
cmake_minimum_required (VERSION 3.9)

project ("nbd" LANGUAGES CXX)

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(MSVC)
  include_directories($ENV{MKL_PATH}/include)
  link_directories($ENV{MKL_PATH}/lib/intel64)
elseif(LINUX)
  include_directories($ENV{MKLROOT}/include)
  link_directories($ENV{MKLROOT}/lib/intel64)
endif()

add_library(nbd STATIC 
  nbd.h
  kernel.cpp kernel.h
  build_tree.cpp build_tree.h 
  aca.cpp aca.h
  h2mv.cpp h2mv.h
  test_util.cpp test_util.h
)
target_include_directories(nbd PRIVATE "./" )
target_link_libraries(nbd mkl_intel_lp64 mkl_sequential mkl_core)

add_executable(symmh2_example
  symmh2_example.cpp
)
target_link_libraries(symmh2_example nbd)
target_include_directories(symmh2_example PRIVATE "./" )

add_executable(h2_example
  h2_example.cpp
)
target_link_libraries(h2_example nbd)
target_include_directories(h2_example PRIVATE "./" )

add_executable(lra_example
  lra_example.cpp
)
target_link_libraries(lra_example nbd)
target_include_directories(lra_example PRIVATE "./" )
