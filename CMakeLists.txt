
cmake_minimum_required (VERSION 3.9)

project ("shimo" LANGUAGES C CXX)

find_package(MPI REQUIRED)

message(STATUS ${MPI_INCLUDE_PATH})
message(STATUS ${MPI_C_LIBRARIES})
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_library(nbd STATIC 
  minblas.c minblas.h
  build_tree.cxx build_tree.hxx
  kernel.cxx kernel.hxx
  linalg.cxx linalg.hxx
  basis.cxx basis.hxx
  umv.cxx umv.hxx
  h2mv.cxx h2mv.hxx
  solver.cxx solver.hxx
  dist.cxx dist.hxx
)
target_include_directories(nbd PRIVATE "./" )
target_compile_options(nbd PRIVATE ${MPI_C_COMPILE_FLAGS})
target_link_libraries(nbd ${MPI_C_LIBRARIES} ${MPI_C_LINK_FLAGS})

add_executable(h2_example
  h2_example.cxx
)
target_compile_options(h2_example PRIVATE ${MPI_C_COMPILE_FLAGS})
target_link_libraries(h2_example nbd ${MPI_C_LIBRARIES} ${MPI_C_LINK_FLAGS})
target_include_directories(h2_example PRIVATE "./" )

add_executable(lorasp
  lorasp.cxx
)
target_compile_options(lorasp PRIVATE ${MPI_C_COMPILE_FLAGS})
target_link_libraries(lorasp nbd ${MPI_C_LIBRARIES} ${MPI_C_LINK_FLAGS})
target_include_directories(lorasp PRIVATE "./" )

add_executable(lra_example
  lra_example.cxx
)
target_link_libraries(lra_example nbd)
target_include_directories(lra_example PRIVATE "./" )
