
add_executable(symmh2_example
symmh2_example.cpp
)
target_link_libraries(symmh2_example nbd)
target_include_directories(symmh2_example PRIVATE "../" "./")

add_executable(h2_example
h2_example.cpp
)
target_link_libraries(h2_example nbd)
target_include_directories(h2_example PRIVATE "../" "./")

add_executable(lra_example
lra_example.cpp
)
target_link_libraries(lra_example nbd)
target_include_directories(lra_example PRIVATE "../" "./")

add_executable(bsp_lu
bsp_lu.cpp
)
target_link_libraries(bsp_lu nbd)
target_include_directories(bsp_lu PRIVATE "../" "./")

