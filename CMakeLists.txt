cmake_minimum_required(VERSION 3.15)
project(MatrixProject CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mfma -g -O3")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Subdirectories
add_subdirectory(Utilities)
add_subdirectory(MatrixLib)
add_subdirectory(SIMD)





