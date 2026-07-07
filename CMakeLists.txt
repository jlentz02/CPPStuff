cmake_minimum_required(VERSION 3.15)
project(MatrixProject CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mfma -g")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include the SIMD directory so CMake processes its targets
add_subdirectory(SIMD)

# If MatrixLib needs to build its own executable:
add_executable(matrix MatrixLib/matrix.cpp)
