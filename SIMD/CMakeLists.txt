# 1. Define the shared utility library
add_library(matmul_utils matmul_utils.cpp)

# 2. Expose this directory's path so other folders can find the headers
target_include_directories(matmul_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# 3. Define your separate executables
add_executable(MATMULv1 MATMULv1.cpp)
target_link_libraries(MATMULv1 PRIVATE matmul_utils)

add_executable(MATMULv2 MATMULv2.cpp)
target_link_libraries(MATMULv2 PRIVATE matmul_utils)

add_executable(AVX_exe AVX.cpp)
target_link_libraries(AVX_exe PRIVATE matmul_utils)