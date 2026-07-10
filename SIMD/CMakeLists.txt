add_executable(MATMULv1 MATMULv1.cpp)
target_link_libraries(MATMULv1 PRIVATE matmul_utils)

add_executable(MATMULv2 MATMULv2.cpp)
target_link_libraries(MATMULv2 PRIVATE matmul_utils)

add_executable(AVX_exe AVX.cpp)
target_link_libraries(AVX_exe PRIVATE matmul_utils)

