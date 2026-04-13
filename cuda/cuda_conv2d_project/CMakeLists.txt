cmake_minimum_required(VERSION 3.18)
project(cuda_conv2d LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(BUILD_TESTS "Build tests" ON)

add_library(conv2d_cuda STATIC
    src/conv2d.cu
)

target_include_directories(conv2d_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_options(conv2d_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

add_executable(conv2d_demo src/main.cpp)
target_link_libraries(conv2d_demo PRIVATE conv2d_cuda)
set_target_properties(conv2d_demo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(BUILD_TESTS)
    enable_testing()
    add_executable(test_conv2d tests/test_conv2d.cpp)
    target_link_libraries(test_conv2d PRIVATE conv2d_cuda)
    set_target_properties(test_conv2d PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME test_conv2d COMMAND test_conv2d)
endif()
