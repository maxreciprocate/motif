cmake_minimum_required(VERSION 3.17)
project(jam CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

find_package(Threads)
find_package(CUDAToolkit)

add_executable(jam main.cc jam.cu jam.h)
target_link_libraries(jam PRIVATE CUDA::cudart)
