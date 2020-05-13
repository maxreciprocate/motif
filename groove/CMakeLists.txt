cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_COMPILER /usr/bin/gcc-10)
set(CMAKE_CXX_COMPILER /usr/bin/g++-10)
project(groove)
find_package(Threads REQUIRED)
# find_package(CUDA REQUIRED)

add_subdirectory(pybind11)
pybind11_add_module(groove main.cpp motif/motif.cpp motif/src/readers/file_readers.h
                    motif/src/readers/file_readers.cpp)
# pybind11_add_module(gpu_library src/jam/gpu_library.cpp  src/jam/gpu_library.cu)
# target_link_libraries(groove Threads::Threads stdc++fs)
