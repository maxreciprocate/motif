cmake_minimum_required(VERSION 3.10)
project(jam_lib LANGUAGES CXX CUDA)

add_subdirectory(pybind11)

set(CMAKE_C_COMPILER /usr/bin/gcc-9)
set(CMAKE_CXX_COMPILER /usr/bin/g++-9)

##pybind11 sets -fvisibility=hidden in INTERFACE_COMPILE_OPTIONS on it's module target
get_target_property(modifacecopts module INTERFACE_COMPILE_OPTIONS)
list(REMOVE_ITEM modifacecopts "-fvisibility=hidden")
set_target_properties(module PROPERTIES INTERFACE_COMPILE_OPTIONS "${modifacecopts}")

add_library(jam_lib MODULE
    jam_lib.cpp
    jam/jam_run.cc
    jam/jam.h
    jam/jam.cu
)

set_target_properties(jam_lib PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET "hidden"
    CXX_VISIBILITY_PRESET "hidden"
    INTERPROCEDURAL_OPTIMIZATION TRUE
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

target_link_libraries(jam_lib PRIVATE pybind11::module)
