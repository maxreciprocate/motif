cmake_minimum_required(VERSION 3.17)
project(motif LANGUAGES CXX CUDA)

add_subdirectory(pybind11)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CMAKE_CXX_STANDARD 11)

find_package(CUDAToolkit)

# pybind11 sets -fvisibility=hidden in INTERFACE_COMPILE_OPTIONS on it's module target
get_target_property(modifacecopts module INTERFACE_COMPILE_OPTIONS)
list(REMOVE_ITEM modifacecopts "-fvisibility=hidden")
set_target_properties(module PROPERTIES INTERFACE_COMPILE_OPTIONS "${modifacecopts}")

add_library(motif MODULE
    main.cc
    jam/jam.h
    jam/jam.cu
    jam/motif.h
    jam/motif.cc
    jam/queue.h
)

set_target_properties(motif PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET "hidden"
    CXX_VISIBILITY_PRESET "hidden"
    INTERPROCEDURAL_OPTIMIZATION TRUE
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

target_link_libraries(motif PRIVATE pybind11::module CUDA::cudart)
