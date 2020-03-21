cmake_minimum_required(VERSION 3.13)

project(.)

set(CMAKE_CXX_STANDARD 17)

find_package(Threads REQUIRED)

add_executable(main main.cpp motif.cpp file_readers.cpp file_readers.h BitStream.cpp)

target_link_libraries(main Threads::Threads)