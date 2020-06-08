cmake_minimum_required(VERSION 3.13)

project(groove)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_COMPILER /usr/bin/gcc-9)
set(CMAKE_CXX_COMPILER /usr/bin/g++-9)

find_package(Threads REQUIRED)

add_executable(groove main.cpp motif.cpp src/readers/file_readers.cpp src/readers/file_readers.cpp src/queue/ConcurrentQueue.h)

target_link_libraries(groove Threads::Threads)
