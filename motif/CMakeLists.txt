cmake_minimum_required(VERSION 3.13)

project(.)

set(CMAKE_CXX_STANDARD 17)

find_package(Threads REQUIRED)

add_executable(main main.cpp motif.cpp src/readers/file_readers.cpp src/readers/file_readers.cpp src/queue/ConcurrentQueue.cpp)

target_link_libraries(main Threads::Threads)

#add_executable(queue src/queue/ConcurrentQueue.cpp src/readers/file_readers.h)