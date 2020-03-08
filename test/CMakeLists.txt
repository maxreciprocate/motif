cmake_minimum_required(VERSION 3.13)

project(motiftest)

include_directories("${CMAKE_SOURCE_DIR}/")

file(GLOB sources "${CMAKE_SOURCE_DIR}/*.cc")
add_library(objs OBJECT ${sources})

file(GLOB tests "${PROJECT_SOURCE_DIR}/*.cc")
list(REMOVE_ITEM tests "${PROJECT_SOURCE_DIR}/main.cc")

foreach(file ${tests})
  set(name)
  message("another one .. ${file}")
  get_filename_component(name ${file} NAME_WE)
  add_executable("${name}_tests" $<TARGET_OBJECTS:objs> ${file} "${PROJECT_SOURCE_DIR}/main.cc")
  target_link_libraries("${name}_tests" gtest_main)
  add_test(NAME ${name} COMMAND "${name}_tests")
endforeach()
