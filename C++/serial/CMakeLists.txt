cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-O4")

add_library(neural_network neural_network.cpp)

add_executable(executable main.cpp)
target_link_libraries(executable neural_network)
