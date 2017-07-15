cmake_minimum_required(VERSION 3.7)
project(ray_tracing)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
add_executable(ray_tracing ${SOURCE_FILES})