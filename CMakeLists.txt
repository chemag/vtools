cmake_minimum_required(VERSION 3.10)

project(vtools)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BUILD_PYBINDINGS ON CACHE BOOL "Enable building of pybindings" FORCE)

# Add liblcvm subdirectory
add_subdirectory(lib/liblcvm)
