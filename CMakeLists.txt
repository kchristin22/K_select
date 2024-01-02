cmake_minimum_required(VERSION 3.25)
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
message("${CMAKE_CXX_COMPILER}")
project(K_SELECT)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++20 -fopenmp -O3 -ftree-vectorize")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(MPI) #make it REQUIRED, if you want
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

find_package(CURL)

# include_directories(lib/inc)

# add_library(library lib/src/mmio.cpp)

file(GLOB SOURCES "src/*.cpp")

add_executable(output main.cpp ${SOURCES})
target_include_directories(output PUBLIC inc ${CURL_INCLUDE_DIRS})
target_link_libraries(output ${MPI_C_LIBRARIES} ${CURL_LIBRARIES})
# target_link_libraries(output PRIVATE library)
