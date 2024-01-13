cmake_minimum_required(VERSION 3.25)
project(K_SELECT)

set(CMAKE_CXX_COMPILER mpic++)
message("${CMAKE_CXX_COMPILER}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++20 -fopenmp -O3 -ftree-vectorize")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(MPI) # make it REQUIRED, if you want
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

find_package(CURL)

file(GLOB SOURCES "src/*.cpp")

add_executable(output main.cpp ${SOURCES})
target_include_directories(output PUBLIC inc ${CURL_INCLUDE_DIRS})
target_link_libraries(output ${MPI_CXX_LIBRARIES} ${CURL_LIBRARIES})