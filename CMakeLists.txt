project(SparseA2ABench)
cmake_minimum_required(VERSION 3.16)

add_executable(SparseA2ABenchmark main.cpp)

find_package(MPI REQUIRED)
target_link_libraries(SparseA2ABenchmark PUBLIC MPI::MPI_CXX)
