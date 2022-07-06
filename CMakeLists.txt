project(SparseA2ABench)
cmake_minimum_required(VERSION 3.16)

add_executable(SparseA2ABenchmark main.cpp)
target_compile_features(SparseA2ABenchmark PRIVATE cxx_std_17)

find_package(MPI REQUIRED)
target_link_libraries(SparseA2ABenchmark PUBLIC MPI::MPI_CXX)

add_subdirectory(external/KaGen)
target_link_libraries(SparseA2ABenchmark PUBLIC KaGen::KaGen)
