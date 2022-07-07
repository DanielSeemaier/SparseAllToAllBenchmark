#pragma once

#include <mpi.h>

#include <chrono>

using Resolution = std::chrono::nanoseconds;

inline auto now() { return std::chrono::steady_clock::now(); }

inline int get_comm_size(MPI_Comm comm) {
    int size;
    MPI_Comm_size(comm, &size);
    return size;
}

inline int get_comm_rank(MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    return rank;
}
