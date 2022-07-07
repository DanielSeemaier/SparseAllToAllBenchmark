#pragma once

#include <mpi.h>

#include <cstdlib>
#include <vector>

using AlltoallTopology = std::vector<std::size_t>;

AlltoallTopology create_identitiy_topology(const std::size_t num_entries,
                                           MPI_Comm comm);

AlltoallTopology create_complete_topology(const std::size_t num_entries,
                                          MPI_Comm comm);

AlltoallTopology create_grid_adjacent_topology(const std::size_t num_entries,
                                               const bool diagonal_neighbors,
                                               MPI_Comm comm);

enum class Generator {
    RGG2D,
    RGG3D,
    RHG,
    RMAT,
};

enum class CommunicationMode {
    EDGE_CUT,
    COMMUNICATION_VOLUME,
};

struct KaGenSettings {
    Generator generator;
    long n, m;
    double gamma;
    double a, b, c;
};

AlltoallTopology create_graph_topology(const KaGenSettings &generator_settings,
                                       const CommunicationMode mode,
                                       const std::size_t scale, MPI_Comm comm);
