#include "topology.h"

#include <kagen.h>
#include <mpi.h>

#include <cassert>
#include <cmath>

#include "common.h"

AlltoallTopology create_identitiy_topology(const std::size_t num_entries,
                                           MPI_Comm comm) {
    const int size = get_comm_size(comm);
    const int rank = get_comm_rank(comm);
    AlltoallTopology identitiy(size);
    identitiy[rank] = num_entries;
    return identitiy;
}

AlltoallTopology create_complete_topology(const std::size_t num_entries,
                                          MPI_Comm comm) {
    const int size = get_comm_size(comm);
    AlltoallTopology complete(size);
    std::fill(complete.begin(), complete.end(), num_entries);
    return complete;
}

AlltoallTopology create_grid_adjacent_topology(const std::size_t num_entries,
                                               const bool diagonal_neighbors,
                                               MPI_Comm comm) {
    // Assumed grid structure:
    // 0 1 2
    // 3 4 5
    // 6 7 8
    // Communicate with adjacent cells only
    const int size = get_comm_size(comm);
    const int rank = get_comm_rank(comm);
    const int row_length = std::floor(std::sqrt(size));
    const int my_row = rank / row_length;
    const int my_col = rank % row_length;

    auto encode = [&](const int row, const int col) {
        const auto id = row * row_length + col;
        assert(id < size);
        return id;
    };

    AlltoallTopology topology(size);
    if (my_row > 0) {  // up
        topology[encode(my_row - 1, my_col)] = num_entries;
    }
    if (my_col > 0) {  // left
        topology[encode(my_row, my_col - 1)] = num_entries;
    }
    if (my_col + 1 < row_length && rank + 1 < size) {  // right
        topology[encode(my_row, my_col + 1)] = num_entries;
    }
    if (rank + row_length < size) {  // down
        topology[encode(my_row + 1, my_col)] = num_entries;
    }
    if (diagonal_neighbors) {
        if (my_row > 0 && my_col > 0) {  // top left
            topology[encode(my_row - 1, my_col - 1)] = num_entries;
        }
        if (my_row > 0 && my_col + 1 < row_length) {  // top right
            topology[encode(my_row - 1, my_col + 1)] = num_entries;
        }
        if (my_col > 0 && rank + row_length - 1 < size) {  // bottom left
            topology[encode(my_row + 1, my_col - 1)] = num_entries;
        }
        if (my_col + 1 < row_length &&
            rank + row_length + 1 < size) {  // bottom right
            topology[encode(my_row + 1, my_col + 1)] = num_entries;
        }
    }

    return topology;
}

namespace {
using namespace kagen;
KaGenResult generate(const KaGenSettings &settings, MPI_Comm comm) {
    KaGen kagen(comm);

    switch (settings.generator) {
        case Generator::RGG2D:
            return kagen.GenerateRGG2D_NM(settings.n, settings.m);

        case Generator::RGG3D:
            return kagen.GenerateRGG3D_NM(settings.n, settings.m);

        case Generator::RHG:
            return kagen.GenerateRHG_NM(settings.gamma, settings.n, settings.m);

        case Generator::RMAT:
            return kagen.GenerateRMAT(settings.n, settings.m, settings.a,
                                      settings.b, settings.c);
    }

    __builtin_unreachable();
}
}  // namespace

AlltoallTopology create_graph_topology(const KaGenSettings &generator_settings,
                                       const CommunicationMode mode,
                                       const std::size_t scale, MPI_Comm comm) {
    const auto rank = get_comm_rank(comm);
    const auto size = get_comm_size(comm);

    const auto graph = generate(generator_settings, comm);
    const auto vtxdist =
        kagen::BuildVertexDistribution<int>(graph, MPI_INT, MPI_COMM_WORLD);

    auto find_owner = [&](const int node) {
        auto it = std::upper_bound(vtxdist.begin() + 1, vtxdist.end(), node);
        return std::distance(vtxdist.begin(), it) - 1;
    };

    AlltoallTopology topology(size);

    std::vector<bool> covered(size);
    std::vector<int> covered_ids;

    int last_u = 0;
    for (const auto &[u, v] : graph.edges) {
        if (v < graph.vertex_range.first || v >= graph.vertex_range.second) {
            const int pe = find_owner(v);
            if (mode == CommunicationMode::EDGE_CUT) {
                topology[pe] += scale;
            } else if (mode == CommunicationMode::COMMUNICATION_VOLUME &&
                       !covered[pe]) {
                topology[pe] += scale;
                covered[pe] = true;
                covered_ids.push_back(pe);
            }
        }
        if (mode == CommunicationMode::COMMUNICATION_VOLUME && u != last_u) {
            for (const int id : covered_ids) {
                covered[id] = false;
            }
            covered_ids.clear();
        }
        last_u = u;
    }

    return topology;
}
