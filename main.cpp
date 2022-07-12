#include <math.h>
#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <ratio>
#include <unordered_map>
#include <vector>

#include "alltoall.h"
#include "common.h"
#include "topology.h"

constexpr int NUM_REPETITIONS = 10;
constexpr int NUM_WARMUPS = 1;

constexpr bool csv = true;

template <typename Data>
auto get_competitors() {
    return std::vector<std::pair<
        std::string,
        std::function<AlltoallResult<Data>(
            const std::vector<std::vector<Data>> &, MPI_Datatype, MPI_Comm)>>>{
        std::make_pair("alltoallv", mpi_alltoallv<Data>),
        // std::make_pair("alltoall", mpi_alltoall<Data>),
        std::make_pair("complete_isend_recv",
                       complete_send_recv_alltoall<Data>),
        std::make_pair("sparse_isend_recv", sparse_send_recv_alltoall<Data>),
        std::make_pair("grid_2d", grid_alltoall<Data>),
    };
}

template <typename Data>
void generate_data(Data *dst, const std::size_t n) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<Data> dist(
        std::numeric_limits<Data>::lowest(), std::numeric_limits<Data>::max());
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] = dist(mt);
    }
}

std::size_t encode(const int from, const int to, const int size) {
    return static_cast<std::size_t>(from) * static_cast<std::size_t>(size) +
           static_cast<std::size_t>(to);
}

template <typename Duration>
double compute_mbs(const std::size_t nbytes, const Duration time) {
    const auto period = 1.0 * Duration::period::num / Duration::period::den;
    /*
    std::cout
        << std::endl
        << nbytes << " Bytes in "
        << std::chrono::duration_cast<std::chrono::milliseconds>(time).count()
        << " ms == " << (1.0 * nbytes / 1'000'000) / (time.count() * period)
        << std::endl;
        */
    return (1.0 * nbytes / 1'000'000) / (time.count() * period);
}

template <typename Data>
void run_benchmark(const std::string topology_name, AlltoallTopology topology,
                   const int scale, MPI_Datatype data_type, MPI_Comm comm) {
    using namespace std::chrono_literals;

    const int size = get_comm_size(comm);
    const int rank = get_comm_rank(comm);

    // Generate data
    std::vector<std::vector<Data>> data(size);
    for (int pe = 0; pe < size; ++pe) {
        const std::size_t n = topology[pe];
        data[pe].resize(n);
        generate_data(data[pe].data(), n);
    }

    // Message sizes
    const auto data_size = sizeof(Data);

    const long local_size =
        std::accumulate(topology.begin(), topology.end(), 0);

    long global_min_size, global_max_size, global_total_size;
    MPI_Reduce(&local_size, &global_min_size, 1, MPI_LONG, MPI_MIN, 0, comm);
    MPI_Reduce(&local_size, &global_max_size, 1, MPI_LONG, MPI_MAX, 0, comm);
    MPI_Reduce(&local_size, &global_total_size, 1, MPI_LONG, MPI_SUM, 0, comm);
    const long global_avg_size = global_total_size / size;
    const long total_nbytes = global_total_size * data_size;

    if (!csv && rank == 0) {
        std::cout << "Running topology " << topology_name << " with "
                  << total_nbytes / 1'000'000 << " MB in total" << std::endl;
        std::cout << std::endl;
    }

    // Warmup
    for (int round = 0; round < NUM_WARMUPS; ++round) {
        for (const auto &[name, competitor] : get_competitors<Data>()) {
            competitor(data, data_type, comm);
        }
    }

    // Measurement
    std::vector<std::vector<Data>> expected_result;

    for (const auto &[competitor_name, competitor] : get_competitors<Data>()) {
        if (!csv && rank == 0) {
            std::cout << std::setw(30) << std::left << competitor_name;
        }

        auto total_time = Resolution(0s);
        for (int round = 0; round < NUM_REPETITIONS; ++round) {
            // Run algorithm
            const auto [ans, time] = competitor(data, data_type, comm);
            if (!ans.empty()) {
                if (expected_result.empty()) {
                    expected_result = std::move(ans);
                } else if (expected_result != ans) {
                    std::cout << "bad result " << expected_result.size() << " "
                              << ans.size() << std::endl;
                    std::exit(1);
                }
            }

            if (rank == 0) {
                if (csv) {
                    std::cout
                        << size << "," << topology_name << "," << data_size
                        << "," << scale << "," << global_total_size << ","
                        << global_max_size << "," << global_avg_size << ","
                        << global_min_size << "," << competitor_name << ","
                        << round << "," << time.count() << ","
                        << compute_mbs(total_nbytes, time) << std::endl;
                } else {
                    std::cout << std::setw(7);
                    if (ans.empty()) {
                        std::cout << "NA" << std::flush;
                    } else {
                        std::cout << std::chrono::duration_cast<
                                         std::chrono::milliseconds>(time)
                                         .count()
                                  << std::flush;
                    }
                }

                total_time += time;
            }
        }

        if (rank == 0) {
            const double mbs =
                compute_mbs(total_nbytes * data_size, total_time);
            if (!csv) {
                std::cout << "== " << std::fixed << std::setprecision(0)
                          << std::setw(5) << mbs << " MB/s" << std::endl;
            }
        }
    }

    if (!csv && rank == 0) {
        std::cout << std::endl;
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    if (argc < 3) {
        std::cout << "./" << argv[0] << " <N> <M>" << std::endl;
        std::exit(1);
    }

    const int N = std::atoi(argv[1]);
    const int M = std::atoi(argv[2]);

    const int size = get_comm_size(MPI_COMM_WORLD);
    const int rank = get_comm_rank(MPI_COMM_WORLD);

    if (rank == 0) {
        if (csv) {
            std::cout << "MPI,Topology,ElementSize,Scale,TotalSize,MaxSize,"
                         "AvgSize,MinSize,"
                         "Algorithm,Run,Time,MBs"
                      << std::endl;
        } else {
            std::cout << "MPI_SIZE=" << size << std::endl;
            std::cout << std::endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) { std::cout << "Creating topologies ..." << std::endl; }
    const auto id_topology = create_identitiy_topology(1, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) { std::cout << "Created ID topology" << std::endl; }
    const auto grid_4_topology =
        create_grid_adjacent_topology(1, false, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) { std::cout << "Created 4-Cells topology" << std::endl; }
    const auto grid_8_topology =
        create_grid_adjacent_topology(1, true, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) { std::cout << "Created 8-Cells topology" << std::endl; }
    const auto rgg2d_topology = create_graph_topology(
        {.generator = Generator::RGG2D, .n = 1 << N, .m = 1 << M},
        CommunicationMode::EDGE_CUT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) { std::cout << "Created RGG2D topology" << std::endl; }
    const auto rgg3d_topology = create_graph_topology(
        {.generator = Generator::RGG3D, .n = 1 << N, .m = 1 << M},
        CommunicationMode::EDGE_CUT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) { std::cout << "Created RGG3D topology" << std::endl; }
    const auto rhg_topology = create_graph_topology(
        {.generator = Generator::RHG, .n = 1 << N, .m = 1 << M, .gamma = 3.0},
        CommunicationMode::EDGE_CUT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) { std::cout << "Created RHG topology" << std::endl; }
    /*
    const auto rmat_topology =
        create_graph_topology({.generator = Generator::RMAT,
                               .n = 1 << N,
                               .m = 1 << M,
                               .a = 0.1,
                               .b = 0.2,
                               .c = 0.3},
                              CommunicationMode::EDGE_CUT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) { std::cout << "Created RMAT topology" << std::endl; }
    */

    for (std::size_t scale : {1, 5, 10, 15, 20}) {
        const auto message_size = 1 << scale;

        run_benchmark<int>("identity",
                           scale_topology(id_topology, message_size), scale,
                           MPI_INT, MPI_COMM_WORLD);

        run_benchmark<int>("adjacent_cells_4",
                           scale_topology(grid_4_topology, message_size), scale,
                           MPI_INT, MPI_COMM_WORLD);

        run_benchmark<int>("adjacent_cells_8",
                           scale_topology(grid_8_topology, message_size), scale,
                           MPI_INT, MPI_COMM_WORLD);

        run_benchmark<int>("rgg2d", scale_topology(rgg2d_topology, scale),
                           scale, MPI_INT, MPI_COMM_WORLD);

        run_benchmark<int>("rgg3d", scale_topology(rgg3d_topology, scale),
                           scale, MPI_INT, MPI_COMM_WORLD);

        run_benchmark<int>("rhg", scale_topology(rhg_topology, scale), scale,
                           MPI_INT, MPI_COMM_WORLD);

	/*
        run_benchmark<int>("rmat", scale_topology(rmat_topology, scale), scale,
                           MPI_INT, MPI_COMM_WORLD);
			   */
    }

    MPI_Finalize();

    return 0;
}
