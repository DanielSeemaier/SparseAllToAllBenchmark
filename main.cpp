#include <math.h>
#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
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
        std::make_pair("alltoall", mpi_alltoall<Data>),
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
                   MPI_Datatype data_type, MPI_Comm comm) {
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
                    std::cout << size << "," << topology_name << ","
                              << data_size << "," << global_total_size << ","
                              << global_max_size << "," << global_avg_size
                              << "," << global_min_size << ","
                              << competitor_name << "," << round << ","
                              << time.count() << ","
                              << compute_mbs(total_nbytes, time) << std::endl;
                } else {
                    std::cout << std::setw(7);
                    if (ans.empty()) {
                        std::cout << "NA" << std::flush;
                    } else {
                        std::cout << time.count() << std::flush;
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

    const int size = get_comm_size(MPI_COMM_WORLD);
    const int rank = get_comm_rank(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "MPI_SIZE=" << size << std::endl;
        std::cout << std::endl;
        if (csv) {
            std::cout
                << "MPI,Topology,ElementSize,TotalSize,MaxSize,AvgSize,MinSize,"
                   "Algorithm,Run,Time,MBs"
                << std::endl;
        }
    }

    for (std::size_t message_size :
         {1 << 0, 1 << 5, 1 << 10, 1 << 15, 1 << 20, 1 << 25}) {
        run_benchmark<int>(
            "identity", create_identitiy_topology(message_size, MPI_COMM_WORLD),
            MPI_INT, MPI_COMM_WORLD);

        /*
                run_benchmark<int>(
                    "complete", create_complete_topology(message_size,
           MPI_COMM_WORLD), MPI_INT, MPI_COMM_WORLD);
        */

        run_benchmark<int>(
            "adjacent_cells_4",
            create_grid_adjacent_topology(message_size, false, MPI_COMM_WORLD),
            MPI_INT, MPI_COMM_WORLD);

        run_benchmark<int>(
            "adjacent_cells_8",
            create_grid_adjacent_topology(message_size, true, MPI_COMM_WORLD),
            MPI_INT, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}
