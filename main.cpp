#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <ratio>
#include <unordered_map>
#include <vector>

constexpr int NUM_REPETITIONS = 5;
constexpr int NUM_WARMUPS = 0;

int get_comm_size(MPI_Comm comm) {
    int size;
    MPI_Comm_size(comm, &size);
    return size;
}

int get_comm_rank(MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    return rank;
}

auto now() { return std::chrono::steady_clock::now(); }

template <typename Data>
using AlltoallResult =
    std::tuple<std::vector<std::vector<Data>>, std::chrono::milliseconds>;

using AlltoallTopology = std::unordered_map<std::size_t, std::size_t>;

template <typename Data>
AlltoallResult<Data> complete_send_recv_alltoall(
    const std::vector<std::vector<Data>> &data, MPI_Datatype data_type,
    MPI_Comm comm) {
    const int size = get_comm_size(comm);

    // Exchange send / recv counts to preallocate recv buffer
    std::vector<int> send_counts(size);
    for (int pe = 0; pe < size; ++pe) {
        send_counts[pe] = data[pe].size();
    }
    std::vector<int> recv_counts(size);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
                 comm);

    std::vector<std::vector<Data>> result(size);
    for (int pe = 0; pe < size; ++pe) {
        result[pe].resize(recv_counts[pe]);
    }

    // Perform alltoall
    const auto start = now();
    std::vector<MPI_Request> requests(size);
    for (int pe = 0; pe < size; ++pe) {
        MPI_Isend(data[pe].data(), data[pe].size(), data_type, pe, 0, comm,
                  &requests[pe]);
    }
    for (int i = 0; i < size; ++i) {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, 0, comm, &status);
        int count;
        MPI_Get_count(&status, data_type, &count);
        const int pe = status.MPI_SOURCE;

        MPI_Recv(result[pe].data(), count, data_type, pe, 0, comm,
                 MPI_STATUS_IGNORE);
    }
    MPI_Waitall(size, requests.data(), MPI_STATUS_IGNORE);
    const auto end = now();

    const auto time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    return {std::move(result), time};
}

template <typename Data>
AlltoallResult<Data> mpi_alltoallv(const std::vector<std::vector<Data>> &data,
                                   MPI_Datatype data_type, MPI_Comm comm) {
    const int size = get_comm_size(comm);

    std::vector<int> send_counts(size);
    std::vector<int> recv_counts(size);
    std::vector<int> send_displs(size + 1);
    std::vector<int> recv_displs(size + 1);

    // Compute send counts + send displs
    for (int pe = 0; pe < size; ++pe) {
        send_counts[pe] = static_cast<int>(data[pe].size());
    }
    std::partial_sum(send_counts.begin(), send_counts.end(),
                     send_displs.begin() + 1);

    // Build buffer
    std::vector<Data> send_buf(send_displs.back());
    for (int pe = 0; pe < size; ++pe) {
        std::copy(data[pe].begin(), data[pe].end(),
                  send_buf.begin() + send_displs[pe]);
    }

    // Exchange send counts + send displs
    const auto start_alltoall = now();
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
                 comm);
    const auto end_alltoall = now();

    // Compute recv displs
    std::partial_sum(recv_counts.begin(), recv_counts.end(),
                     recv_displs.begin() + 1);

    // Exchange data
    std::vector<Data> recv_buf(recv_displs.back());
    const auto start_alltoallv = now();
    MPI_Alltoallv(send_buf.data(), send_counts.data(), send_displs.data(),
                  data_type, recv_buf.data(), recv_counts.data(),
                  recv_displs.data(), data_type, comm);
    const auto end_alltoallv = now();

    // Build output format
    std::vector<std::vector<Data>> result(size);
    for (int pe = 0; pe < size; ++pe) {
        result[pe].resize(recv_counts[pe]);
        std::copy(recv_buf.begin() + recv_displs[pe],
                  recv_buf.begin() + recv_displs[pe + 1], result[pe].begin());
    }

    const auto elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            (end_alltoall - start_alltoall) +
            (end_alltoallv - start_alltoallv));

    return {std::move(result), elapsed_time};
}

template <typename Data>
auto get_competitors() {
    return std::vector<std::pair<
        std::string,
        std::function<AlltoallResult<Data>(
            const std::vector<std::vector<Data>> &, MPI_Datatype, MPI_Comm)>>>{
        std::make_pair("MPI_Alltoallv", mpi_alltoallv<Data>),
        std::make_pair("Complete Isend+Recv",
                       complete_send_recv_alltoall<Data>),
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

void print_topology(AlltoallTopology &topology, MPI_Comm comm) {
    const int size = get_comm_size(comm);
    for (int from = 0; from < size; ++from) {
        for (int to = 0; to < size; ++to) {
            std::cout << topology[encode(from, to, size)];
        }
        std::cout << std::endl;
    }
}

template <typename Data>
void run_benchmark(AlltoallTopology topology, MPI_Datatype data_type,
                   MPI_Comm comm) {
    using namespace std::chrono_literals;

    const int size = get_comm_size(comm);
    const int rank = get_comm_rank(comm);

    // Generate data
    std::vector<std::vector<Data>> data(size);
    for (int pe = 0; pe < size; ++pe) {
        const std::size_t n = topology[encode(rank, pe, size)];
        data[pe].resize(n);
        generate_data(data[pe].data(), n);
    }

    // Total memory
    std::size_t total_nbytes = 0;
    for (const auto &[key, value] : topology) {
        total_nbytes += value;
    }
    total_nbytes *= sizeof(Data);

    // Warmup
    for (int round = 0; round < NUM_WARMUPS; ++round) {
        for (const auto &[name, competitor] : get_competitors<Data>()) {
            competitor(data, data_type, comm);
        }
    }

    // Measurement
    for (const auto &[name, competitor] : get_competitors<Data>()) {
        if (rank == 0) {
            std::cout << std::setw(30) << std::left << name;
        }

        auto total_time = 0ms;
        for (int round = 0; round < NUM_REPETITIONS; ++round) {
            const auto [ans, time] = competitor(data, data_type, comm);

            if (rank == 0) {
                std::cout << std::setw(6) << time.count() << std::flush;
                total_time += time;
            }
        }

        if (rank == 0) {
            const double mbs = 1.0 * total_nbytes / total_time.count() / 1000.0;
            std::cout << "== " << std::fixed << std::setprecision(0)
                      << std::setw(5) << mbs << " MB/s" << std::endl;
        }
    }
}

AlltoallTopology create_identitiy_topology(const std::size_t num_entries,
                                           MPI_Comm comm) {
    const int size = get_comm_size(comm);
    AlltoallTopology identitiy;
    for (int pe = 0; pe < size; ++pe) {
        identitiy[encode(pe, pe, size)] = num_entries;
    }
    return identitiy;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    const int size = get_comm_size(MPI_COMM_WORLD);
    const int rank = get_comm_rank(MPI_COMM_WORLD);

    run_benchmark<int>(create_identitiy_topology(100'000'000, MPI_COMM_WORLD),
                       MPI_INT, MPI_COMM_WORLD);

    MPI_Finalize();
}
