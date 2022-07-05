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

using AlltoallTopology = std::vector<std::size_t>;

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
    MPI_Waitall(size, requests.data(), MPI_STATUSES_IGNORE);
    const auto end = now();

    const auto time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    return {std::move(result), time};
}

template <typename Data>
AlltoallResult<Data> sparse_send_recv_alltoall(
    const std::vector<std::vector<Data>> &data, MPI_Datatype data_type,
    MPI_Comm comm) {
    const int size = get_comm_size(comm);
    const int rank = get_comm_rank(comm);
    std::vector<MPI_Request> requests;
    requests.reserve(size);
    std::vector<std::uint8_t> send_message_to(size);

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

    const auto start = now();
    for (int pe = 0; pe < size; ++pe) {
        if (data[pe].empty()) {
            continue;
        }

        send_message_to[pe] = 1;
        requests.emplace_back();

        MPI_Issend(data[pe].data(), data[pe].size(), data_type, pe, 0, comm,
                   &requests.back());
    }

    int isend_done = 0;
    while (isend_done == 0) {
        int iprobe_success = 1;
        while (iprobe_success > 0) {
            iprobe_success = 0;

            MPI_Status status;
            MPI_Iprobe(MPI_ANY_SOURCE, 0, comm, &iprobe_success, &status);
            if (iprobe_success) {
                int count;
                MPI_Get_count(&status, data_type, &count);
                const int pe = status.MPI_SOURCE;
                MPI_Recv(result[pe].data(), count, data_type, pe, 0, comm,
                         MPI_STATUS_IGNORE);
            }
        }

        isend_done = 0;
        MPI_Testall(requests.size(), requests.data(), &isend_done,
                    MPI_STATUSES_IGNORE);
    }

    MPI_Request barrier_request;
    MPI_Ibarrier(comm, &barrier_request);

    int ibarrier_done = 0;
    while (ibarrier_done == 0) {
        int iprobe_success = 1;
        while (iprobe_success > 0) {
            iprobe_success = 0;

            MPI_Status status;
            MPI_Iprobe(MPI_ANY_SOURCE, 0, comm, &iprobe_success, &status);
            if (iprobe_success) {
                int count;
                MPI_Get_count(&status, data_type, &count);
                const int pe = status.MPI_SOURCE;
                MPI_Recv(result[pe].data(), count, data_type, pe, 0, comm,
                         MPI_STATUS_IGNORE);
            }
        }

        MPI_Test(&barrier_request, &ibarrier_done, MPI_STATUS_IGNORE);
    }
    const auto end = now();

    const auto time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    return {std::move(result), time};
}

template <typename Data>
AlltoallResult<Data> mpi_alltoall(const std::vector<std::vector<Data>> &data,
                                  MPI_Datatype data_type, MPI_Comm comm) {
    const int size = get_comm_size(comm);

    // Check if applicable
    bool all_same = true;
    for (int pe = 1; pe < size; ++pe) {
        all_same &= data[pe].size() == data[pe - 1].size();
    }
    long all_same_key = all_same ? data[0].size() : -1;
    std::vector<long> all_keys(size);
    MPI_Allgather(&all_same_key, 1, MPI_LONG, all_keys.data(), 1, MPI_LONG,
                  comm);
    for (int pe = 1; pe < size; ++pe) {
        all_same &= all_keys[pe] >= 0 && all_keys[pe] == all_keys[pe - 1];
    }

    if (!all_same) {
        using namespace std::chrono_literals;
        return {std::vector<std::vector<Data>>{}, 0ms};
    }

    const std::size_t elements_per_pe = data[0].size();

    std::vector<Data> send_buf(elements_per_pe * size);
    for (int pe = 0; pe < size; ++pe) {
        std::copy(data[pe].begin(), data[pe].end(),
                  send_buf.begin() + elements_per_pe * pe);
    }

    std::vector<Data> recv_buf(elements_per_pe * size);
    const auto start = now();
    MPI_Alltoall(send_buf.data(), elements_per_pe, data_type, recv_buf.data(),
                 elements_per_pe, data_type, comm);
    const auto end = now();

    std::vector<std::vector<Data>> result(size,
                                          std::vector<Data>(elements_per_pe));
    for (int pe = 0; pe < size; ++pe) {
        std::copy(recv_buf.begin() + elements_per_pe * pe,
                  recv_buf.begin() + elements_per_pe * (pe + 1),
                  result[pe].begin());
    }

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
        std::make_pair("MPI_Alltoall", mpi_alltoall<Data>),
        std::make_pair("Complete Isend+Recv",
                       complete_send_recv_alltoall<Data>),
        std::make_pair("Sparse Isend+Recv", sparse_send_recv_alltoall<Data>),
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

template <typename Data>
void run_benchmark(AlltoallTopology topology, MPI_Datatype data_type,
                   MPI_Comm comm) {
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

    // Total memory
    long local_total_nbytes = 0;
    for (auto nbytes : topology) {
        local_total_nbytes += nbytes;
    }
    long total_nbytes;
    MPI_Allreduce(&local_total_nbytes, &total_nbytes, 1, MPI_LONG, MPI_SUM,
                  comm);

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
                std::cout << std::setw(7);
                if (ans.empty()) {
                    std::cout << "-" << std::flush;
                } else {
                    std::cout << time.count() << std::flush;
                }
                total_time += time;
            }
        }

        if (rank == 0) {
            const double mbs = 1.0 * total_nbytes / total_time.count() / 1000.0;
            std::cout << "== " << std::fixed << std::setprecision(0)
                      << std::setw(5) << mbs << " MB/s" << std::endl;
        }
    }

    if (rank == 0) {
        std::cout << std::endl;
    }
}

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

    return topology;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    const int size = get_comm_size(MPI_COMM_WORLD);
    const int rank = get_comm_rank(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "MPI_SIZE=" << size << std::endl;
        std::cout << std::endl;
    }

    long num_entries;

    num_entries = 100'000'000;
    if (rank == 0) {
        std::cout << "Identity topology with " << num_entries / 1'000'000 << " M * 4 bytes"
                  << std::endl;
    }
    run_benchmark<int>(create_identitiy_topology(num_entries, MPI_COMM_WORLD),
                       MPI_INT, MPI_COMM_WORLD);

    num_entries = 10'000'000;
    if (rank == 0) {
        std::cout << "Complete topology with " << num_entries / 1'000'000 << " M * 4 bytes"
                  << std::endl;
    }
    run_benchmark<int>(create_complete_topology(num_entries, MPI_COMM_WORLD),
                       MPI_INT, MPI_COMM_WORLD);

    num_entries = 25'000'000;
    if (rank == 0) {
        std::cout << "Grid-adjacent topology with " << num_entries / 1'000'000
                  << " M * 4 bytes" << std::endl;
    }
    run_benchmark<int>(
        create_grid_adjacent_topology(num_entries, MPI_COMM_WORLD), MPI_INT,
        MPI_COMM_WORLD);

    MPI_Finalize();
}
