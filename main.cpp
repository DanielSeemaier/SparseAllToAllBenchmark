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

constexpr int NUM_REPETITIONS = 5;
constexpr int NUM_WARMUPS = 0;

constexpr bool csv = true;

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
AlltoallResult<Data> grid_alltoall(const std::vector<std::vector<Data>> &data,
                                   MPI_Datatype data_type, MPI_Comm comm) {
    const int size = get_comm_size(comm);
    const int rank = get_comm_rank(comm);
    const int grid_size = std::sqrt(size);
    if (grid_size * grid_size != size) {
        using namespace std::chrono_literals;
        return {std::vector<std::vector<Data>>{}, 0ms};
    }

    const int row = rank / grid_size;
    const int col = rank % grid_size;

    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(comm, row, rank, &row_comm);
    MPI_Comm_split(comm, col, rank, &col_comm);

    std::vector<int> data_counts(size);
    for (int pe = 0; pe < size; ++pe) {
        data_counts[pe] = data[pe].size();
    }

    // 1st hop send/recv counts/displs
    std::vector<int> row_send_counts(grid_size);
    for (int row = 0; row < grid_size; ++row) {
        for (int col = 0; col < grid_size; ++col) {
            const int pe = row * grid_size + col;
            row_send_counts[row] += data[pe].size();
        }
    }

    std::vector<int> row_recv_counts(grid_size);
    const auto start_1st_hop_counts = now();
    MPI_Alltoall(row_send_counts.data(), 1, MPI_INT, row_recv_counts.data(), 1,
                 MPI_INT, col_comm);
    const auto end_1st_hop_counts = now();

    std::vector<int> row_send_displs(grid_size + 1);
    std::vector<int> row_recv_displs(grid_size + 1);
    std::partial_sum(row_send_counts.begin(), row_send_counts.end(),
                     row_send_displs.begin() + 1);
    std::partial_sum(row_recv_counts.begin(), row_recv_counts.end(),
                     row_recv_displs.begin() + 1);

    std::vector<Data> row_send_buf(row_send_displs.back());
    int offset = 0;
    for (int pe = 0; pe < size; ++pe) {
        std::copy(data[pe].begin(), data[pe].end(),
                  row_send_buf.data() + offset);
        offset += data[pe].size();
    }

    std::vector<Data> row_recv_buf(row_recv_displs.back());

    // Exchange 1st hop payload
    const auto start_1st_hop = now();
    MPI_Alltoallv(row_send_buf.data(), row_send_counts.data(),
                  row_send_displs.data(), data_type, row_recv_buf.data(),
                  row_recv_counts.data(), row_recv_displs.data(), data_type,
                  col_comm);
    const auto end_1st_hop = now();

    // Exchange counts within payload
    std::vector<int> row_counts(size);

    const auto start_payload_counts = now();
    MPI_Alltoall(data_counts.data(), grid_size, MPI_INT, row_counts.data(),
                 grid_size, MPI_INT, col_comm);
    const auto end_payload_counts = now();

    std::vector<int> row_displs(size + 1);
    std::partial_sum(row_counts.begin(), row_counts.end(),
                     row_displs.begin() + 1);

    // Assertion:
    // row_data containts data for each PE in the same row as this PE:
    // col1, ..., coln, col1, ..., coln, ...
    // The sizes are given by row_counts:
    // size1, ..., sizen, size1, ..., sizen, ...
    // The displacements are given by row_displs, thus, the data for PE 1 (in
    // row_comm) is given by row_data[displs(1) ... displs(1) + size(1)] AND
    // row_data[displs(n+1) ... displs(n+1) + size(n+1)] AND ...

    std::vector<int> col_counts(grid_size);
    std::vector<int> col_subcounts(size);

    for (int row = 0; row < grid_size; ++row) {
        for (int col = 0; col < grid_size; ++col) {
            const int pe = row * grid_size + col;
            col_counts[col] += row_counts[pe];
            col_subcounts[row + col * grid_size] = row_counts[pe];
        }
    }

    std::vector<int> col_displs(grid_size + 1);
    std::partial_sum(col_counts.begin(), col_counts.end(),
                     col_displs.begin() + 1);

    std::vector<Data> col_data(row_recv_buf.size());
    for (int col = 0; col < grid_size; ++col) {
        int i = col_displs[col];
        for (int row = 0; row < grid_size; ++row) {
            const int pe = row * grid_size + col;
            const int row_displ = row_displs[pe];
            const int row_count = row_counts[pe];

            std::copy(row_recv_buf.begin() + row_displ,
                      row_recv_buf.begin() + row_displ + row_count,
                      col_data.begin() + i);
            i += row_count;
        }
    }

    // Exchange counts
    std::vector<int> col_recv_counts(grid_size);

    const auto start_2nd_hop_counts = now();
    MPI_Alltoall(col_counts.data(), 1, MPI_INT, col_recv_counts.data(), 1,
                 MPI_INT, row_comm);
    const auto end_2nd_hop_counts = now();

    std::vector<int> col_recv_displs(grid_size + 1);
    std::partial_sum(col_recv_counts.begin(), col_recv_counts.end(),
                     col_recv_displs.begin() + 1);

    // Exchange col payload
    std::vector<Data> col_recv_buf(col_recv_displs.back());

    const auto start_2nd_hop = now();
    MPI_Alltoallv(col_data.data(), col_counts.data(), col_displs.data(),
                  data_type, col_recv_buf.data(), col_recv_counts.data(),
                  col_recv_displs.data(), data_type, row_comm);
    const auto end_2nd_hop = now();

    std::vector<int> subcounts(size);

    const auto start_final_counts = now();
    MPI_Alltoall(col_subcounts.data(), grid_size, MPI_INT, subcounts.data(),
                 grid_size, MPI_INT, row_comm);
    const auto end_final_counts = now();

    std::vector<std::vector<Data>> result(size);

    int displ = 0;
    for (int col = 0; col < grid_size; ++col) {
        for (int row = 0; row < grid_size; ++row) {
            const int index = col * grid_size + row;
            const int pe = row * grid_size + col;
            const int size = subcounts[index];

            result[pe].resize(size);
            std::copy(col_recv_buf.begin() + displ,
                      col_recv_buf.begin() + displ + size, result[pe].begin());
            displ += size;
        }
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);

    const auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
        (end_1st_hop_counts - start_1st_hop_counts) +
        (end_1st_hop - start_1st_hop) +
        (end_payload_counts - start_payload_counts) +
        (end_2nd_hop_counts - start_2nd_hop_counts) +
        (end_2nd_hop - start_2nd_hop) +
        (end_final_counts - start_final_counts));
    return {std::move(result), time};
}

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

        auto total_time = 0ms;
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
                    const double mbs =
                        1.0 * total_nbytes / time.count() / 1000.0;
                    std::cout << size << "," << topology_name << ","
                              << data_size << "," << global_total_size << ","
                              << global_max_size << "," << global_avg_size
                              << "," << global_min_size << ","
                              << competitor_name << "," << round << ","
                              << time.count() << "," << mbs << std::endl;
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
            const double mbs = 1.0 * total_nbytes /
                               (total_time.count() / NUM_REPETITIONS) / 1000.0;
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
        if (csv) {
            std::cout
                << "MPI,Topology,ElementSize,TotalSize,MaxSize,AvgSize,MinSize,"
                   "Algorithm,Run,Time,MBs"
                << std::endl;
        }
    }

    for (std::size_t message_size : {1, 1 << 10, 1 << 20}) {
        run_benchmark<int>(
            "identity", create_identitiy_topology(message_size, MPI_COMM_WORLD),
            MPI_INT, MPI_COMM_WORLD);

        run_benchmark<int>(
            "complete", create_complete_topology(message_size, MPI_COMM_WORLD),
            MPI_INT, MPI_COMM_WORLD);

        run_benchmark<int>(
            "adjacent_cells",
            create_grid_adjacent_topology(message_size, MPI_COMM_WORLD),
            MPI_INT, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}
