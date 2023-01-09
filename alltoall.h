#pragma once

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <sstream>
#include <tuple>
#include <vector>

#include "common.h"

template <typename Data>
using AlltoallResult = std::tuple<std::vector<std::vector<Data>>, Resolution>;

template <typename Data>
AlltoallResult<Data> grid_alltoall(const std::vector<std::vector<Data>> &data,
                                   MPI_Datatype data_type, MPI_Comm comm) {
    const int size = get_comm_size(comm);
    const int rank = get_comm_rank(comm);
    const int grid_size = std::sqrt(size);
    if (grid_size * grid_size != size) {
        using namespace std::chrono_literals;
        return {std::vector<std::vector<Data>>{}, 0s};
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

    // Exchange counts within payload
    std::vector<int> row_counts(size);

    const auto start_payload_counts = now();
    MPI_Alltoall(data_counts.data(), grid_size, MPI_INT, row_counts.data(),
                 grid_size, MPI_INT, col_comm);
    const auto end_payload_counts = now();

    // 1st hop send/recv counts/displs
    std::vector<int> row_send_counts(grid_size);
    for (int row = 0; row < grid_size; ++row) {
        for (int col = 0; col < grid_size; ++col) {
            const int pe = row * grid_size + col;
            row_send_counts[row] += data[pe].size();
        }
    }

    std::vector<int> row_recv_counts(grid_size);
    for (int row = 0; row < grid_size; ++row) {
        for (int col = 0; col < grid_size; ++col) {
            const int pe = row * grid_size + col;
            row_recv_counts[row] += row_counts[pe];
        }
    }

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
    std::vector<int> subcounts(size);
    const auto start_final_counts = now();
    MPI_Alltoall(col_subcounts.data(), grid_size, MPI_INT, subcounts.data(),
                 grid_size, MPI_INT, row_comm);
    const auto end_final_counts = now();

    std::vector<int> col_recv_counts(grid_size);
    for (int row = 0; row < grid_size; ++row) {
        int sum = 0;
        for (int col = 0; col < grid_size; ++col) {
            sum += subcounts[row * grid_size + col];
        }
        col_recv_counts[row] = sum;
    }

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

    const auto time = std::chrono::duration_cast<Resolution>(
        (end_1st_hop - start_1st_hop) +
        (end_payload_counts - start_payload_counts) +
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

    const auto time = std::chrono::duration_cast<Resolution>(end - start);
    return {std::move(result), time};
}

template <typename Data>
AlltoallResult<Data> sparse_send_recv_alltoall(
    const std::vector<std::vector<Data>> &data, MPI_Datatype data_type,
    MPI_Comm comm) {
    static int tag = 0;
    ++tag;

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

        MPI_Issend(data[pe].data(), data[pe].size(), data_type, pe, tag, comm,
                   &requests.back());
    }

    int isend_done = 0;
    while (isend_done == 0) {
        int iprobe_success = 1;
        while (iprobe_success > 0) {
            iprobe_success = 0;

            MPI_Status status;
            MPI_Iprobe(MPI_ANY_SOURCE, tag, comm, &iprobe_success, &status);
            if (iprobe_success) {
                int count;
                MPI_Get_count(&status, data_type, &count);
                const int pe = status.MPI_SOURCE;
                MPI_Recv(result[pe].data(), count, data_type, pe,
                         status.MPI_TAG, comm, MPI_STATUS_IGNORE);
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
            MPI_Iprobe(MPI_ANY_SOURCE, tag, comm, &iprobe_success, &status);
            if (iprobe_success) {
                int count;
                MPI_Get_count(&status, data_type, &count);
                const int pe = status.MPI_SOURCE;
                MPI_Recv(result[pe].data(), count, data_type, pe,
                         status.MPI_TAG, comm, MPI_STATUS_IGNORE);
            }
        }

        MPI_Test(&barrier_request, &ibarrier_done, MPI_STATUS_IGNORE);
    }
    const auto end = now();

    const auto time = std::chrono::duration_cast<Resolution>(end - start);
    return {std::move(result), time};
}

template <typename Data>
AlltoallResult<Data> sparse_send_recv_alltoall_new(
    const std::vector<std::vector<Data>> &data, MPI_Datatype data_type,
    MPI_Comm comm) {
    static int tag = 0;
    ++tag;

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

        MPI_Issend(data[pe].data(), data[pe].size(), data_type, pe, tag, comm,
                   &requests.back());
    }

    MPI_Request barrier_request = MPI_REQUEST_NULL;
    while (true) {
      int probe_sucess = false;
      MPI_Status status;
      MPI_Iprobe(MPI_ANY_SOURCE, tag, comm, &probe_sucess, &status);
      if (probe_sucess) {
        int count;
        MPI_Get_count(&status, data_type, &count);
        const int pe = status.MPI_SOURCE;
        MPI_Recv(result[pe].data(), count, data_type, pe, status.MPI_TAG, comm,
                 MPI_STATUS_IGNORE);
      }
      if (barrier_request != MPI_REQUEST_NULL) {
        int barrier_finished = false;
        MPI_Test(&barrier_request, &barrier_finished, MPI_STATUS_IGNORE);
        if (barrier_finished) {
          break;
        }
      } else {
        int all_sends_finished = false;
        MPI_Testall(requests.size(), requests.data(), &all_sends_finished, MPI_STATUSES_IGNORE);
        if (all_sends_finished) {
          MPI_Ibarrier(comm, &barrier_request);
        }
      }
    }
    const auto end = now();

    const auto time = std::chrono::duration_cast<Resolution>(end - start);
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
        return {std::vector<std::vector<Data>>{}, 0s};
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

    const auto time = std::chrono::duration_cast<Resolution>(end - start);
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

    const auto elapsed_time = std::chrono::duration_cast<Resolution>(
        (end_alltoall - start_alltoall) + (end_alltoallv - start_alltoallv));

    return {std::move(result), elapsed_time};
}

