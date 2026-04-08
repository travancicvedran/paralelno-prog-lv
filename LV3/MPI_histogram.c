#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define ARRAY_SIZE 10000000  // 10^7
#define BIN_COUNT 256        // Histogram bins [0-255]

void compute_histogram(int* data, int size, int* hist) {
    for (int i = 0; i < size; i++) {
        hist[data[i]]++;
    }
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int* data = NULL;
    int local_hist[BIN_COUNT] = {0};
    int global_hist[BIN_COUNT] = {0};
    
    double start_time, end_time;
    
    // Calculate segment sizes properly (handle remainder)
    int base_segment_size = ARRAY_SIZE / size;
    int remainder = ARRAY_SIZE % size;
    int local_size = base_segment_size + (rank < remainder ? 1 : 0);
    
    // Calculate displacements for Scatterv
    int* sendcounts = NULL;
    int* displs = NULL;
    
    if (rank == 0) {
        // Allocate and initialize the full array
        data = (int*)malloc(ARRAY_SIZE * sizeof(int));
        srand(time(NULL));
        for (int i = 0; i < ARRAY_SIZE; i++) {
            data[i] = rand() % BIN_COUNT;
        }
        
        // Prepare scatter parameters
        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        int offset = 0;
        for (int i = 0; i < size; i++) {
            sendcounts[i] = base_segment_size + (i < remainder ? 1 : 0);
            displs[i] = offset;
            offset += sendcounts[i];
        }
        
        printf("Starting parallel histogram computation with %d processes\n", size);
        printf("Array size: %d, segments distributed with varying sizes\n", ARRAY_SIZE);
        
        start_time = MPI_Wtime();
    }
    
    // Allocate local buffer
    int* local_data = (int*)malloc(local_size * sizeof(int));
    
    // FIXED: Proper non-blocking scatter
    if (rank == 0) {
        MPI_Request* send_requests = (MPI_Request*)malloc((size - 1) * sizeof(MPI_Request));
        int req_count = 0;
        
        // Post non-blocking sends to all other processes
        for (int i = 1; i < size; i++) {
            int send_size = base_segment_size + (i < remainder ? 1 : 0);
            MPI_Isend(data + displs[i], send_size, MPI_INT, i, 0, MPI_COMM_WORLD, &send_requests[req_count++]);
        }
        
        // Copy process 0's own data
        for (int i = 0; i < local_size; i++) {
            local_data[i] = data[displs[0] + i];
        }
        
        // Wait for all sends to complete
        MPI_Waitall(req_count, send_requests, MPI_STATUSES_IGNORE);
        free(send_requests);
    } else {
        // Other processes post non-blocking receive
        MPI_Request recv_request;
        MPI_Irecv(local_data, local_size, MPI_INT, 0, 0, MPI_COMM_WORLD, &recv_request);
        MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
    }
    
    // Compute local histogram
    compute_histogram(local_data, local_size, local_hist);
    
    // Aggregate histograms
    MPI_Reduce(local_hist, global_hist, BIN_COUNT, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        end_time = MPI_Wtime();
        double parallel_time = end_time - start_time;
        
        // Serial implementation for comparison
        int* serial_hist = (int*)calloc(BIN_COUNT, sizeof(int));
        double serial_start = MPI_Wtime();
        compute_histogram(data, ARRAY_SIZE, serial_hist);
        double serial_end = MPI_Wtime();
        double serial_time = serial_end - serial_start;
        
        // Verify correctness
        int correct = 1;
        for (int i = 0; i < BIN_COUNT; i++) {
            if (serial_hist[i] != global_hist[i]) {
                correct = 0;
                printf("Mismatch at bin %d: serial=%d, parallel=%d\n", 
                       i, serial_hist[i], global_hist[i]);
            }
        }
        
        printf("\n=== Results ===\n");
        printf("Parallel time: %f seconds\n", parallel_time);
        printf("Serial time:   %f seconds\n", serial_time);
        printf("Speedup:       %fx\n", serial_time / parallel_time);
        printf("Correctness:   %s\n", correct ? "PASSED" : "FAILED");
        
        // Print first few bins as verification
        printf("\nFirst 10 histogram bins:\n");
        for (int i = 0; i < 10; i++) {
            printf("Bin %3d: %d\n", i, global_hist[i]);
        }
        
        free(serial_hist);
        free(sendcounts);
        free(displs);
    }
    
    free(local_data);
    if (rank == 0) free(data);
    
    MPI_Finalize();
    return 0;
}