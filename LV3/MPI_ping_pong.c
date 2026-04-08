#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank;
    MPI_Status status;
    double data;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        data = 1.0;
        original = data;
        printf("Process %d: Sending value = %f to process 1\n", rank, data);
        MPI_Send(&data, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);

        MPI_Recv(&data, 1, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &status);
        printf("\nProcess %d: Received value = %f from process %d with tag %d\n", rank, data, status.MPI_SOURCE, status.MPI_TAG);

        if (data == original + 1.0) {
            printf("Process %d: ✓ Data integrity verified\n", rank);
        } else {
            printf("Process %d: ✗ Data corruption!\n", rank);
        }
    }

    else if (rank == 1) {
        MPI_Recv(&data, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        printf("Process %d: Received value = %f from process %d with tag %d\n", rank, data, status.MPI_SOURCE, status.MPI_TAG);

        data++;

        printf("Process %d: Sending value = %f back to process 0\n", rank, data);
        MPI_Send(&data, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}