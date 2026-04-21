/**
 * Assignment 1: Histogram Computation
 * Hybrid MPI + OpenMP program to compute a histogram over a large dataset.
 *
 * Strategy:
 *   - MPI distributes data generation and counting across nodes/processes.
 *   - OpenMP parallelizes the per-process local histogram computation.
 *   - MPI_Reduce combines local histograms into the global result.
 *
 * Compile:
 *   mpicc -fopenmp -O2 -o histogram_hybrid histogram_hybrid.c
 *
 * Run locally (example: 4 MPI processes, 2 threads each):
 *   OMP_NUM_THREADS=2 mpirun -np 4 ./histogram_hybrid 1000000000
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

#define HIST_SIZE 256   /* values in [0, 255] */

/* ------------------------------------------------------------------
 * Tiny xorshift32 PRNG – lock-free, cheap, per-thread seeding.
 * Much faster than rand_r() for large datasets.
 * ------------------------------------------------------------------ */
static inline unsigned int xorshift32(unsigned int *state)
{
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

int main(int argc, char **argv)
{
    int rank, size;
    double t_start, t_end;

    /* ----------------------------------------------------------------
     * MPI initialisation
     * ---------------------------------------------------------------- */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ----------------------------------------------------------------
     * Parse dataset size from command-line argument
     * ---------------------------------------------------------------- */
    long long total_n = 1000000000LL; /* default: 10^9 */
    if (argc >= 2) {
        total_n = atoll(argv[1]);
        if (total_n <= 0) {
            if (rank == 0)
                fprintf(stderr, "Error: dataset size must be positive.\n");
            MPI_Finalize();
            return 1;
        }
    }

    /* ----------------------------------------------------------------
     * Partition the dataset evenly across MPI processes.
     * Last process gets the remainder.
     * ---------------------------------------------------------------- */
    long long base_chunk = total_n / size;
    long long remainder  = total_n % size;
    long long local_n    = base_chunk + (rank < remainder ? 1 : 0);

    int num_threads = 1;
#pragma omp parallel
    {
#pragma omp single
        num_threads = omp_get_num_threads();
    }

    if (rank == 0) {
        printf("=== Hybrid Histogram Computation ===\n");
        printf("Total dataset size : %lld\n", total_n);
        printf("MPI processes      : %d\n", size);
        printf("OpenMP threads     : %d (per process)\n", num_threads);
        printf("Total threads      : %d\n", size * num_threads);
        printf("=====================================\n");
        fflush(stdout);
    }

    /* ----------------------------------------------------------------
     * Allocate local histogram (one per process).
     * ---------------------------------------------------------------- */
    long long local_hist[HIST_SIZE];
    memset(local_hist, 0, sizeof(local_hist));

    /* ----------------------------------------------------------------
     * Parallel histogram computation using OpenMP.
     *
     * Each thread maintains a private partial histogram to avoid
     * atomic contention, then accumulates into the shared local_hist.
     * ---------------------------------------------------------------- */
    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();

#pragma omp parallel
    {
        /* Private histogram for this thread */
        long long priv_hist[HIST_SIZE];
        memset(priv_hist, 0, sizeof(priv_hist));

        int tid = omp_get_thread_num();

        /* Seed: mix rank, thread id and wall time for uniqueness */
        unsigned int seed = (unsigned int)(
            (rank + 1) * 1000003u ^
            (tid  + 1) * 999983u  ^
            (unsigned int)((size_t)(MPI_Wtime() * 1e9))
        );
        if (seed == 0) seed = 1; /* xorshift32 must not be seeded with 0 */

        /* Each thread processes its share of the local chunk */
#pragma omp for schedule(static)
        for (long long i = 0; i < local_n; i++) {
            unsigned int val = xorshift32(&seed) & 0xFFu; /* [0, 255] */
            priv_hist[val]++;
        }

        /* Merge private histogram into shared local histogram */
#pragma omp critical
        {
            for (int b = 0; b < HIST_SIZE; b++)
                local_hist[b] += priv_hist[b];
        }
    } /* end omp parallel */

    /* ----------------------------------------------------------------
     * MPI reduction: sum all local histograms into global histogram
     * on rank 0.
     * ---------------------------------------------------------------- */
    long long global_hist[HIST_SIZE];
    MPI_Reduce(local_hist, global_hist, HIST_SIZE,
               MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    t_end = MPI_Wtime();

    /* ----------------------------------------------------------------
     * Verification and output (rank 0 only)
     * ---------------------------------------------------------------- */
    if (rank == 0) {
        double elapsed = t_end - t_start;

        /* Verify: sum of all bins must equal total_n */
        long long total_counted = 0;
        for (int b = 0; b < HIST_SIZE; b++)
            total_counted += global_hist[b];

        printf("\nExecution time     : %.4f seconds\n", elapsed);
        printf("Throughput         : %.2f M integers/sec\n",
               (double)total_n / elapsed / 1e6);
        printf("Verification sum   : %lld (expected %lld) -> %s\n",
               total_counted, total_n,
               total_counted == total_n ? "PASS" : "FAIL");

        /* Print first 16 bins as a sanity check */
        printf("\nSample histogram bins [0..15]:\n");
        for (int b = 0; b < 16; b++) {
            printf("  bin[%3d] = %lld (%.4f%%)\n",
                   b, global_hist[b],
                   100.0 * global_hist[b] / total_n);
        }

        /* Print CSV-friendly performance line for easy data collection */
        printf("\nCSV,%d,%d,%lld,%.4f,%.2f\n",
               size, num_threads, total_n, elapsed,
               (double)total_n / elapsed / 1e6);
    }

    MPI_Finalize();
    return 0;
}
