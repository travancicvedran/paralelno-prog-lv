/*
gcc -fopenmp -o z2 .\LV2\z2.c
set OMP_NUM_THREADS=4 && z2.exe
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define BINS 256
#define CACHE_LINE 64

void histogram_serial(const int *data, long N, long *hist)
{
    memset(hist, 0, BINS * sizeof(long));
    for (long i = 0; i < N; i++)
        hist[data[i]]++;
}

void histogram_atomic(const int *data, long N, long *hist)
{
    memset(hist, 0, BINS * sizeof(long));

#pragma omp parallel for schedule(static)
    for (long i = 0; i < N; i++) {
#pragma omp atomic
        hist[data[i]]++;
    }
}

void benchmark(long N)
{
    printf("N = %ld\n", N);

    int *data = (int *)malloc(N * sizeof(int));

    srand(42);
    for (long i = 0; i < N; i++)
        data[i] = rand() % BINS;

    long *hist_serial   = calloc(BINS, sizeof(long));
    long *hist_atomic   = calloc(BINS, sizeof(long));
    double t0, t1;

    t0 = omp_get_wtime();
    histogram_serial(data, N, hist_serial);
    t1 = omp_get_wtime();
    printf("\nSerial\n");
    printf("Time = %.6f s\n", t1 - t0);
    double t_serial = t1 - t0;

    t0 = omp_get_wtime();
    histogram_atomic(data, N, hist_atomic);
    t1 = omp_get_wtime();
    printf("\nParallel - atomic\n");
    printf("Time = %.6f s  (speedup vs serial: %.2fx)\n",t1 - t0, t_serial / (t1 - t0));

    free(data);
    free(hist_serial);
    free(hist_atomic);
}

int main(void)
{
    printf("OpenMP Histogram Computation\n");
    printf("Threads = %d\n", omp_get_max_threads());

    benchmark(1000000);

    printf("\nDone.\n");
    return 0;
}