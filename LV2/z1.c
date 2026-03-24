/*
gcc -fopenmp -o z1 .\LV2\z1.c
set OMP_NUM_THREADS=4 && z1.exe
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* fills matrix in row-major order */
void init_matrix(double *matrix, int N, const char *schedule_name, int chunk)
{
    double t_start = omp_get_wtime();

#pragma omp parallel for schedule(runtime)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i * N + j] = (double)(i * N + j);
        }
    }

    double t_end = omp_get_wtime();
    printf("  %-10s chunk=%-4d  time = %.6f s\n",
           schedule_name, chunk, t_end - t_start);
}

void benchmark(int N)
{
    printf("\nN = %d",N);

    double *matrix = (double *)malloc((size_t)N * N * sizeof(double));

    omp_set_schedule(omp_sched_static, 1);
    init_matrix(matrix, N, "static", 1);

    omp_set_schedule(omp_sched_dynamic, 1);
    init_matrix(matrix, N, "dynamic", 1);

    omp_set_schedule(omp_sched_guided, 1);
    init_matrix(matrix, N, "guided", 1);

    printf("%.0f",matrix[0]);

    free(matrix);
}

int main(void)
{
    printf("OpenMP Parallel Matrix Initialization\n");
    printf("Threads = %d\n", omp_get_max_threads());

    benchmark(500);

    printf("\nDone.\n");
    return 0;
}