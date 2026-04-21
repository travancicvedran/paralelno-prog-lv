#!/bin/bash
# -----------------------------------------------------------------------
# collect_results.sh
# Gathers all CSV performance lines from SLURM output files and prints
# a formatted table + computes speedup relative to the serial baseline.
#
# Usage:  bash collect_results.sh [output_directory]
# -----------------------------------------------------------------------

OUTDIR="${1:-.}"

echo "Collecting results from: $OUTDIR"
echo ""
echo "MPI_procs | OMP_threads | Total_cores | Dataset      | Time(s) | Throughput(M/s) | Speedup"
echo "---------|-------------|-------------|--------------|---------|-----------------|--------"

# Extract serial baseline (1 MPI, 1 OMP) for speedup calculation
BASELINE_TIME=$(grep -h '^CSV,' "$OUTDIR"/*.out 2>/dev/null | \
    awk -F',' '$2==1 && $3==1 {print $5; exit}')

if [ -z "$BASELINE_TIME" ]; then
    BASELINE_TIME=1   # avoid division by zero
fi

grep -h '^CSV,' "$OUTDIR"/*.out 2>/dev/null | \
    sort -t',' -k2,2n -k3,3n | \
    awk -F',' -v base="$BASELINE_TIME" \
    '{
        mpi=$2; omp=$3; n=$4; t=$5; tp=$6;
        cores = mpi * omp;
        speedup = base / t;
        printf "%9d | %11d | %11d | %12s | %7.4f | %15.2f | %.2fx\n",
               mpi, omp, cores, n, t, tp, speedup
    }'

echo ""
echo "Tip: Sort by Throughput descending to find optimal configuration."
