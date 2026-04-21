#!/bin/bash
# -----------------------------------------------------------------------
# Scaling sweep: submit one job per configuration.
# Adjust DATASET_SIZE and cluster limits as needed.
# Usage:  bash submit_sweep.sh
# -----------------------------------------------------------------------

BINARY=./histogram_hybrid
DATASET=1000000000          # 10^9 integers
WALLTIME="00:10:00"
MEM="4G"

# Configurations: "nodes ntasks_per_node cpus_per_task"
CONFIGS=(
    "1 1  1"    #  1 MPI proc,   1 OMP thread  (serial baseline)
    "1 1  4"    #  1 MPI proc,   4 OMP threads
    "1 1  8"    #  1 MPI proc,   8 OMP threads
    "1 2  1"    #  2 MPI procs,  1 OMP thread each
    "1 2  4"    #  2 MPI procs,  4 OMP threads each
    "1 4  1"    #  4 MPI procs,  1 OMP thread each
    "1 4  2"    #  4 MPI procs,  2 OMP threads each
    "2 1  4"    #  2 nodes, 1 MPI proc/node, 4 threads
    "2 2  4"    #  2 nodes, 2 MPI procs/node, 4 threads
    "2 4  2"    #  2 nodes, 4 MPI procs/node, 2 threads
    "4 2  4"    #  4 nodes, 2 MPI procs/node, 4 threads
    "4 4  2"    #  4 nodes, 4 MPI procs/node, 2 threads
)

for cfg in "${CONFIGS[@]}"; do
    read -r nodes ntasks cpus <<< "$cfg"
    total_mpi=$(( nodes * ntasks ))
    total_cores=$(( total_mpi * cpus ))
    jobname="hist_n${nodes}_t${ntasks}_c${cpus}"

    echo "Submitting: nodes=$nodes ntasks/node=$ntasks cpus/task=$cpus  (${total_mpi} MPI × ${cpus} OMP = ${total_cores} cores)"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${jobname}
#SBATCH --output=${jobname}_%j.out
#SBATCH --error=${jobname}_%j.err
#SBATCH --nodes=${nodes}
#SBATCH --ntasks-per-node=${ntasks}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --time=${WALLTIME}
#SBATCH --mem=${MEM}

export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK

echo "=== Config: ${nodes} nodes | ${ntasks} tasks/node | ${cpus} cpus/task ==="
echo "OMP_NUM_THREADS=\$OMP_NUM_THREADS"
srun --mpi=pmix ${BINARY} ${DATASET}
EOF

done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Collect CSV lines with: grep '^CSV,' *.out"
