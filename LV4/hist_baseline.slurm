#!/bin/bash
#SBATCH --job-name=hist_baseline
#SBATCH --output=hist_baseline_%j.out
#SBATCH --error=hist_baseline_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --mem=4G

# -----------------------------------------------------------------------
# Baseline hybrid run:
#   2 nodes × 2 MPI tasks × 4 OpenMP threads = 16 total threads
# -----------------------------------------------------------------------

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job ID          : $SLURM_JOB_ID"
echo "Nodes           : $SLURM_JOB_NUM_NODES"
echo "Tasks per node  : $SLURM_NTASKS_PER_NODE"
echo "CPUs per task   : $SLURM_CPUS_PER_TASK"
echo "OMP_NUM_THREADS : $OMP_NUM_THREADS"
echo "Node list       : $SLURM_JOB_NODELIST"
echo "-------------------------------------------"

# Run with 10^9 integers
srun --mpi=pmix ./histogram_hybrid 1000000000
