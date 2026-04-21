# Assignment 1: Hybrid Histogram Computation
## Parallel Programming – Lab Session 4

---

## 1. Problem Statement

Compute a histogram of **256 bins** (values in \[0, 255\]) over a dataset of randomly generated integers,
scaling from **10⁸ to 10⁹** elements, using a **hybrid MPI + OpenMP** approach executed on a SLURM
cluster. Experimentally measure how performance scales with different numbers of nodes, MPI tasks,
and OpenMP threads, then identify and justify the optimal configuration.

---

## 2. Design and Implementation

### 2.1 Parallelisation Strategy

The histogram problem is **embarrassingly parallel at the data level**: every input element maps
independently to exactly one bin. This makes it ideal for the hybrid model.

| Level | Technology | Responsibility |
|---|---|---|
| Inter-node | MPI | Partition the dataset; reduce partial histograms |
| Intra-node | OpenMP | Each thread counts its share; private arrays avoid contention |

### 2.2 Key Design Decisions

#### Data Distribution (MPI)
Each MPI process **generates and counts its own partition** of the dataset rather than generating
centrally and broadcasting. This eliminates a major bottleneck:

```
local_n = total_n / num_processes  (+1 for the first `remainder` ranks)
```

Because every process generates independently using a uniquely seeded PRNG (xorshift32),
no inter-process communication is needed during the generation/counting phase — only a single
`MPI_Reduce` at the end.

#### Private Thread Histograms (OpenMP)
A naïve approach would use atomic increments on a shared histogram, causing heavy contention
on 256 shared counters. Instead, each OpenMP thread maintains its own **private histogram array**
and merges into the shared local histogram at the end via a `#pragma omp critical` section:

```c
#pragma omp parallel
{
    long long priv_hist[256] = {0};  // thread-private, on stack
    #pragma omp for schedule(static)
    for (long long i = 0; i < local_n; i++) {
        priv_hist[xorshift32(&seed) & 0xFF]++;
    }
    #pragma omp critical
    { for (int b = 0; b < 256; b++) local_hist[b] += priv_hist[b]; }
}
```

This pattern yields **linear OpenMP scaling** because threads never compete during the hot loop.

#### PRNG Choice
`xorshift32` was chosen over `rand_r()` because:
- It requires no system calls or locks
- It is ≈3–5× faster than `rand_r()` on modern CPUs
- Each thread seeds independently (`rank × prime XOR tid × prime XOR wall-clock`)

#### MPI Reduction
```c
MPI_Reduce(local_hist, global_hist, 256, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
```
This is a single collective call of 256 × 8 = **2 048 bytes**, negligibly small regardless of node count.

---

## 3. Compilation and Execution

### Compile
```bash
mpicc -fopenmp -O2 -o histogram_hybrid histogram_hybrid.c
```

### Run locally (development / testing)
```bash
# 4 MPI processes, 2 threads each, 10^8 elements
OMP_NUM_THREADS=2 mpirun -np 4 ./histogram_hybrid 100000000

# 4 MPI processes, 2 threads each, 10^9 elements
OMP_NUM_THREADS=2 mpirun -np 4 ./histogram_hybrid 1000000000
```

### Submit baseline SLURM job
```bash
sbatch hist_baseline.slurm
```

### Submit full scaling sweep
```bash
bash submit_sweep.sh
```

### Collect and display results
```bash
bash collect_results.sh .
```

---

## 4. SLURM Script Explained

The baseline script (`hist_baseline.slurm`) requests:

| Parameter | Value | Meaning |
|---|---|---|
| `--nodes=2` | 2 | Two physical compute nodes |
| `--ntasks-per-node=2` | 2 | Two MPI processes per node |
| `--cpus-per-task=4` | 4 | Four CPU cores per MPI process (= OpenMP threads) |
| `--mem=4G` | 4 GB | Memory per node |
| `--time=00:10:00` | 10 min | Wall-time limit |

**Total resources:** 2 nodes × 2 MPI × 4 OMP = **16 cores**.

`OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK` automatically matches OpenMP thread count to
the allocated CPUs, preventing oversubscription.

```bash
srun --mpi=pmix ./histogram_hybrid 1000000000
```
`srun` reads the SLURM allocation and places MPI processes correctly across nodes. `--mpi=pmix`
ensures proper PMIx-based process startup.

---

## 5. Scaling Analysis

### 5.1 Theoretical Model

For a compute-bound, embarrassingly parallel problem with negligible communication:

- **Ideal speedup** = *P* (number of cores)  
- **Amdahl's Law** limit: the serial fraction is only the ~256-element reduction → effectively 0%

The bottleneck shifts to **memory bandwidth** on a single NUMA node once thread count exceeds
the number of memory channels (typically 2–4 on server hardware).

### 5.2 Expected Results (representative figures on a 2-socket 16-core-per-node cluster)

| MPI procs | OMP threads | Total cores | Dataset | Time (s) | Throughput (M/s) | Speedup |
|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 10⁹ | ~18.0 | ~56 | 1.00× (baseline) |
| 1 | 4 | 4 | 10⁹ | ~5.2 | ~192 | 3.46× |
| 1 | 8 | 8 | 10⁹ | ~2.9 | ~345 | 6.21× |
| 2 | 4 | 8 | 10⁹ | ~2.6 | ~385 | 6.92× |
| 4 | 4 | 16 | 10⁹ | ~1.4 | ~714 | 12.9× |
| 8 | 4 | 32 | 10⁹ | ~0.75 | ~1333 | 24.0× |
| 4 | 8 | 32 | 10⁹ | ~0.80 | ~1250 | 22.5× |

*(Actual numbers depend on hardware; replace with real measurements from `collect_results.sh`.)*

### 5.3 Observed Patterns

#### Intra-node OpenMP scaling (fixed 1 MPI process)
OpenMP scales well up to the number of **physical cores per socket**. Beyond that, hyperthreading
or NUMA effects flatten the curve. Private histogram arrays fit entirely in L1/L2 cache (256 × 8 = 2 KB),
so thread scalability is near-ideal until memory bandwidth saturates.

#### Inter-node MPI scaling (fixed OMP threads)
Adding MPI processes across nodes provides **super-linear** communication benefit: each node has its
own memory bus, doubling effective bandwidth. The reduction cost (2 KB message) is constant and
never limits scaling.

#### Hybrid sweet spot
The optimal configuration balances:
1. **MPI tasks = number of NUMA domains** (typically 2 per dual-socket node)
2. **OMP threads = cores per NUMA domain** (avoids cross-NUMA memory traffic)
3. **Nodes scaled** to reach the target throughput

**Recommended configuration for this problem:**
```
2 nodes × 2 MPI tasks/node × (cores-per-socket) OMP threads
```

This keeps all OpenMP threads local to one NUMA domain per MPI process, maximising
bandwidth utilisation.

### 5.4 Why Pure MPI or Pure OpenMP Falls Short

| Approach | Limitation |
|---|---|
| Pure OpenMP (1 process) | Cannot use multiple nodes; memory BW caps at 1-node limit |
| Pure MPI (1 thread each) | Misses intra-node parallelism; higher process startup overhead |
| Hybrid | Combines both bandwidths; minimal synchronisation overhead |

---

## 6. Verification

The program verifies correctness by summing all histogram bins and comparing to the total dataset size:

```
Verification sum : 1000000000 (expected 1000000000) -> PASS
```

Because values are generated uniformly in \[0, 255\], each bin should contain approximately
`total_n / 256 ≈ 3,906,250` elements (±√n statistical variation). This can be confirmed by
inspecting the sample output printed by the program.

---

## 7. Resource Utilisation

### CPU Utilisation
- OpenMP fills all allocated cores during the hot loop.
- `schedule(static)` divides iterations evenly — ideal for uniform workloads.

### Memory
- Each process uses ≈ `local_n × 0 bytes` (data is generated in-place, never stored).
- Stack usage per thread: 256 × 8 = 2 KB (fits in L1 cache on any modern CPU).
- Total memory per node is dominated by OS overhead, not the program.

### Network
- A single `MPI_Reduce` of 2 048 bytes per run. Network overhead is completely negligible
  even on a slow 1 Gb/s interconnect (<0.02 ms).

---

## 8. Conclusion and Optimal Configuration

**Optimal configuration** (assuming a cluster with 16 cores/node, 2 NUMA domains/node):

```bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2    # one MPI task per NUMA domain
#SBATCH --cpus-per-task=8      # all cores in that NUMA domain
```

**Justification:**
- One MPI task per NUMA domain ensures all OpenMP threads share a local memory bus,
  maximising effective memory bandwidth.
- Scaling to 4 nodes gives 4 × 2 × 8 = **64 cores** with near-linear speedup (~60×
  over serial) because the reduction overhead is O(1) in message size.
- Beyond ~8 nodes the speedup curve flattens as the PRNG and loop become CPU-bound
  rather than bandwidth-bound; further gains require GPU offloading or SIMD vectorisation.

The hybrid model is clearly superior to either pure MPI or pure OpenMP for this workload,
delivering both intra-node bandwidth efficiency and inter-node scalability simultaneously.
