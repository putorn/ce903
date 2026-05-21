# ASTRA-sim Configuration Guide

> How to change simulation parameters and run different scenarios.
> For installation instructions, see [SETUP.md](SETUP.md).

---

## 0. Before Running — Set Up the Conda Environment

All Python tools (trace inspection, Chakra utilities) require the `p903` conda environment.

**First time setup — create from the shared environment file:**
```bash
conda env create -f environment.yml
```

**Every time you open a new terminal:**
```bash
conda activate p903
```

**Verify it is active:**
```bash
conda info --envs
# Should show * next to p903
```

> The `environment.yml` file is in the root of this repository. It pins all package versions so every teammate gets the same environment.

---

## How a Simulation Run Works

Every simulation run takes 4 input files and produces timing results:

```
--workload-configuration    What the GPUs are doing (Chakra trace)
--network-configuration     What the network looks like (topology, speed)
--system-configuration      How GPUs coordinate with each other
--remote-memory-configuration  How memory is handled (rarely changed)
          ↓
    ASTRA-sim runs
          ↓
log/log.log  →  Wall time & Comm time per NPU (in cycles)
```

---

## 1. Network Configuration

**File location:** `examples/network/analytical/`

**Available presets:**

| File | GPUs | Topology |
|------|------|----------|
| `Ring_4npus.yml` | 4 | Ring |
| `Ring_8npus.yml` | 8 | Ring |
| `Ring_16npus.yml` | 16 | Ring |
| `HGX-H100-validated.yml` | 8 | HGX H100 (real hardware model) |

**Parameters you can change:**

```yaml
topology: [ Ring ]       # Network shape — Ring is the most common
npus_count: [ 4 ]        # Number of GPUs
bandwidth: [ 50.0 ]      # Network speed in GB/s — higher = faster = fewer cycles
latency: [ 500.0 ]       # Delay per message in nanoseconds — lower = faster
```

**Example — doubling bandwidth:**
```yaml
bandwidth: [ 100.0 ]     # Was 50.0 — cycles will drop but not by half (latency still applies)
```

**Example — reducing latency:**
```yaml
latency: [ 100.0 ]       # Was 500.0 — bigger effect on small messages
```

**Rule of thumb:**
- `bandwidth` affects large data transfers most
- `latency` affects many small messages most

---

## 2. Workload Configuration

**File location:** `examples/workload/microbenchmarks/`

**Available workloads:**

| Folder | What it simulates |
|--------|------------------|
| `reduce_scatter/` | Each GPU sends a chunk, receives reduced result |
| `all_reduce/` | All GPUs share and combine gradients (most common in training) |
| `all_gather/` | One GPU's data is broadcast to all others |
| `all_to_all/` | Every GPU sends different data to every other GPU |

Each folder has subfolders by GPU count and data size, e.g.:
```
reduce_scatter/
└── 4npus_1MB/
    ├── reduce_scatter.0.et   ← GPU 0 trace
    ├── reduce_scatter.1.et   ← GPU 1 trace
    ├── reduce_scatter.2.et   ← GPU 2 trace
    └── reduce_scatter.3.et   ← GPU 3 trace
```

**How to switch workload** — change the path in `--workload-configuration`:
```bash
# Run all_reduce instead of reduce_scatter
--workload-configuration=examples/workload/microbenchmarks/all_reduce/4npus_1MB/all_reduce
```

---

## 3. System Configuration

**File location:** `examples/system/native_collectives/`

**Available presets:**

| File | Description |
|------|-------------|
| `Ring_4chunks.json` | Ring algorithm, 4 chunks per collective |
| `HGX-H100-validated.json` | Tuned for real H100 hardware |

**Key parameters:**

```json
{
  "scheduling-policy": "LIFO",          // Job scheduling order (LIFO or FIFO)
  "active-chunks-per-dimension": 1,     // Pipeline chunks — higher = more overlap
  "all-reduce-implementation": ["ring"],// Algorithm: ring, recursive_halving, etc.
  "reduce-scatter-implementation": ["ring"],
  "all-gather-implementation": ["ring"],
  "all-to-all-implementation": ["ring"],
  "local-mem-bw": 1600,                 // Local memory bandwidth in GB/s
  "peak-perf": 900                      // Peak compute in TFLOPS (used for roofline)
}
```

**Most likely to change:** `all-reduce-implementation` — when testing different collective algorithms.

---

## 4. Reading the Output

Results are written to `log/log.log` after each run.

```
sys[0] finished, 22240 cycles, exposed communication 22240 cycles.
sys[0], Wall time: 22240
sys[0], Comm time: 22240
```

| Field | Meaning |
|-------|---------|
| `sys[N]` | GPU number N |
| `Wall time` | Total simulated time (cycles) from start to finish |
| `Comm time` | Time spent sending data over the network |
| `Wall time == Comm time` | All time was communication — no compute overlap |
| `Wall time > Comm time` | Some compute happened in parallel with communication |

---

## 5. Quick Scenario Examples

**Scenario A — Faster network**
```bash
# Edit Ring_4npus.yml: bandwidth 50.0 → 100.0, then run:
./build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware \
    --workload-configuration=examples/workload/microbenchmarks/reduce_scatter/4npus_1MB/reduce_scatter \
    --system-configuration=examples/system/native_collectives/Ring_4chunks.json \
    --remote-memory-configuration=examples/remote_memory/analytical/no_memory_expansion.json \
    --network-configuration=examples/network/analytical/Ring_4npus.yml
```

**Scenario B — Different workload (all_reduce)**
```bash
./build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware \
    --workload-configuration=examples/workload/microbenchmarks/all_reduce/4npus_1MB/all_reduce \
    --system-configuration=examples/system/native_collectives/Ring_4chunks.json \
    --remote-memory-configuration=examples/remote_memory/analytical/no_memory_expansion.json \
    --network-configuration=examples/network/analytical/Ring_4npus.yml
```

**Scenario C — More GPUs (8 NPUs)**
```bash
./build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware \
    --workload-configuration=examples/workload/microbenchmarks/all_reduce/8npus_1MB/all_reduce \
    --system-configuration=examples/system/native_collectives/Ring_4chunks.json \
    --remote-memory-configuration=examples/remote_memory/analytical/no_memory_expansion.json \
    --network-configuration=examples/network/analytical/Ring_8npus.yml
```

---

## 6. Loading Custom Chakra Traces (Sprint 3+)

When the team produces real training traces from a Hugging Face workload:

1. Activate the conda environment first:
   ```bash
   conda activate p903
   ```

2. Install Chakra Python tools:
   ```bash
   pip install -e extern/graph_frontend/chakra
   ```

3. Place `.et` files with a shared prefix, one per GPU:
   ```
   my_gpt2_trace.0.et
   my_gpt2_trace.1.et
   my_gpt2_trace.2.et
   my_gpt2_trace.3.et
   ```

4. Run with the prefix path (without `.N.et`):
   ```bash
   --workload-configuration=/path/to/my_gpt2_trace
   ```

ASTRA-sim automatically loads each GPU's file based on NPU index.
