# CE903 Group Project — Network for LLM Training

Role 2: ASTRA-sim & Simulation Setup · Sprint 4

---

## Repository Structure

```
ce903/
├── astra-sim/                   ASTRA-sim 2.0 simulator (built, ready to run)
├── pod_a_pipeline/              TC-01 + TC-03 — trace-to-sim pipeline & fault injection
│   ├── workloads/               Chakra .et trace files (4 NPU)
│   ├── workloads_failure/       Chakra .et trace files (3 NPU — fault scenario)
│   ├── configs/
│   │   ├── network/             Ring topology configs (4 NPU, 3 NPU)
│   │   └── system/              ASTRA-sim system configs
│   ├── results/                 Baseline simulation logs + observability contract
│   ├── results_with_failure/    Failure simulation logs
│   ├── results_fault_run/       C++ injection run logs + fault_events.csv
│   ├── run_pipeline.sh          TC-01 — generate trace → simulate → validate
│   ├── run_fault_injection.py   TC-03 — Python fault injection (critical bar)
│   ├── klye_code.py             Measurement harness — JCT penalty + NCCL stall
│   ├── sprint4_results.py       Final results report
│   ├── fault_events.csv         Fault timestamp T0 (spec-calculated)
│   └── stall_events.csv         Stall detection time (spec-calculated)
├── pod_b_traffic/               TC-02 — ring all-reduce baseline sweep
│   ├── workloads/               Sweep trace files
│   ├── results/                 9 simulation logs (3 sizes × 3 NPU counts)
│   ├── generate_workloads.sh    Generate sweep traces
│   ├── run_baseline.sh          Run 9 simulations + analyse
│   └── analyze_results.py       Analyse and display sweep results
├── fault_injection_hook/        C++ injection hook (stretch goal — Sprint 4)
│   ├── Device.h / Device.cpp    Modified — adds disable() flag to NPU
│   ├── Topology.h / Topology.cpp Modified — adds disable_npu() method
│   ├── main.cc                  Modified — injects fault at specified timestep
│   ├── run_c_injection.py       End-to-end C++ injection runner
│   ├── apply_patch.sh           Apply patch to ASTRA-sim + rebuild
│   └── revert.sh                Restore original ASTRA-sim + rebuild
├── runpod/                      GPU profiling scripts (RunPod A100)
│   ├── kyle_profiler/           GPT-2 profiler — exports execution trace
│   └── roshan_converter/        Converts profiler traces → Chakra .et format
├── roshan_converter/            Local trace converter (Mac version)
├── SETUP.md                     Build guide (macOS / Linux / Windows WSL2)
└── environment.yml              Conda environment (Python 3.11)
```

---

## Setup

**Step 1 — Create conda environment**
```bash
conda env create -f environment.yml
conda activate p903
```

**Step 2 — Install pip packages**
```bash
pip install -r requirements.txt
```

**Step 3 — Build ASTRA-sim**

macOS:
```bash
export PATH="/opt/homebrew/opt/coreutils/libexec/gnubin:/opt/homebrew/bin:$PATH"
export PROTOBUF_FROM_SOURCE=True
bash astra-sim/build/astra_analytical/build.sh -t all
```

Linux:
```bash
bash astra-sim/build/astra_analytical/build.sh -t all
```

> TC-01, TC-02 and TC-03 simulations require the ASTRA-sim binary to be built first.

---

## TC-01 — Trace-to-Sim Pipeline

Validates the full pipeline: trace generation → ASTRA-sim → metric validation.

```bash
cd pod_a_pipeline
bash run_pipeline.sh
```

Results in `pod_a_pipeline/results/`. Key metrics:

| Metric | Value |
|--------|-------|
| JCT (Wall time) | 95,767,236 cycles |
| GPU time | 80,000,000 cycles (83.5%) |
| Comm time (AllReduce) | 15,767,236 cycles (16.5%) |

To use real GPT-2 traces instead of synthetic:
1. Copy `gpt2.0.et … gpt2.3.et` into `pod_a_pipeline/workloads/`
2. Set `WORKLOAD_PREFIX="gpt2"` in `run_pipeline.sh`
3. Re-run `bash run_pipeline.sh`

---

## TC-02 — Ring AllReduce Baseline Sweep

Benchmarks ring AllReduce across 3 message sizes × 3 NPU counts = 9 simulations.

```bash
cd pod_b_traffic
bash generate_workloads.sh
bash run_baseline.sh
```

Results in `pod_b_traffic/results/`. Key finding: bus bandwidth saturates at 16 NPUs at GPT-2 scale.

| Message Size | NPUs | Comm Time (cycles) | Ring Efficiency |
|---|---|---|---|
| 256 MB (BF16) | 4 | 398,542 | 1.5000 |
| 256 MB (BF16) | 16 | 1,159,300 | 1.8750 |
| 256 MB (BF16) | 64 | 3,446,957 | 1.9688 |
| 512 MB (FP32) | 4 | 3,630,842 | 1.5000 |
| 512 MB (FP32) | 16 | 4,539,234 | 1.8750 |
| 512 MB (FP32) | 64 | 7,554,490 | 1.9688 |

---

## TC-03 — Fault Injection & JCT Recovery Penalty

Simulates a GSP Error (XID 119) — the highest-priority GPU failure type
(100% job-failure probability, 99% cascade probability).

### Python approach (critical bar)

Runs ASTRA-sim twice — baseline (4 NPU) and failure (3 NPU after NPU 0 removed):

```bash
cd pod_a_pipeline
conda run -n p903 python run_fault_injection.py
```

### View final results

```bash
conda run -n p903 python pod_a_pipeline/sprint4_results.py
```

### View full metric tables (JCT Penalty + NCCL Stall Duration)

```bash
conda run -n p903 python pod_a_pipeline/klye_code.py
```

### Results

| Metric | Value |
|--------|-------|
| Baseline JCT | 95,767,236 cycles |
| Failure JCT | 98,680,744 cycles |
| **JCT Recovery Penalty** | **2,913,508 cycles (3.0%)** |
| NCCL Stall Duration | 1,031,879 cycles |
| Fault node | NPU 0 (GSP crash) |
| Nodes affected | 25% (1 of 4) |
| Recovery window | 0.3 hrs |

### Spec-calculated vs C++ injection — T0 comparison

| | Spec-calculated | C++ Injection (actual) |
|--|---|---|
| T0 (fault fires) | 28,730,170 cycles | 75,000,000 cycles |
| t_stall_detected | 76,031,879 cycles | 76,031,879 cycles |
| NCCL Stall Duration | 47,301,708 cycles | 1,031,879 cycles |

**Why different?**
Spec-calculated T0 assumes fault fires at 30% of JCT (step 3 of 10).
C++ injection fires T0 at the actual simulation time when AllReduce begins (~75ms),
which is the earliest defensible point the fault begins to cost compute time.

---

## C++ Fault Injection Hook (Stretch Goal)

Patches ASTRA-sim's congestion-aware backend to inject a fault mid-simulation.
When triggered, `Device::send()` drops all chunks — causing ring AllReduce to stall.

### Full end-to-end (apply → rebuild → run → revert)

```bash
conda run -n p903 python fault_injection_hook/run_c_injection.py
```

This automatically:
1. Applies the C++ patch and rebuilds ASTRA-sim
2. Runs baseline simulation
3. Runs fault injection simulation (NPU 0 disabled at T=75,000,000 ns)
4. Runs N-1 simulation (3 NPU ring reformed)
5. Reports JCT Recovery Penalty
6. Reverts ASTRA-sim to original and rebuilds

### Manual patch control

```bash
bash fault_injection_hook/apply_patch.sh   # apply + rebuild
bash fault_injection_hook/revert.sh        # restore + rebuild
```

---

## RunPod (A100 GPU Traces)

```bash
bash runpod/setup.sh                              # run once on pod startup
python runpod/kyle_profiler/run_profiler.py       # export GPT-2 traces
bash runpod/roshan_converter/convert.sh           # convert → Chakra .et
```

Output: `runpod/roshan_converter/workloads/gpt2.*.et` → copy to `pod_a_pipeline/workloads/`
