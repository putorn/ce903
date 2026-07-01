# CE903 Group Project — Network for LLM Training

Final submission — TC-01–TC-04, FR-G1 topology comparison, and Lane B fault-prediction LSTM.

---

## Repository Structure

```
ce903/
├── astra-sim/ ASTRA-sim 2.0 simulator (built, ready to run)
├── pod_a_pipeline/ TC-01 + TC-03 — trace-to-sim pipeline, fault injection & burst extraction
│ ├── workloads/ Chakra .et trace files (4 NPU)
│ ├── workloads_failure/ Chakra .et trace files (3 NPU — fault scenario)
│ ├── configs/
│ │ ├── network/ Ring topology configs (4 NPU, 3 NPU)
│ │ └── system/ ASTRA-sim system configs
│ ├── results/ Baseline simulation logs + observability contract
│ ├── results_with_failure/ Failure simulation logs
│ ├── results_fault_run/ C++ injection run logs + fault_events.csv
│ ├── burst_extractor.py Parses .et files into inter-burst interval series (feeds tc04/)
│ ├── make_synthetic_workloads.py Generates synthetic .et traces when real Chakra traces aren't available
│ ├── lane_b_lstm.py Lane B — fault-prediction LSTM + LOSO-CV harness
│ ├── run_pipeline.sh TC-01 — generate trace → simulate → validate
│ ├── run_fault_injection.py TC-03 — Python fault injection (critical bar)
│ ├── recovery_metrics.py Measurement harness — JCT penalty + NCCL stall
│ ├── sprint4_results.py Final results report
│ ├── fault_events.csv Fault timestamp T0 (spec-calculated)
│ └── stall_events.csv Stall detection time (spec-calculated)
├── pod_b_traffic/ TC-02 — ring all-reduce baseline sweep
│ ├── workloads/ Sweep trace files
│ ├── results/ 9 simulation logs (3 sizes × 3 NPU counts)
│ ├── generate_workloads.sh Generate sweep traces
│ ├── run_baseline.sh Run 9 simulations + analyse
│ └── analyze_results.py Analyse and display sweep results
├── tc04/ TC-04 — predictive (burst-timing LSTM) routing vs ECMP baseline
│ ├── train_lstm.py Trains the burst-timing LSTM (seeded, normalised, horizon-3)
│ ├── generate_routing.py Generates the predictor-informed routing config
│ ├── run_tc04.sh Runs ECMP vs predictor under the TC-03 failure pattern
│ ├── system_ecmp.json / system_predictor.json Routing configs compared by run_tc04.sh
│ └── tc04_results/ Simulation logs from run_tc04.sh
├── frg1/ FR-G1 — topology path-diversity & fault-redundancy comparison
│ ├── run_frg1.sh Two-tier spine-leaf attempt (needs a multi-dimension backend — see note below)
│ ├── run_frg1_fallback.sh Single-tier switch-fabric version — the one actually used for the reported results
│ ├── Ring_16npus.yml / switch_16.yml / switch_16_fail.yml Network configs used by the fallback script
│ ├── spine_leaf_16.yml / spine_leaf_16_spinefail.yml Two-tier configs used by run_frg1.sh
│ ├── system_ring.json / system_spineleaf.json System configs
│ └── FR-G1_results.md Write-up: what FR-G1 closes, method, path-complexity analysis
├── lane_b_data/ Lane B — seeded fault-injection sweep data (fault-prediction LSTM)
│ ├── seed_42/ … seed_51/ Per-seed LOSO-CV inputs/outputs (10 seeded runs, Aggressive MTBE sweep)
│ └── Lane_B_Fault_Injection_Summary.docx Summary write-up of the LOSO-CV results
├── fault_injection_hook/ C++ injection hook (stretch goal)
│ ├── Device.h / Device.cpp Modified — adds disable() flag to NPU
│ ├── Topology.h / Topology.cpp Modified — adds disable_npu() method
│ ├── main.cc Modified — injects fault at specified timestep
│ ├── run_c_injection.py End-to-end C++ injection runner
│ ├── apply_patch.sh Apply patch to ASTRA-sim + rebuild
│ └── revert.sh Restore original ASTRA-sim + rebuild
├── runpod/ GPU profiling scripts (RunPod A100)
│ ├── kyle_profiler/ GPT-2 profiler — exports execution trace
│ └── roshan_converter/ Converts profiler traces → Chakra .et format
├── archive/ Earlier, superseded versions of scripts — kept for provenance, not depended on by anything live
├── docs/ Sprint deliverables and report artefacts
├── SETUP.md Build guide (macOS / Linux / Windows WSL2)
├── README_config.md Experiment-configuration guide
└── environment.yml Conda environment (Python 3.11)
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

> All simulations below (TC-01–TC-04, FR-G1) require the ASTRA-sim binary to be built first. If you don't have real Chakra traces, run `python pod_a_pipeline/make_synthetic_workloads.py` once beforehand to generate synthetic ones.

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
conda run -n p903 python pod_a_pipeline/recovery_metrics.py
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

## TC-04 — Predictive Routing vs ECMP Baseline

Compares a burst-timing LSTM predictor against an ECMP baseline, under the same TC-03
failure pattern (seed 42), so the only thing that differs between the two runs is the
routing config.

> One-time fix if trace generation errors on the Chakra Python binding:
> ```bash
> cd astra-sim
> protoc --proto_path=extern/graph_frontend/chakra/schema/protobuf \
>   --python_out=extern/graph_frontend/chakra/schema/protobuf \
>   extern/graph_frontend/chakra/schema/protobuf/et_def.proto
> ```

```bash
python tc04/train_lstm.py        # trains the burst-timing LSTM
python tc04/generate_routing.py  # generates the predictor-informed routing config
bash tc04/run_tc04.sh            # run from the repo root — runs ECMP vs predictor and prints the verdict
```

Results in `tc04/tc04_results/`. Key metrics:

| Metric | Value |
|--------|-------|
| ECMP JCT | 94,012,616 cycles |
| Predictor JCT | 93,333,832 cycles |
| **Delta (predictor − ECMP)** | **−678,784 cycles (−0.72%)** |
| Train MAE | 2.4% |
| Failure MAE | 8.1% |
| Prediction horizon | 3 steps |

The predictor beats ECMP, but by a modest margin — the workload is compute-dominated
(80M of the ~94M cycles are fixed GPU compute), so the routing improvement can only act
on the smaller communication slice.

---

## FR-G1 — Topology Path-Diversity & Fault Redundancy

Tests whether a path-diverse topology (inspired by Alibaba's HPN dual-ToR design) beats
the ring on an identical 16-NPU GPT-2 workload, and how gracefully it degrades under a
switch failure.

`run_frg1.sh` attempts the full two-tier spine-leaf design, but ASTRA-sim's
congestion-aware analytical backend only supports single-dimension topologies — use
`run_frg1_fallback.sh` instead, which tells the same story with a single-tier switch
fabric (the version the results below come from):

```bash
bash frg1/run_frg1_fallback.sh   # run from the repo root
```

Results in `frg1/frg1_results/` (see also `frg1/FR-G1_results.md`):

| Run | JCT (cycles) | vs Ring |
|---|---|---|
| Ring (16 NPUs) | 99,937,920 | — baseline |
| Switch fabric, healthy | 118,927,680 | +19% (worse) |
| Switch fabric, degraded (1 switch link halved) | 156,428,160 | +31.5% vs healthy |

The path-diversity hypothesis was **not** supported at this scale: the ring remains
bandwidth-optimal, which is a confirmation of established practice rather than a failed
result. A full dual-ToR test (rack locality, packet-level ECMP) needs the ns-3 backend —
recorded as future work.

---

## Lane B — Fault-Prediction LSTM

A single-layer LSTM (sliding window, 4 input features) that predicts GPU-fault
probability ahead of time, trained on ten seeded simulations (seeds 42–51) under an
aggressive MTBE fault sweep.

```bash
conda run -n p903 python pod_a_pipeline/lane_b_lstm.py
```

Per-seed data and outputs are in `lane_b_data/` (see `Lane_B_Fault_Injection_Summary.docx`
for the write-up). Evaluated by leave-one-seed-out cross-validation:

| Metric | Value |
|---|---|
| Precision / Recall | 100% / 100% |
| Lead time | 0 steps (same-step detection) |

The detector is architecturally complete but only reaches same-step detection, because
ASTRA-sim fires faults as an instantaneous event with no pre-failure degradation ramp to
anticipate — an honest finding about the simulator, not the model. A synthetic
degradation-ramp extension that would demonstrate genuine ahead-of-time prediction is
scoped as future work.

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
bash fault_injection_hook/apply_patch.sh # apply + rebuild
bash fault_injection_hook/revert.sh # restore + rebuild
```

---

## RunPod (A100 GPU Traces)

```bash
bash runpod/setup.sh # run once on pod startup
python runpod/kyle_profiler/run_profiler.py # export GPT-2 traces
bash runpod/roshan_converter/convert.sh # convert → Chakra .et
```

Output: `runpod/roshan_converter/workloads/gpt2.*.et` → copy to `pod_a_pipeline/workloads/`

---

## Archive

`archive/` holds earlier, standalone versions of scripts that were superseded as the
pipeline was consolidated (an early burst extractor, an early LSTM trainer, an early
routing generator, and an early TC-04 wrapper). They're kept for provenance rather than
deleted — nothing in the live pipeline depends on them. See `archive/README.md` for the
full mapping from each old file to what replaced it.
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
conda run -n p903 python pod_a_pipeline/recovery_metrics.py
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
