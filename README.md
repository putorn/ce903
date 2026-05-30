# CE903 Group Project — Network for LLM Training

Role 2: ASTRA-sim & Simulation Setup · Sprint 3

---

## Repository Structure

```
ce903/
├── astra-sim/           ASTRA-sim 2.0 simulator (built, ready to run)
├── pod_b_traffic/       TC-02 — ring all-reduce baseline sweep
├── pod_a_pipeline/      TC-01 — trace-to-sim pipeline
├── runpod/              GPU profiling scripts for RunPod A100
├── SETUP.md             Build guide (macOS / Linux / Windows WSL2)
└── environment.yml      Conda environment (Python 3.11)
```

---

## Setup

**Step 1 — Create conda environment (Python 3.11)**
```bash
conda env create -f environment.yml
conda activate p903
```

**Step 2 — Install pip packages**
```bash
pip install -r requirements.txt
```

For the profiler pipeline (trace conversion) also run:
```bash
pip install "git+https://github.com/facebookresearch/param.git#subdirectory=train/compute/python"
```

> TC-01 and TC-02 simulations do not require Step 2 — ASTRA-sim is a C++ binary.

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

---

## TC-02 — All-Reduce Baseline (Pod B)

```bash
cd pod_b_traffic
bash generate_workloads.sh   # generate .et files (run once)
bash run_baseline.sh         # run 9 simulations + analyse
```

Results in `pod_b_traffic/results/`. Re-analyse without re-running:
```bash
python analyze_results.py
```

---

## TC-01 — Trace-to-Sim Pipeline (Pod A)

```bash
cd pod_a_pipeline
bash run_pipeline.sh         # generate trace → simulate → validate
```

To use Roshan's real trace instead of the synthetic one:
1. Copy `gpt2.0.et … gpt2.3.et` into `pod_a_pipeline/workloads/`
2. Set `WORKLOAD_PREFIX="gpt2"` in `run_pipeline.sh` (line 31)
3. Re-run `bash run_pipeline.sh`

---

## RunPod (A100 GPU traces)

```bash
# On RunPod — run once after starting the pod
bash runpod/setup.sh

# Kyle: export GPT-2 profiler traces
python runpod/kyle_profiler/run_profiler.py

# Roshan: convert traces to Chakra .et format
bash runpod/roshan_converter/convert.sh
```

Output: `runpod/roshan_converter/workloads/gpt2.*.et` → copy to `pod_a_pipeline/workloads/`.
