# CE903 Group Project — Network for LLM Training

Sprint 2 · Role 2: ASTRA-sim & Simulation Setup

---

## What This Repository Contains

| Folder / File | Description |
|---------------|-------------|
| `astra-sim/` | ASTRA-sim 2.0 simulator — cloned, built, and ready to run |
| `SETUP.md` | How to install and build ASTRA-sim on your machine |
| `README_config.md` | How to change simulation parameters and run scenarios |
| `environment.yml` | Conda environment — recreate with `conda env create -f environment.yml` |
| `docs/` | Sprint 2 report and deliverables |

---

## Quick Start

**1. Set up conda environment**
```bash
conda env create -f astra-sim/environment.yml
conda activate p903
```

**2. Read the setup guide for your OS**

See [SETUP.md](SETUP.md)
- macOS → follow directly
- Linux → follow directly, use `apt-get` instead of `brew`
- Windows → must enable WSL2 first (instructions in SETUP.md Section 0)

**3. Run an example simulation**
```bash
cd astra-sim
./build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware \
    --workload-configuration=examples/workload/microbenchmarks/reduce_scatter/4npus_1MB/reduce_scatter \
    --system-configuration=examples/system/native_collectives/Ring_4chunks.json \
    --remote-memory-configuration=examples/remote_memory/analytical/no_memory_expansion.json \
    --network-configuration=examples/network/analytical/Ring_4npus.yml
```

**4. Check results**
```bash
cat astra-sim/log/log.log
```

---

## Repository Structure

```
ce903/
├── README.md                        ← You are here
├── .gitignore
├── SETUP.md                         ← Installation guide
├── README_config.md                 ← Configuration guide
├── environment.yml                  ← Conda environment
├── astra-sim/                       ← Simulator source code
│   ├── examples/                    ← Bundled workloads and configs
│   ├── build/                       ← Compiled binaries (not uploaded to GitHub)
│   └── log/                         ← Simulation output (not uploaded to GitHub)
└── docs/
    └── Role2_ASTRA_sim_Sprint2.docx ← Sprint 2 deliverable report
```

---

## Sprint 3 — Next Steps

When the team produces Chakra traces from real Hugging Face training:
1. Place `.et` files in `astra-sim/` (one per GPU, e.g. `trace.0.et`, `trace.1.et`)
2. Point `--workload-configuration` at the trace prefix
3. See [README_config.md](README_config.md) Section 6 for full instructions
