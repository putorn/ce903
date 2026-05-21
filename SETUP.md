# ASTRA-sim 2.0 — Setup Guide (CE903 Role 2)

Reproduces a working analytical-backend build of ASTRA-sim 2.0 on macOS (Apple Silicon).
Tested on 2026-05-21.

---

## 0. Which OS Are You On?

ASTRA-sim is **Linux-native**. Before doing anything, check which OS you have:

| OS | What to do |
|----|-----------|
| macOS | Follow this guide from Section 1 directly |
| Linux / Ubuntu | Follow this guide from Section 1 — skip the Homebrew parts, use `apt-get` instead |
| Windows | **Must set up WSL2 first** — see Section 0.1 below before continuing |

---

### 0.1 Windows Users — Set Up WSL2 First

WSL2 (Windows Subsystem for Linux) lets you run Ubuntu inside Windows without a separate VM. ASTRA-sim cannot be built natively on Windows — WSL2 is required.

**Step 1 — Enable WSL2**

Open PowerShell as Administrator and run:
```powershell
wsl --install
```
This installs WSL2 with Ubuntu automatically. Restart your computer when prompted.

**Step 2 — Open Ubuntu**

After restart, open the **Ubuntu** app from the Start menu. You will be inside a Linux terminal.

**Step 3 — Install conda inside WSL2**

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Follow the prompts, then restart the terminal.

**Step 4 — Create the p903 environment**

```bash
conda create -n p903 python=3.11
conda activate p903
```

**Step 5 — Continue from Section 2 of this guide**

From this point, all commands are the same as Linux/Ubuntu. Use `apt-get` instead of `brew`.

---

## 1. System & Environment

| Item | Value |
|------|-------|
| OS | macOS 26.3.1 (Darwin 25.3.0) |
| Architecture | ARM64 (Apple Silicon M1) |
| Shell | zsh |
| conda env | `p903` (Python 3.11.15) |
| Conda version | 26.3.2 |

> **Note for teammates on Linux/Ubuntu:** The build steps are identical. Replace the Homebrew (`brew`) commands with the Linux equivalents shown in the comments below each step.

---

## 2. Dependency Versions

| Tool | Version | How installed |
|------|---------|---------------|
| Apple Clang / `clang++` | 17.0.0 | Xcode Command Line Tools (pre-installed) |
| `git` | 2.50.1 | Pre-installed on macOS |
| `cmake` | 4.3.2 | `brew install cmake` |
| OpenMPI (`mpirun`) | 5.0.9 | `brew install openmpi` |
| Protocol Buffers (`protoc`) | 34.1 (libprotoc) | `brew install protobuf` |
| GNU coreutils (`nproc`) | 9.11 | `brew install coreutils` |
| Python | 3.11.15 | conda env `p903` |

**Linux/Ubuntu equivalents:**
```bash
sudo apt-get install -y cmake g++ libopenmpi-dev protobuf-compiler
```
(No coreutils install needed — `nproc` is built-in on Linux.)

---

## 3. Clone the Repository

Clone with `--recurse-submodules` — this is mandatory. Without it, the Chakra trace library, network backend, and helper libraries will all be missing and the build will fail.

```bash
git clone --recurse-submodules https://github.com/astra-sim/astra-sim.git
cd astra-sim
```

This clones 7 submodules:
- `extern/graph_frontend/chakra` — Chakra trace format (ET files)
- `extern/helper/fmt`, `extern/helper/spdlog` — logging/formatting libraries
- `extern/network_backend/analytical` — analytical network model
- `extern/network_backend/ns-3`, `extern/network_backend/csg-htsim` — other backends (not used here)
- `extern/remote_memory_backend/analytical` — remote memory model

---

## 4. Build — Analytical Backend

The analytical backend is the lightest build (no ns-3 or HTS-sim required).

### macOS — extra PATH needed

The build script uses `nproc` (Linux command for CPU count). On macOS, GNU coreutils provides it but installs it with a `g` prefix (`gnproc`). Export the gnubin path so the script finds `nproc`:

```bash
export PATH="/opt/homebrew/opt/coreutils/libexec/gnubin:/opt/homebrew/bin:$PATH"
```

### Set protobuf mode

ASTRA-sim's CMakeLists.txt has two protobuf linking modes. Homebrew's protobuf 34.1 (v5.x) requires the CONFIG mode so that abseil is correctly linked as a separate library. **Without this, the build will fail at the link step** with `ld: symbol(s) not found for architecture arm64`.

```bash
export PROTOBUF_FROM_SOURCE=True
```

### Run the build

```bash
bash build/astra_analytical/build.sh -t all
```

Options for `-t`:
- `all` — builds both congestion-unaware and congestion-aware executables
- `congestion_unaware` — lighter, simpler model
- `congestion_aware` — models network congestion

Build output (both executables):
```
build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware
build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware
```

Convenience symlinks are also created:
```
build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra   → Congestion_Unaware
build/astra_analytical/build/AstraCongestion/bin/AstraCongestion   → Congestion_Aware
```

---

## 5. Errors Encountered and Fixes

### Error 1 — `nproc: command not found`

**What happened:** The build script calls `nproc` to detect how many CPU threads to use for parallel compilation. macOS does not ship `nproc`.

**Fix:** Install GNU coreutils via Homebrew, then prepend the gnubin path to `$PATH`:
```bash
brew install coreutils
export PATH="/opt/homebrew/opt/coreutils/libexec/gnubin:$PATH"
```

### Error 2 — `ld: symbol(s) not found for architecture arm64` (abseil/protobuf)

**What happened:** Homebrew's protobuf 34.1 (v5.x) uses Google's abseil library internally and requires it to be linked explicitly. The default build path in CMakeLists.txt uses the old `find_package(Protobuf REQUIRED)` module which omits abseil from the link command.

**Fix:** Set the environment variable `PROTOBUF_FROM_SOURCE=True` before building. This switches CMake to `find_package(protobuf CONFIG REQUIRED)` mode, which correctly imports abseil as a transitive dependency:
```bash
export PROTOBUF_FROM_SOURCE=True
bash build/astra_analytical/build.sh -t all
```

---

## 6. Running an Example Simulation

```bash
ASTRA_SIM="build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"
EXAMPLE_DIR="examples"

"${ASTRA_SIM}" \
    --workload-configuration="${EXAMPLE_DIR}/workload/microbenchmarks/reduce_scatter/4npus_1MB/reduce_scatter" \
    --system-configuration="${EXAMPLE_DIR}/system/native_collectives/Ring_4chunks.json" \
    --remote-memory-configuration="${EXAMPLE_DIR}/remote_memory/analytical/no_memory_expansion.json" \
    --network-configuration="${EXAMPLE_DIR}/network/analytical/Ring_4npus.yml"
```

### What the flags mean

| Flag | Purpose |
|------|---------|
| `--workload-configuration` | Path prefix for Chakra `.et` trace files (without `.N.et` suffix) |
| `--system-configuration` | System config (collective algorithm, chunk sizes, scheduling policy) |
| `--remote-memory-configuration` | Remote memory model config |
| `--network-configuration` | Network topology and bandwidth/latency (YAML) |

---

## 7. Output Files

After a simulation run, two files are written:

| File | Contents |
|------|----------|
| `log/log.log` | Per-NPU simulation results: total wall-clock cycles, communication cycles, and topology info |
| `log/err.log` | Error/warning messages (empty if the run is clean) |

**Example log output (reduce_scatter, 4 NPUs, 1 MB):**
```
sys[0] finished, 22240 cycles, exposed communication 22240 cycles.
sys[0], Wall time: 22240
sys[0], Comm time: 22240
```

- **Wall time** — total simulated time (in cycles) from start to finish for that NPU
- **Comm time** — time spent in network communication operations
- When Comm time equals Wall time, 100% of time was in communication (no compute overlap — expected for a pure collective microbenchmark with no compute workload)

---

## 8. Using Custom Chakra Traces

Chakra traces are binary protobuf files with the `.et` extension (Execution Trace). One file is required **per NPU**.

### Naming convention

ASTRA-sim automatically loads files based on the NPU index. If you pass:
```
--workload-configuration=/path/to/my_workload
```
It will look for:
```
/path/to/my_workload.0.et   ← NPU 0
/path/to/my_workload.1.et   ← NPU 1
...
/path/to/my_workload.N-1.et ← NPU N-1
```

### Generating custom traces

Use the Chakra tooling (already cloned in `extern/graph_frontend/chakra`):

```bash
# Install Chakra Python tools in your conda environment
conda activate p903
pip install -e extern/graph_frontend/chakra

# Generate ET files using Chakra's trace generator or converter
# See: extern/graph_frontend/chakra/README.md
```

Chakra can generate traces from:
- PyTorch distributed training (via `torch.distributed` hooks)
- Manual specification via the protobuf schema in `extern/graph_frontend/chakra/schema/protobuf/et_def.proto`

---

## 9. Quick Rebuild Reference

```bash
# From inside the astra-sim directory:
export PATH="/opt/homebrew/opt/coreutils/libexec/gnubin:/opt/homebrew/bin:$PATH"
export PROTOBUF_FROM_SOURCE=True

# Clean build
bash build/astra_analytical/build.sh -l   # removes build/ directory
bash build/astra_analytical/build.sh -t all

# Debug build (slower binary, full symbols for gdb/lldb)
bash build/astra_analytical/build.sh -t all -d
```
