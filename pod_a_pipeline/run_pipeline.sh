#!/bin/bash
# Pod A — TC-01 Pipeline Run
#
# Validates the full trace-to-sim pipeline:
#   1. Generate a synthetic GPT-2 training step trace (.et files).
#   2. Feed the trace into ASTRA-sim analytical backend.
#   3. Record JCT (Wall time), Comm time, and verify no errors.
#
# TC-01 pass criteria:
#   - Simulation completes for all NPUs without error.
#   - Wall time > Comm time (compute phases are modelled correctly).
#   - All NPUs report identical Wall time (symmetric ring collective).
#
# Synthetic trace models one GPT-2 124M training step (4 NPUs):
#   forward(25ms) → backward(50ms) → allreduce(512MB FP32) → optimizer(5ms)
set -e

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ASTRA_SIM_DIR="${SCRIPT_DIR}/../astra-sim"
POD_A_DIR="${SCRIPT_DIR}"

ASTRA_SIM_BIN="${ASTRA_SIM_DIR}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"
REMOTE_MEMORY="${ASTRA_SIM_DIR}/examples/remote_memory/analytical/no_memory_expansion.json"
SYSTEM_CFG="${POD_A_DIR}/configs/system/Ring_gpt2.json"
NETWORK_CFG="${POD_A_DIR}/configs/network/Ring_4npus.yml"

# ── Workload prefix ────────────────────────────────────────────────────────────
# Change this to the real trace prefix when Roshan delivers the converted .et files.
# Must match the filename before the ".N.et" suffix (e.g. "real_gpt2" for real_gpt2.0.et).
WORKLOAD_PREFIX="gpt2"  # swap to "real_gpt2" (or whatever Roshan names it)
WORKLOAD_DIR="${POD_A_DIR}/workloads"
RESULT_DIR="${POD_A_DIR}/results"

# ── macOS: fix nproc and protobuf ─────────────────────────────────────────────
export PATH="/opt/homebrew/opt/coreutils/libexec/gnubin:/opt/homebrew/bin:${PATH}"
export PROTOBUF_FROM_SOURCE=True

# ── Check binary ───────────────────────────────────────────────────────────────
if [ ! -f "${ASTRA_SIM_BIN}" ]; then
    echo "[Pod A] Binary not found. Building ASTRA-sim..."
    bash "${ASTRA_SIM_DIR}/build/astra_analytical/build.sh" -t all
else
    echo "[Pod A] ASTRA-sim binary found — skipping build."
fi

# ── Step 1: Generate synthetic GPT-2 trace ────────────────────────────────────
# Creates gpt2_step.0.et through gpt2_step.3.et in workloads/.
# Each file represents one NPU's view of a single GPT-2 training step.
echo ""
echo "[Pod A] ── Step 1: Generating synthetic GPT-2 trace ──────────────────────"
conda run -n p903 python "${POD_A_DIR}/generate_synthetic_trace.py" \
    --npus 4 \
    --output "${WORKLOAD_DIR}" \
    --astra-sim "${ASTRA_SIM_DIR}"

# ── Step 2: Run ASTRA-sim ─────────────────────────────────────────────────────
# ASTRA-sim writes log/log.log relative to CWD, so we run from the astra-sim dir.
echo ""
echo "[Pod A] ── Step 2: Running ASTRA-sim (TC-01 pipeline) ────────────────────"
echo "         workload : ${WORKLOAD_DIR}/${WORKLOAD_PREFIX}"
echo "         network  : ${NETWORK_CFG}"
echo "         system   : ${SYSTEM_CFG}"

mkdir -p "${RESULT_DIR}"
cd "${ASTRA_SIM_DIR}"

# Clear stale logs before running — ASTRA-sim appends rather than overwrites,
# so leftover entries from previous runs would corrupt the TC-01 NPU count check.
rm -f log/log.log log/err.log

"${ASTRA_SIM_BIN}" \
    --workload-configuration="${WORKLOAD_DIR}/${WORKLOAD_PREFIX}" \
    --system-configuration="${SYSTEM_CFG}" \
    --remote-memory-configuration="${REMOTE_MEMORY}" \
    --network-configuration="${NETWORK_CFG}"

# Save logs before anything overwrites them.
cp log/log.log "${RESULT_DIR}/log.log"
cp log/err.log "${RESULT_DIR}/err.log"
echo "[Pod A] Logs saved → ${RESULT_DIR}/"
cd "${SCRIPT_DIR}"

# ── Step 3: Validate TC-01 ────────────────────────────────────────────────────
echo ""
echo "[Pod A] ── Step 3: TC-01 Validation ──────────────────────────────────────"
conda run -n p903 python "${POD_A_DIR}/validate_tc01.py"

echo ""
echo "[Pod A] Done. Full log at pod_a_pipeline/results/log.log"
