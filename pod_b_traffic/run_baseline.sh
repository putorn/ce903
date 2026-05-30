#!/bin/bash
# Pod B — Simulation Runner + Analyser (TC-02)
#
# Runs ring all-reduce simulations for all sweep combinations and analyses results.
# Requires workloads/ to be populated first — run generate_workloads.sh before this.
#
# Sweep: 3 sizes × 3 NPU counts = 9 simulations
#   Sizes : 1 MB, 256 MB (GPT-2 BF16), 512 MB (GPT-2 FP32)
#   NPUs  : 4, 16, 64
set -e

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ASTRA_SIM_DIR="${SCRIPT_DIR}/../astra-sim"
POD_B_DIR="${SCRIPT_DIR}"

ASTRA_SIM_BIN="${ASTRA_SIM_DIR}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"
REMOTE_MEMORY="${ASTRA_SIM_DIR}/examples/remote_memory/analytical/no_memory_expansion.json"
SYSTEM_CFG="${POD_B_DIR}/configs/system/Ring_allreduce.json"
WORKLOAD_DIR="${POD_B_DIR}/workloads"

# ── macOS: fix nproc and protobuf ─────────────────────────────────────────────
export PATH="/opt/homebrew/opt/coreutils/libexec/gnubin:/opt/homebrew/bin:${PATH}"
export PROTOBUF_FROM_SOURCE=True

# ── Check binary ───────────────────────────────────────────────────────────────
if [ ! -f "${ASTRA_SIM_BIN}" ]; then
    echo "[run] Binary not found. Building ASTRA-sim first..."
    bash "${ASTRA_SIM_DIR}/build/astra_analytical/build.sh" -t all
else
    echo "[run] ASTRA-sim binary found — skipping build."
fi

# ── Check workloads exist ──────────────────────────────────────────────────────
# Abort early if generate_workloads.sh has not been run yet.
if [ ! -d "${WORKLOAD_DIR}/all_reduce" ]; then
    echo "[run] ERROR: workloads/ is empty. Run generate_workloads.sh first."
    exit 1
fi

# ── Helper: return network config path for a given NPU count ──────────────────
network_cfg_for() {
    echo "${POD_B_DIR}/configs/network/Ring_${1}npus.yml"
}

# ── Helper: run one simulation and save logs ──────────────────────────────────
run_simulation() {
    local size_label="${1}"
    local npus="${2}"
    local result_dir="${POD_B_DIR}/results/${size_label}/${npus}npus"

    mkdir -p "${result_dir}"

    echo ""
    echo "[run] ── ${size_label} | ${npus} NPUs ────────────────────────────────"

    # ASTRA-sim writes log/log.log relative to CWD — run from its directory,
    # then copy logs out before the next run overwrites them.
    cd "${ASTRA_SIM_DIR}"

    "${ASTRA_SIM_BIN}" \
        --workload-configuration="${WORKLOAD_DIR}/all_reduce/${npus}npus_${size_label}/all_reduce" \
        --system-configuration="${SYSTEM_CFG}" \
        --remote-memory-configuration="${REMOTE_MEMORY}" \
        --network-configuration="$(network_cfg_for "${npus}")"

    cp log/log.log "${result_dir}/log.log"
    cp log/err.log "${result_dir}/err.log"
    echo "[run] Saved → ${result_dir}/"

    cd "${SCRIPT_DIR}"
}

# ── Run sweep: 3 sizes × 3 NPU counts = 9 simulations ────────────────────────
SIZES_MB=(1 256 512)
NPUS=(4 16 64)

echo ""
echo "[run] ── Running 9 simulations ───────────────────────────────────────────"

for size_mb in "${SIZES_MB[@]}"; do
    for npus in "${NPUS[@]}"; do
        run_simulation "${size_mb}MB" "${npus}"
    done
done

# ── Analyse ───────────────────────────────────────────────────────────────────
echo ""
echo "[run] ── Analysing results ───────────────────────────────────────────────"
conda run -n p903 python "${POD_B_DIR}/analyze_results.py"

echo ""
echo "[run] Done. Logs in results/{size}/{npus}/"
