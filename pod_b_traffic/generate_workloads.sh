#!/bin/bash
# Pod B — Workload Generator
#
# Creates Chakra .et trace files for all sweep combinations:
#   Sizes : 1 MB (microbenchmark), 256 MB (GPT-2 BF16), 512 MB (GPT-2 FP32)
#   NPUs  : 4, 16, 64
#   Total : 9 sets of .et files → workloads/all_reduce/{npus}npus_{size}MB/
#
# Run this once before run_baseline.sh.
# Already-generated sets are skipped automatically (safe to re-run).
set -e

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ASTRA_SIM_DIR="${SCRIPT_DIR}/../astra-sim"
WORKLOAD_DIR="${SCRIPT_DIR}/workloads"

# ── macOS: fix nproc and protobuf ─────────────────────────────────────────────
export PATH="/opt/homebrew/opt/coreutils/libexec/gnubin:/opt/homebrew/bin:${PATH}"
export PROTOBUF_FROM_SOURCE=True

# ── Sweep parameters ──────────────────────────────────────────────────────────
SIZES_MB=(1 256 512)
NPUS=(4 16 64)

# ── Generate ──────────────────────────────────────────────────────────────────
echo "[generate] Starting workload generation"
echo "[generate] Output → ${WORKLOAD_DIR}"
echo ""

for size_mb in "${SIZES_MB[@]}"; do
    for npus in "${NPUS[@]}"; do
        et_dir="${WORKLOAD_DIR}/all_reduce/${npus}npus_${size_mb}MB"

        if [ -d "${et_dir}" ]; then
            echo "[generate] SKIP  ${npus}npus_${size_mb}MB  (already exists)"
            continue
        fi

        echo "[generate] CREATE ${npus}npus_${size_mb}MB ..."
        # Generator imports chakra internals, so astra-sim must be on the Python path.
        conda run -n p903 python -c "
import sys
sys.path.insert(0, '${ASTRA_SIM_DIR}')
from examples.workload.microbenchmarks.generator_scripts.all_reduce import generate_all_reduce
generate_all_reduce(npus_count=${npus}, coll_size=${size_mb}, path='${WORKLOAD_DIR}')
"
    done
done

echo ""
echo "[generate] Done. All .et files are in workloads/all_reduce/"
