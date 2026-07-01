#!/bin/bash
# tc04/run_tc04.sh — TC-04: predictor-informed routing vs ECMP under the TC-03 failure pattern.
#
# Runs ASTRA-sim twice with an IDENTICAL workload, network and seed; the ONLY thing
# that differs between the two runs is the system/routing config (ECMP vs predictor).
# Prints the JCT delta and a pass/fail verdict. Mirrors the proven run_fault_injection.py.
#
# Prerequisite: the Chakra Python binding must exist (one-time fix):
#   cd ../astra-sim
#   protoc --proto_path=extern/graph_frontend/chakra/schema/protobuf \
#          --python_out=extern/graph_frontend/chakra/schema/protobuf \
#          extern/graph_frontend/chakra/schema/protobuf/et_def.proto
#
# Run from the repo root:  bash tc04/run_tc04.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
ASTRA="${REPO_ROOT}/astra-sim"

BIN="${ASTRA}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
REMOTE_MEM="${ASTRA}/examples/remote_memory/analytical/no_memory_expansion.json"
NET="${REPO_ROOT}/pod_a_pipeline/configs/network/Ring_3npus.yml"   # TC-03 failure topology (NPU 0 removed)
WL_DIR="${REPO_ROOT}/pod_a_pipeline/workloads_failure"
WL_PREFIX="${WL_DIR}/gpt2_step"
ECMP_CFG="${SCRIPT_DIR}/system_ecmp.json"
PRED_CFG="${SCRIPT_DIR}/system_predictor.json"
LOGDIR="${ASTRA}/log"
OUT="${SCRIPT_DIR}/tc04_results"
mkdir -p "${OUT}" "${LOGDIR}"

# Use the p903 conda env if it exists, else fall back to the current python3
if conda env list 2>/dev/null | grep -q '\bp903\b'; then
  PY="conda run --no-capture-output -n p903 python"
else
  PY="python3"
fi

echo "[TC-04] 1/3  Generating 3-NPU gpt2_step Chakra traces (TC-03 failure: NPU 0 removed)..."
${PY} "${REPO_ROOT}/pod_a_pipeline/generate_synthetic_trace.py" \
    --npus 3 --output "${WL_DIR}" --astra-sim "${ASTRA}"

run_sim () {
    local name="$1" cfg="$2"
    # Progress text goes to stderr so the command substitution that calls this
    # function captures ONLY the JCT number on stdout — nothing else.
    echo "[TC-04] Running ${name}  (system=$(basename "${cfg}"))..." >&2
    rm -f "${LOGDIR}/log.log" "${LOGDIR}/err.log"
    # Send the simulator's console output straight to the result log; if it reached
    # stdout it would pollute the captured value (that was the earlier parse failure).
    ( cd "${ASTRA}" && "${BIN}" \
        --workload-configuration="${WL_PREFIX}" \
        --system-configuration="${cfg}" \
        --remote-memory-configuration="${REMOTE_MEM}" \
        --network-configuration="${NET}" ) > "${OUT}/log_${name}.log" 2>&1
    # JCT = Wall time (cycles); identical across NPUs for a symmetric ring.
    # -m1 stops at the first match (clean exit, no SIGPIPE under pipefail).
    grep -m1 -oP 'Wall time:\s*\K[0-9]+' "${OUT}/log_${name}.log"
}

echo "[TC-04] 2/3  Two simulation runs (seed-matched, failure topology)..."
JCT_ECMP=$(run_sim ecmp "${ECMP_CFG}")
JCT_PRED=$(run_sim predictor "${PRED_CFG}")

echo ""
echo "[TC-04] 3/3  Result"
echo "======================================================"
echo "  ECMP      median JCT : ${JCT_ECMP} cycles"
echo "  Predictor median JCT : ${JCT_PRED} cycles"
python3 - "${JCT_ECMP}" "${JCT_PRED}" <<'PY'
import sys
e, p = float(sys.argv[1]), float(sys.argv[2])
d = p - e
print(f"  Delta (pred - ecmp)  : {d:,.0f} cycles ({d/e*100:+.2f}%)")
print("  VERDICT:", "PASS  — predictor achieves lower JCT than ECMP"
      if p < e else
      "NO IMPROVEMENT — predictor >= ECMP (still a valid, reportable result)")
PY
echo "======================================================"
echo "  Logs saved to ${OUT}/  — send both numbers to Brad."
