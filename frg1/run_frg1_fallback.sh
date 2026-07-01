#!/bin/bash
# FR-G1 — FALLBACK (single-tier switch fabric)
#
# Only reach for this if run_frg1.sh errors on the two-tier spine-leaf step. It tells the
# same FR-G1 story — a path-diverse fabric against the ring, plus a switch-failure case —
# but with a single-dimension switch topology, which astra-sim definitely supports (it's the
# same shape as their validated HGX-H100 example). There's no multi-dimension config
# anywhere here, so nothing new can trip it.
#
#   1. Ring (16 NPUs)           — baseline
#   2. Switch fabric, healthy   — path-diverse topology (every NPU one hop via the switch)
#   3. Switch fabric, degraded  — fault-redundancy test (switch bandwidth halved)
#
# Run from the repo root:  bash frg1/run_frg1_fallback.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
ASTRA="${REPO_ROOT}/astra-sim"
BIN="${ASTRA}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
REMOTE_MEM="${ASTRA}/examples/remote_memory/analytical/no_memory_expansion.json"
LOGDIR="${ASTRA}/log"
OUT="${SCRIPT_DIR}/frg1_results"
WL_DIR="${SCRIPT_DIR}/workloads_16"
WL_PREFIX="${WL_DIR}/gpt2_step"
NPUS=16
mkdir -p "${OUT}" "${LOGDIR}"

# Use the p903 conda env if it exists, else fall back to the current python3
if conda env list 2>/dev/null | grep -q '\bp903\b'; then
  PY="conda run --no-capture-output -n p903 python"
else
  PY="python3"
fi

echo "Fallback run — single-tier switch fabric. Building the 16-NPU workload first..."
${PY} "${REPO_ROOT}/pod_a_pipeline/generate_synthetic_trace.py" \
    --npus ${NPUS} --output "${WL_DIR}" --astra-sim "${ASTRA}"

run_sim () {
    local name="$1" net="$2" sys="$3"
    echo "  Running ${name} — on $(basename "${net}")..." >&2
    rm -f "${LOGDIR}/log.log" "${LOGDIR}/err.log"
    ( cd "${ASTRA}" && "${BIN}" \
        --workload-configuration="${WL_PREFIX}" \
        --system-configuration="${sys}" \
        --remote-memory-configuration="${REMOTE_MEM}" \
        --network-configuration="${net}" ) > "${OUT}/log_${name}.log" 2>&1
    grep -m1 -oP 'Wall time:\s*\K[0-9]+' "${OUT}/log_${name}.log"
}

# Ring and switch fabric are both single-dimension, so they share the 1-D system config
# (system_ring.json) — no per-dimension lists to get wrong.
echo "Ring first, for comparison."
JCT_RING=$(run_sim ring       "${SCRIPT_DIR}/Ring_16npus.yml"   "${SCRIPT_DIR}/system_ring.json")
echo "Then the healthy switch fabric."
JCT_SW=$(run_sim   switch     "${SCRIPT_DIR}/switch_16.yml"     "${SCRIPT_DIR}/system_ring.json")
echo "And the switch with its bandwidth halved — the failure case."
JCT_SWF=$(run_sim  switchfail "${SCRIPT_DIR}/switch_16_fail.yml" "${SCRIPT_DIR}/system_ring.json")

echo ""
echo "Here's where we landed —"
echo "  Ring (16)               — ${JCT_RING} cycles"
echo "  Switch fabric, healthy  — ${JCT_SW} cycles"
echo "  Switch fabric, degraded — ${JCT_SWF} cycles"
python3 - "${JCT_RING}" "${JCT_SW}" "${JCT_SWF}" <<'PY'
import sys
ring, sw, swf = map(float, sys.argv[1:4])
tb = (sw - ring) / ring * 100
fp = (swf - sw) / sw * 100
verdict = ("the switch fabric lands below the ring — path diversity is doing real work"
           if sw < ring else
           "no gain at this scale — which we just note honestly in the write-up")
print(f"  Switch vs ring             : {tb:+.2f}%  — {verdict}")
print(f"  Cost of the degraded switch : {fp:+.2f}%  — how far JCT rises under failure")
PY
echo ""
echo "Logs in ${OUT}/. Send me these three numbers and FR-G1 is done."
