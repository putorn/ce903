#!/bin/bash
# FR-G1 — Spine-Leaf Topology
#
# This is the whole of FR-G1 in a single run, and the idea behind it is simple. We take
# the exact same GPT-2 workload the ring runs used and put it on three different networks
# in turn — a plain ring, a spine-leaf, and a spine-leaf with one of its two spines knocked
# out. Because the workload never changes, any difference in Job Completion Time is the
# network alone, doing its job or failing to. That's precisely what FR-G1 asks us to show:
# that a path-diverse topology helps, and that it degrades gracefully when a switch dies.
#
# What it does, in order:
#   1. Builds the 16-NPU GPT-2 trace — the same training step we use everywhere else.
#   2. Runs the ring baseline, so we've got something honest to measure against.
#   3. Runs the healthy spine-leaf — the one we'd like to see beat the ring.
#   4. Runs the spine-leaf with a spine degraded — the fault-redundancy test.
# Then it hands back the two numbers that actually matter: how much the spine-leaf helps,
# and how much losing a spine costs us.
#
# One thing before you set it going — you'll need the Chakra Python binding in place, the
# same one-time fix we did for TC-04. If generate_synthetic_trace.py still complains, run:
#   cd ../astra-sim
#   protoc --proto_path=extern/graph_frontend/chakra/schema/protobuf \
#          --python_out=extern/graph_frontend/chakra/schema/protobuf \
#          extern/graph_frontend/chakra/schema/protobuf/et_def.proto
#
# Then, from the repo root:  bash frg1/run_frg1.sh
set -euo pipefail

# The script works out where everything lives from its own location, so there's nothing
# to edit by hand — just keep frg1/ sitting at the repo root, next to tc04/.
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

# The trace generator runs through the p903 conda env — that's the one carrying protobuf.
PY="conda run --no-capture-output -n p903 python"

echo "Right — workload first. Same 16-NPU GPT-2 step the rest of the project uses, so the"
echo "comparison stays fair..."
${PY} "${REPO_ROOT}/pod_a_pipeline/generate_synthetic_trace.py" \
    --npus ${NPUS} --output "${WL_DIR}" --astra-sim "${ASTRA}"

# Each run is the same workload on a different network. We push all the simulator's chatter
# into a log file and hand back only the one number we care about — the Wall time (JCT) —
# so nothing leaks into the comparison at the end. (That stray-output trap is exactly what
# bit the first TC-04 script, so we're keeping it clean here from the start.)
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

echo "Now the three runs — ring first, to give us our point of comparison."
JCT_RING=$(run_sim ring      "${SCRIPT_DIR}/Ring_16npus.yml"             "${SCRIPT_DIR}/system_ring.json")
echo "Then the healthy spine-leaf — this is the one we want coming in under the ring."
JCT_SL=$(run_sim   spineleaf "${SCRIPT_DIR}/spine_leaf_16.yml"          "${SCRIPT_DIR}/system_spineleaf.json")
echo "And lastly the spine-leaf with a spine down — to see how gracefully it carries on."
JCT_SLF=$(run_sim  spinefail "${SCRIPT_DIR}/spine_leaf_16_spinefail.yml" "${SCRIPT_DIR}/system_spineleaf.json")

echo ""
echo "Here's where we landed —"
echo "  Ring (16)                  — ${JCT_RING} cycles"
echo "  Spine-leaf, healthy        — ${JCT_SL} cycles"
echo "  Spine-leaf, one spine lost — ${JCT_SLF} cycles"
python3 - "${JCT_RING}" "${JCT_SL}" "${JCT_SLF}" <<'PY'
import sys
ring, sl, slf = map(float, sys.argv[1:4])
tb = (sl - ring) / ring * 100
fp = (slf - sl) / sl * 100
verdict = ("the spine-leaf lands below the ring — path diversity is doing real work"
           if sl < ring else
           "no gain at this scale — which we just note honestly in the write-up")
print(f"  Spine-leaf vs ring         : {tb:+.2f}%  — {verdict}")
print(f"  Cost of losing one spine   : {fp:+.2f}%  — how far JCT rises when a spine degrades")
PY
echo ""
echo "Logs are tucked in ${OUT}/ if you want to look closer. Send me those three numbers"
echo "and FR-G1 is done."
