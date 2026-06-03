#!/bin/bash
# Restore original ASTRA-sim files and rebuild.
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ASTRA_SIM_DIR="${SCRIPT_DIR}/../astra-sim"

restore() {
    local dst="$1"
    if [ -f "${dst}.orig" ]; then
        cp "${dst}.orig" "${dst}"
        echo "[revert] $(basename ${dst}) restored"
    fi
}

restore "${ASTRA_SIM_DIR}/extern/network_backend/analytical/include/astra-network-analytical/congestion_aware/Device.h"
restore "${ASTRA_SIM_DIR}/extern/network_backend/analytical/congestion_aware/network/Device.cpp"
restore "${ASTRA_SIM_DIR}/extern/network_backend/analytical/include/astra-network-analytical/congestion_aware/Topology.h"
restore "${ASTRA_SIM_DIR}/extern/network_backend/analytical/congestion_aware/topology/Topology.cpp"
restore "${ASTRA_SIM_DIR}/astra-sim/network_frontend/analytical/congestion_aware/main.cc"

export PATH="/opt/homebrew/opt/coreutils/libexec/gnubin:/opt/homebrew/bin:${PATH}"
export PROTOBUF_FROM_SOURCE=True
echo "[revert] Rebuilding ASTRA-sim..."
bash "${ASTRA_SIM_DIR}/build/astra_analytical/build.sh" -t all

echo "[revert] Done. ASTRA-sim restored to original."
