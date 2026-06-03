#!/bin/bash
# Apply fault injection patch to ASTRA-sim and rebuild.
# Backup originals before copying.
# For full end-to-end: use run_c_injection.py instead.
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ASTRA_SIM_DIR="${SCRIPT_DIR}/../astra-sim"

copy_with_backup() {
    local src="${SCRIPT_DIR}/$1"
    local dst="$2"
    if [ ! -f "${dst}.orig" ]; then
        cp "${dst}" "${dst}.orig"
    fi
    cp "${src}" "${dst}"
    echo "[patch] $1 applied"
}

copy_with_backup "Device.h"   "${ASTRA_SIM_DIR}/extern/network_backend/analytical/include/astra-network-analytical/congestion_aware/Device.h"
copy_with_backup "Device.cpp" "${ASTRA_SIM_DIR}/extern/network_backend/analytical/congestion_aware/network/Device.cpp"
copy_with_backup "Topology.h"   "${ASTRA_SIM_DIR}/extern/network_backend/analytical/include/astra-network-analytical/congestion_aware/Topology.h"
copy_with_backup "Topology.cpp" "${ASTRA_SIM_DIR}/extern/network_backend/analytical/congestion_aware/topology/Topology.cpp"
copy_with_backup "main.cc"    "${ASTRA_SIM_DIR}/astra-sim/network_frontend/analytical/congestion_aware/main.cc"

export PATH="/opt/homebrew/opt/coreutils/libexec/gnubin:/opt/homebrew/bin:${PATH}"
export PROTOBUF_FROM_SOURCE=True
echo "[patch] Rebuilding ASTRA-sim..."
bash "${ASTRA_SIM_DIR}/build/astra_analytical/build.sh" -t all

echo "[patch] Done. To revert: bash fault_injection_hook/revert.sh"
