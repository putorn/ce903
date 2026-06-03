"""
Sprint 4 — C++ Fault Injection End-to-End Runner

What this does
--------------
1. Applies the patch (copies modified C++ files into ASTRA-sim + rebuilds).
2. Runs baseline simulation (4 NPU, no fault) — gets JCT_baseline.
3. Runs fault simulation (4 NPU, NPU 0 disabled at T0) — fault_events.csv written.
4. Runs N-1 simulation (3 NPU) — gets JCT_failure (ring reformed after fault).
5. Prints JCT Recovery Penalty report.
6. Reverts ASTRA-sim to original and rebuilds.

Why two simulation runs for JCT (step 3 + 4)
----------------------------------------------
The C++ injection (step 3) proves the fault fires at the right time and logs T0.
But ASTRA-sim's ring AllReduce stalls and never completes once a member is
disabled — there is no ring reformation logic yet. So JCT_failure comes from
the 3-NPU run (step 4), which is equivalent for a permanent node-removal.

Usage:
    conda run -n p903 python fault_injection_hook/run_c_injection.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
CE903_DIR     = os.path.dirname(SCRIPT_DIR)
ASTRA_SIM_DIR = os.path.join(CE903_DIR, "astra-sim")
POD_A_DIR     = os.path.join(CE903_DIR, "pod_a_pipeline")

ASTRA_SIM_BIN = os.path.join(
    ASTRA_SIM_DIR, "build", "astra_analytical", "build", "bin",
    "AstraSim_Analytical_Congestion_Aware",
)
REMOTE_MEMORY = os.path.join(
    ASTRA_SIM_DIR, "examples", "remote_memory", "analytical", "no_memory_expansion.json",
)
BUILD_SCRIPT = os.path.join(ASTRA_SIM_DIR, "build", "astra_analytical", "build.sh")

ASTRA_LOG_DIR = os.path.join(ASTRA_SIM_DIR, "log")

# Fault parameters
FAULT_NODE_ID = 0
FAULT_TIME    = 28_730_170   # ns — 30% of baseline JCT (after step 3 of 10)


# ── Helpers ───────────────────────────────────────────────────────────────────

def rebuild():
    """Rebuild ASTRA-sim congestion-aware binary."""
    env = {**os.environ, "PROTOBUF_FROM_SOURCE": "True",
           "PATH": f"/opt/homebrew/opt/coreutils/libexec/gnubin:/opt/homebrew/bin:{os.environ.get('PATH', '')}"}

    # CMakeCache.txt (and nested _deps caches) store the original source path.
    # If the repo was moved, every cache mismatches. Wipe the entire build dir
    # so CMake starts fresh — rebuild takes longer but succeeds cleanly.
    build_dir = os.path.join(ASTRA_SIM_DIR, "build", "astra_analytical", "build")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
        print("[build] Cleared stale build directory (full clean)")

    print("[build] Rebuilding ASTRA-sim (takes a few minutes)...")
    subprocess.run(["bash", BUILD_SCRIPT, "-t", "all"], check=True, env=env)


def apply_patch():
    """Copy modified C++ files into ASTRA-sim source."""
    targets = {
        "Device.h": os.path.join(
            ASTRA_SIM_DIR, "extern", "network_backend", "analytical",
            "include", "astra-network-analytical", "congestion_aware", "Device.h"),
        "Device.cpp": os.path.join(
            ASTRA_SIM_DIR, "extern", "network_backend", "analytical",
            "congestion_aware", "network", "Device.cpp"),
        "Topology.h": os.path.join(
            ASTRA_SIM_DIR, "extern", "network_backend", "analytical",
            "include", "astra-network-analytical", "congestion_aware", "Topology.h"),
        "Topology.cpp": os.path.join(
            ASTRA_SIM_DIR, "extern", "network_backend", "analytical",
            "congestion_aware", "topology", "Topology.cpp"),
        "main.cc": os.path.join(
            ASTRA_SIM_DIR, "astra-sim", "network_frontend", "analytical",
            "congestion_aware", "main.cc"),
    }

    for fname, dst in targets.items():
        src = os.path.join(SCRIPT_DIR, fname)
        # backup original if not already done
        backup = dst + ".orig"
        if not os.path.exists(backup):
            shutil.copy(dst, backup)
        shutil.copy(src, dst)
        print(f"[patch] {fname} → applied")


def revert_patch():
    """Restore original C++ files in ASTRA-sim source."""
    targets = [
        os.path.join(ASTRA_SIM_DIR, "extern", "network_backend", "analytical",
                     "include", "astra-network-analytical", "congestion_aware", "Device.h"),
        os.path.join(ASTRA_SIM_DIR, "extern", "network_backend", "analytical",
                     "congestion_aware", "network", "Device.cpp"),
        os.path.join(ASTRA_SIM_DIR, "extern", "network_backend", "analytical",
                     "include", "astra-network-analytical", "congestion_aware", "Topology.h"),
        os.path.join(ASTRA_SIM_DIR, "extern", "network_backend", "analytical",
                     "congestion_aware", "topology", "Topology.cpp"),
        os.path.join(ASTRA_SIM_DIR, "astra-sim", "network_frontend", "analytical",
                     "congestion_aware", "main.cc"),
    ]
    for dst in targets:
        backup = dst + ".orig"
        if os.path.exists(backup):
            shutil.copy(backup, dst)
            print(f"[revert] {os.path.basename(dst)} → restored")


def clear_logs():
    """Remove stale ASTRA-sim logs before a run."""
    os.makedirs(ASTRA_LOG_DIR, exist_ok=True)
    for f in ("log.log", "err.log", "fault_events.csv"):
        p = os.path.join(ASTRA_LOG_DIR, f)
        if os.path.exists(p):
            os.remove(p)


def run_astra_sim(workload_prefix, network_cfg, system_cfg, results_dir, env_extra=None):
    """Run ASTRA-sim, save logs to results_dir. Returns log.log path."""
    os.makedirs(results_dir, exist_ok=True)
    clear_logs()

    env = {**os.environ, "PROTOBUF_FROM_SOURCE": "True",
           "PATH": f"/opt/homebrew/opt/coreutils/libexec/gnubin:/opt/homebrew/bin:{os.environ.get('PATH', '')}"}
    if env_extra:
        env.update(env_extra)

    cmd = [
        ASTRA_SIM_BIN,
        f"--workload-configuration={workload_prefix}",
        f"--system-configuration={system_cfg}",
        f"--remote-memory-configuration={REMOTE_MEMORY}",
        f"--network-configuration={network_cfg}",
    ]

    result = subprocess.run(cmd, cwd=ASTRA_SIM_DIR, capture_output=False, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"ASTRA-sim exited with code {result.returncode}")

    log_src = os.path.join(ASTRA_LOG_DIR, "log.log")
    shutil.copy(log_src, os.path.join(results_dir, "log.log"))

    # Copy fault_events.csv if the injection run wrote one
    fault_src = os.path.join(ASTRA_LOG_DIR, "fault_events.csv")
    if os.path.exists(fault_src):
        shutil.copy(fault_src, os.path.join(results_dir, "fault_events.csv"))

    return os.path.join(results_dir, "log.log")


def parse_jct(log_path):
    """Return average Wall time from log, or None if not found."""
    if not os.path.exists(log_path):
        return None
    wall_times = []
    with open(log_path) as f:
        for line in f:
            m = re.search(r"Wall time:\s*(\d+)", line)
            if m:
                wall_times.append(int(m.group(1)))
    return sum(wall_times) / len(wall_times) if wall_times else None


def generate_traces(npus, output_dir):
    """Generate synthetic GPT-2 traces for npus NPUs."""
    os.makedirs(output_dir, exist_ok=True)
    gen = os.path.join(POD_A_DIR, "generate_synthetic_trace.py")
    subprocess.run(
        ["conda", "run", "--no-capture-output", "-n", "p903",
         "python", gen, "--npus", str(npus), "--output", output_dir, "--astra-sim", ASTRA_SIM_DIR],
        check=True,
    )
    return os.path.join(output_dir, "gpt2_step")


def make_network_config(npus):
    path = os.path.join(POD_A_DIR, "configs", "network", f"Ring_{npus}npus.yml")
    with open(path, "w") as f:
        f.write(f"topology: [Ring]\nnpus_count: [{npus}]\nbandwidth: [50.0]\nlatency: [500.0]\n")
    return path


def make_system_config(npus):
    template = os.path.join(POD_A_DIR, "configs", "system", "Ring_gpt2.json")
    with open(template) as f:
        cfg = json.load(f)
    cfg["preferred-dataset-splits"] = npus
    path = os.path.join(POD_A_DIR, "configs", "system", f"Ring_gpt2_{npus}npus.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Sprint 4 — C++ Fault Injection End-to-End")
    print("=" * 70)

    # ── Step 1: Apply patch + rebuild ─────────────────────────────────────────
    print("\n[1/6] Applying C++ fault injection patch...")
    apply_patch()
    rebuild()

    baseline_4npu_network = os.path.join(POD_A_DIR, "configs", "network", "Ring_4npus.yml")
    baseline_4npu_system  = os.path.join(POD_A_DIR, "configs", "system", "Ring_gpt2.json")
    baseline_prefix       = os.path.join(POD_A_DIR, "workloads", "gpt2_step")
    results_baseline      = os.path.join(POD_A_DIR, "results")
    results_fault_run     = os.path.join(POD_A_DIR, "results_fault_run")
    results_failure       = os.path.join(POD_A_DIR, "results_with_failure")

    # ── Step 2: Baseline run (no fault) ───────────────────────────────────────
    print("\n[2/6] Baseline run (4 NPU, no fault)...")
    baseline_log = run_astra_sim(baseline_prefix, baseline_4npu_network, baseline_4npu_system, results_baseline)
    jct_baseline = parse_jct(baseline_log)
    print(f"      JCT_baseline = {jct_baseline:,.0f} ns")

    # ── Step 3: Fault injection run (NPU 0 disabled at T0) ────────────────────
    # This proves the fault fires at the right time — NPU 0's send() drops
    # chunks, AllReduce stalls, simulation ends with event queue drained.
    # Wall time is NOT logged (AllReduce never completes) — that is expected.
    print(f"\n[3/6] Fault injection run (NPU {FAULT_NODE_ID} disabled at T={FAULT_TIME} ns)...")
    fault_log = run_astra_sim(
        baseline_prefix, baseline_4npu_network, baseline_4npu_system,
        results_fault_run,
        env_extra={
            "ASTRA_FAULT_NODE_ID": str(FAULT_NODE_ID),
            "ASTRA_FAULT_TIME":    str(FAULT_TIME),
        },
    )
    fault_events_csv = os.path.join(results_fault_run, "fault_events.csv")
    print(f"      fault_events.csv → {fault_events_csv}")

    # ── Step 4: N-1 run (3 NPU — ring reformed after fault) ───────────────────
    print("\n[4/6] N-1 run (3 NPU — ring reformed after removing NPU 0)...")
    workloads_failure = os.path.join(POD_A_DIR, "workloads_failure")
    failure_prefix    = generate_traces(3, workloads_failure)
    failure_log = run_astra_sim(
        failure_prefix, make_network_config(3), make_system_config(3), results_failure
    )
    jct_failure = parse_jct(failure_log)
    print(f"      JCT_failure  = {jct_failure:,.0f} ns")

    # ── Step 5: Report ────────────────────────────────────────────────────────
    penalty = jct_failure - jct_baseline
    print("\n" + "=" * 70)
    print("  JCT Recovery Penalty Report (C++ injection)")
    print("=" * 70)
    print(f"  Baseline JCT    : {jct_baseline:>16,.0f}  ns")
    print(f"  Failure  JCT    : {jct_failure:>16,.0f}  ns")
    print(f"  Penalty         : {penalty:>16,.0f}  ns  ({penalty / jct_baseline * 100:.1f}% overhead)")
    print(f"\n  Fault node      : NPU {FAULT_NODE_ID}")
    print(f"  Fault fired at  : T={FAULT_TIME:,} ns")
    print(f"  fault_events.csv: {fault_events_csv}")
    print("=" * 70)

    # ── Step 6: Revert patch + rebuild ────────────────────────────────────────
    print("\n[6/6] Reverting C++ patch and rebuilding original ASTRA-sim...")
    revert_patch()
    rebuild()
    print("\n[done] ASTRA-sim restored to original.")


if __name__ == "__main__":
    sys.exit(main())
