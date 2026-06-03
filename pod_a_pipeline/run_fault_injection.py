"""
Pod A — GSP Fault Injection Harness
Sprint 4 · TC-03 · Single-fault demo

What this does
--------------
Simulates a GSP error (XID 119) — the highest-priority failure type in the
failure taxonomy (100% job-fail rate, 99% cascade probability) — by running
ASTRA-sim twice and taking the JCT difference:

  Run 1 (baseline) : N   NPUs, full ring  →  JCT_baseline
  Run 2 (failure)  : N-1 NPUs, ring minus the failed node  →  JCT_failure

  JCT Recovery Penalty = JCT_failure − JCT_baseline

Why two runs instead of mid-run C++ injection
----------------------------------------------
True mid-run injection requires patching Device.h in the congestion-aware
backend so one node stops responding after a fixed timestep.  For a permanent
node-removal (the 99% cascade branch — the demo path), the two-run approach
is mathematically equivalent: the cluster simply continues without that node.
No C++ changes are needed, making this the right critical-bar implementation.
The C++ hook remains the Sprint 4 stretch goal.

Observability contract fields written to results/observability_contract.json
(Pod B / klye_code.py reads these alongside the results directories):

  fault_node_id              — which NPU failed (NPU 0, fixed for demo)
  fault_timestamp_T0         — conceptual injection point (30% of baseline JCT)
  recovery_window_hrs        — 0.3 hrs (UIUC/IBM empirical average, fixed demo)
  nodes_affected_pct         — 1/N_baseline × 100
  reboot_events_per_epoch    — 1
  jct_baseline_cycles        — raw JCT from baseline run
  jct_with_failure_cycles    — raw JCT from failure run
  jct_recovery_penalty_cycles — the metric Pod B verifies (FR-G2, TC-03)

Output layout expected by klye_code.py
---------------------------------------
  pod_a_pipeline/results/log.log               — baseline simulation log
  pod_a_pipeline/results_with_failure/log.log  — failure simulation log
  pod_a_pipeline/results/observability_contract.json
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
ASTRA_SIM_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "astra-sim"))
CONFIGS_DIR   = os.path.join(SCRIPT_DIR, "configs")
WORKLOADS_DIR = os.path.join(SCRIPT_DIR, "workloads")

# Sprint 4 targets the congestion-aware binary — it models link states and
# is required for the routing-benefit experiments in Sprint 5.
ASTRA_SIM_BIN = os.path.join(
    ASTRA_SIM_DIR,
    "build", "astra_analytical", "build", "bin",
    "AstraSim_Analytical_Congestion_Aware",
)
REMOTE_MEMORY_CFG = os.path.join(
    ASTRA_SIM_DIR,
    "examples", "remote_memory", "analytical", "no_memory_expansion.json",
)

RESULTS_BASELINE  = os.path.join(SCRIPT_DIR, "results")
RESULTS_FAILURE   = os.path.join(SCRIPT_DIR, "results_with_failure")
OBSERVABILITY_OUT = os.path.join(RESULTS_BASELINE, "observability_contract.json")

# ── Fault parameters (GSP error XID 119 — Thursday demo values) ───────────────

BASELINE_NPUS       = 4
FAULT_NODE_ID       = 0                      # NPU 0 fails — fixed for demo
FAILURE_NPUS        = BASELINE_NPUS - 1      # 3 NPUs remain after fault
RECOVERY_WINDOW_HRS = 0.3                    # UIUC/IBM empirical average


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_astra_sim(
    workload_prefix: str,
    network_cfg: str,
    system_cfg: str,
    results_dir: str,
) -> str:
    """
    Run ASTRA-sim with the given config files, save logs to results_dir.
    Returns the path to the saved log.log.

    ASTRA-sim appends to existing log files rather than overwriting them, so
    stale logs are cleared before each run to keep results clean.
    ASTRA-sim also writes its log dir relative to CWD, so we run it from the
    astra-sim root directory.
    """
    os.makedirs(results_dir, exist_ok=True)

    astra_log_dir = os.path.join(ASTRA_SIM_DIR, "log")
    os.makedirs(astra_log_dir, exist_ok=True)

    for fname in ("log.log", "err.log"):
        stale = os.path.join(astra_log_dir, fname)
        if os.path.exists(stale):
            os.remove(stale)

    cmd = [
        ASTRA_SIM_BIN,
        f"--workload-configuration={workload_prefix}",
        f"--system-configuration={system_cfg}",
        f"--remote-memory-configuration={REMOTE_MEMORY_CFG}",
        f"--network-configuration={network_cfg}",
    ]

    result = subprocess.run(cmd, cwd=ASTRA_SIM_DIR, capture_output=True, text=True)

    if result.returncode != 0:
        print(result.stderr[:2000], file=sys.stderr)
        raise RuntimeError(f"ASTRA-sim exited with code {result.returncode}")

    shutil.copy(os.path.join(astra_log_dir, "log.log"), os.path.join(results_dir, "log.log"))
    shutil.copy(os.path.join(astra_log_dir, "err.log"), os.path.join(results_dir, "err.log"))

    return os.path.join(results_dir, "log.log")


def parse_jct(log_path: str) -> float:
    """
    Extract Wall time (JCT) from an ASTRA-sim log file.
    Wall time is the end-to-end job completion time per NPU.
    Returns the average across all NPUs — they are identical for a symmetric ring.
    """
    wall_times: list[int] = []
    with open(log_path) as fh:
        for line in fh:
            m = re.search(r"Wall time:\s*(\d+)", line)
            if m:
                wall_times.append(int(m.group(1)))

    if not wall_times:
        raise ValueError(f"No 'Wall time' lines found in {log_path}")

    return sum(wall_times) / len(wall_times)


def generate_traces(npus: int, output_dir: str) -> str:
    """
    Generate synthetic GPT-2 124M traces for the given NPU count.
    Calls generate_synthetic_trace.py inside the p903 conda environment.
    Returns the workload prefix string ASTRA-sim expects (path without .N.et).
    """
    os.makedirs(output_dir, exist_ok=True)

    gen_script = os.path.join(SCRIPT_DIR, "generate_synthetic_trace.py")
    cmd = [
        "conda", "run", "--no-capture-output", "-n", "p903",
        "python", gen_script,
        "--npus", str(npus),
        "--output", output_dir,
        "--astra-sim", ASTRA_SIM_DIR,
    ]
    subprocess.run(cmd, check=True)

    # generate_synthetic_trace.py always writes gpt2_step.N.et.
    return os.path.join(output_dir, "gpt2_step")


def make_network_config(npus: int) -> str:
    """
    Write a Ring network config for the given NPU count.
    Bandwidth and latency are kept identical to the baseline (50 GB/s, 500 ns)
    so the only variable between the two runs is the number of nodes.
    Returns the path to the written YAML file.
    """
    path = os.path.join(CONFIGS_DIR, "network", f"Ring_{npus}npus.yml")
    with open(path, "w") as fh:
        fh.write(f"topology: [Ring]\n")
        fh.write(f"npus_count: [{npus}]\n")
        fh.write(f"bandwidth: [50.0]  # GB/s\n")
        fh.write(f"latency: [500.0]   # ns\n")
    return path


def make_system_config(npus: int) -> str:
    """
    Copy the baseline system config and set preferred-dataset-splits to match
    the new NPU count.  All other parameters (scheduling policy, ring impls,
    collective optimisation) remain unchanged so the comparison is apples-to-apples.
    Returns the path to the new JSON file.
    """
    template = os.path.join(CONFIGS_DIR, "system", "Ring_gpt2.json")
    with open(template) as fh:
        cfg = json.load(fh)

    cfg["preferred-dataset-splits"] = npus

    path = os.path.join(CONFIGS_DIR, "system", f"Ring_gpt2_{npus}npus.json")
    with open(path, "w") as fh:
        json.dump(cfg, fh, indent=4)
    return path


def write_observability_contract(
    jct_baseline: float,
    jct_failure: float,
    npus_baseline: int,
) -> dict:
    """
    Write the Sprint 4 observability contract fields to JSON.

    fault_timestamp_T0 is expressed in cycles.  In the single-fault demo recipe,
    the fault fires 'after step 3 of 10', i.e. 30% into the run.  We approximate
    this as 30% of the baseline JCT, which gives Pod B a concrete cycle timestamp
    to pair with the hard-stall detection timestamp for the NCCL Stall Duration
    metric (FR-N2).
    """
    fault_timestamp_T0 = jct_baseline * 0.30

    contract = {
        "fault_node_id":               FAULT_NODE_ID,
        "fault_timestamp_T0":          fault_timestamp_T0,
        "recovery_window_hrs":         RECOVERY_WINDOW_HRS,
        "nodes_affected_pct":          round(1 / npus_baseline * 100, 1),
        "reboot_events_per_epoch":     1,
        "jct_baseline_cycles":         jct_baseline,
        "jct_with_failure_cycles":     jct_failure,
        "jct_recovery_penalty_cycles": jct_failure - jct_baseline,
    }

    os.makedirs(os.path.dirname(OBSERVABILITY_OUT), exist_ok=True)
    with open(OBSERVABILITY_OUT, "w") as fh:
        json.dump(contract, fh, indent=4)

    return contract


def print_report(contract: dict) -> None:
    baseline = contract["jct_baseline_cycles"]
    failure  = contract["jct_with_failure_cycles"]
    penalty  = contract["jct_recovery_penalty_cycles"]

    print()
    print("=" * 70)
    print("  Pod A — JCT Recovery Penalty Report")
    print("  GSP Error (XID 119) · single-fault demo · N=4 → N-1=3 NPUs")
    print("=" * 70)
    print(f"  Baseline JCT    : {baseline:>16,.0f}  cycles")
    print(f"  Failure  JCT    : {failure:>16,.0f}  cycles")
    print(f"  Penalty         : {penalty:>16,.0f}  cycles  ({penalty / baseline * 100:.1f}% overhead)")
    print()
    print(f"  Fault node      : NPU {contract['fault_node_id']}")
    print(f"  Nodes affected  : {contract['nodes_affected_pct']}%")
    print(f"  Recovery window : {contract['recovery_window_hrs']} hrs  (UIUC/IBM average)")
    print(f"  T0 (fault time) : {contract['fault_timestamp_T0']:,.0f}  cycles")
    print("=" * 70)
    print(f"\n  Observability contract → {OBSERVABILITY_OUT}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 70)
    print("  Pod A — GSP Fault Injection Harness")
    print("  Sprint 4 · TC-03 · Single-fault demo")
    print("=" * 70)

    # ── Step 1: Baseline run (4 NPUs, existing gpt2_step traces) ──────────────
    # Re-runs the baseline even if results already exist, so the two runs use
    # the same binary and configs and the JCT comparison is clean.
    print(f"\n[1/4] Baseline run ({BASELINE_NPUS} NPUs)...")
    baseline_log = run_astra_sim(
        workload_prefix=os.path.join(WORKLOADS_DIR, "gpt2_step"),
        network_cfg=os.path.join(CONFIGS_DIR, "network", "Ring_4npus.yml"),
        system_cfg=os.path.join(CONFIGS_DIR, "system", "Ring_gpt2.json"),
        results_dir=RESULTS_BASELINE,
    )
    jct_baseline = parse_jct(baseline_log)
    print(f"      JCT = {jct_baseline:,.0f} cycles")

    # ── Step 2: Generate N-1 traces (3 NPUs) ──────────────────────────────────
    # NPU 0 is removed (GSP crash — permanently inoperable for this job).
    # The remaining 3 NPUs form a new ring and run the same workload.
    print(f"\n[2/4] Generating {FAILURE_NPUS}-NPU traces (NPU {FAULT_NODE_ID} removed)...")
    workloads_failure_dir = os.path.join(SCRIPT_DIR, "workloads_failure")
    failure_prefix = generate_traces(FAILURE_NPUS, workloads_failure_dir)

    # ── Step 3: Failure run (3 NPUs) ──────────────────────────────────────────
    print(f"\n[3/4] Failure run ({FAILURE_NPUS} NPUs)...")
    failure_log = run_astra_sim(
        workload_prefix=failure_prefix,
        network_cfg=make_network_config(FAILURE_NPUS),
        system_cfg=make_system_config(FAILURE_NPUS),
        results_dir=RESULTS_FAILURE,
    )
    jct_failure = parse_jct(failure_log)
    print(f"      JCT = {jct_failure:,.0f} cycles")

    # ── Step 4: Compute penalty + write observability contract ─────────────────
    print(f"\n[4/4] Computing JCT Recovery Penalty...")
    contract = write_observability_contract(jct_baseline, jct_failure, BASELINE_NPUS)

    print_report(contract)
    return 0


if __name__ == "__main__":
    sys.exit(main())
