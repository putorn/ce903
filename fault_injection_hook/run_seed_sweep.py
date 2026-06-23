"""
Sprint 5 — Seeded MTBE Fault-Injection Sweep
For Lane B (lane_b_lstm.py): produces fault_events.csv, stall_events.csv
(with stall_type), and step_metrics.csv per seed, seeds 42-51.

Why this isn't "live" in-process fault recovery
-------------------------------------------------
The real C++ hook (Device.h/.cpp, Topology.h/.cpp, main.cc) only supports a
single permanent disable() per process — there is no enable()/recovery, and
once a node is disabled the simulation stalls and never logs a Wall time.
generate_synthetic_trace.py also only ever encodes ONE training step per
.et trace (forward -> backward -> AllReduce -> optimizer), so "step N+1"
already meant "re-invoke the binary" before this script existed.

This script gets real recovery the only way this engine supports it: each
training step is its own ASTRA-sim subprocess call. A "hard" fault disables
a node for real (proving the C++ stall fires) and then permanently drops to
N-1 NPUs for every subsequent step in that seed, reusing the existing N->N-1
mechanism from run_c_injection.py. A "soft" fault also disables a node for
real for that one step, but the node count is unchanged for the next step
since each step is a fresh process — i.e. it "recovers" for free.

Because the engine never produces a finite Wall time once a node is
disabled (by design — see main.cc), a faulted step's step_time/
completion_ratio are the literal k*expected_step_time / 1/k contract values
from the spec, not a measured number. Healthy steps use a real measured
Wall time from ASTRA-sim.

Usage:
    conda run -n p903 python fault_injection_hook/run_seed_sweep.py
"""

from __future__ import annotations

import csv
import os
import random
import sys

from run_c_injection import (
    POD_A_DIR,
    apply_patch,
    revert_patch,
    rebuild,
    run_astra_sim,
    parse_jct,
    generate_traces,
    make_network_config,
    make_system_config,
)

RUNS_DIR = os.path.join(POD_A_DIR, "runs")

SEEDS = range(42, 52)
NUM_STEPS = 10
BASELINE_NPUS = 4

# ── "Aggressive" sweep profile ──────────────────────────────────────────────
# MTBE_NODE_HOURS is the sweep's descriptive label only. With only 10 steps
# per run, a real Poisson process at 25 node-hours MTBE (4 NPUs) would need
# ~35x more node-hours of exposure than a 10-step run can ever accumulate,
# so it would produce zero faults in ~77% of 10-seed sweeps (confirmed
# empirically). Per explicit decision: each seed gets a guaranteed 1-2
# faults instead of letting Poisson/MTBE math decide *whether* a fault
# happens — MTBE/step-count stay as labels, not the occurrence mechanism.
MTBE_NODE_HOURS = 25.0
MIN_FAULTS_PER_SEED = 1
MAX_FAULTS_PER_SEED = 2

# Share of faults that are permanent (hard) vs transient (soft).
HARD_FAULT_PROBABILITY = 0.3

SOFT_K = 1.5
HARD_K = 3.0

# Placeholder workload unit for node_throughput = WORK / step_time.
NOMINAL_WORK_UNITS_PER_STEP = 1.0e9

BASELINE_WORKLOAD_PREFIX = os.path.join(POD_A_DIR, "workloads", "gpt2_step")
BASELINE_NETWORK_CFG = os.path.join(POD_A_DIR, "configs", "network", "Ring_4npus.yml")
BASELINE_SYSTEM_CFG = os.path.join(POD_A_DIR, "configs", "system", "Ring_gpt2.json")

# npus -> (workload_prefix, network_cfg, system_cfg), populated lazily.
# Deterministic given npus, so safe to reuse across seeds/steps.
_npu_config_cache: dict[int, tuple[str, str, str]] = {
    BASELINE_NPUS: (BASELINE_WORKLOAD_PREFIX, BASELINE_NETWORK_CFG, BASELINE_SYSTEM_CFG),
}


def configs_for(npus: int) -> tuple[str, str, str]:
    if npus not in _npu_config_cache:
        workloads_dir = os.path.join(RUNS_DIR, f"_workloads_{npus}npu")
        workload_prefix = generate_traces(npus, workloads_dir)
        network_cfg = make_network_config(npus)
        system_cfg = make_system_config(npus)
        _npu_config_cache[npus] = (workload_prefix, network_cfg, system_cfg)
    return _npu_config_cache[npus]


def measure_expected_step_time() -> float:
    """One clean 4-NPU run -> deterministic baseline step time (no RNG in the analytical backend)."""
    results_dir = os.path.join(RUNS_DIR, "_baseline_probe")
    log_path = run_astra_sim(
        BASELINE_WORKLOAD_PREFIX, BASELINE_NETWORK_CFG, BASELINE_SYSTEM_CFG, results_dir
    )
    expected = parse_jct(log_path)
    if expected is None:
        raise RuntimeError("Baseline probe run produced no Wall time — cannot derive expected_step_time")
    return expected


def run_step(n_npus: int, fault_node_id: int | None, fault_time: int, results_dir: str) -> str:
    """Run one training step at n_npus NPUs, optionally injecting a real disable() fault."""
    workload_prefix, network_cfg, system_cfg = configs_for(n_npus)

    env_extra = None
    if fault_node_id is not None:
        env_extra = {
            "ASTRA_FAULT_NODE_ID": str(fault_node_id),
            "ASTRA_FAULT_TIME": str(fault_time),
        }

    return run_astra_sim(workload_prefix, network_cfg, system_cfg, results_dir, env_extra=env_extra)


def simulate_seed(seed: int, expected_step_time: float) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    active_nodes = list(range(BASELINE_NPUS))

    num_faults = rng.randint(MIN_FAULTS_PER_SEED, MAX_FAULTS_PER_SEED)
    fault_steps = set(rng.sample(range(NUM_STEPS), num_faults))

    fault_rows: list[dict] = []
    stall_rows: list[dict] = []
    step_rows: list[dict] = []

    t_cursor = 0.0
    results_dir = os.path.join(RUNS_DIR, f"_seed_{seed}_scratch")

    for step in range(NUM_STEPS):
        nodes_this_step = list(active_nodes)
        n = len(nodes_this_step)

        # n > 1 guard: a hard fault earlier in this seed may have already
        # dropped the ring to 1 NPU, in which case further faults are skipped.
        faulted = step in fault_steps and n > 1

        fault_node_id = None
        stall_type = None

        if faulted:
            stall_type = "hard" if rng.random() < HARD_FAULT_PROBABILITY else "soft"
            fault_node_id = rng.choice(nodes_this_step)
            k = HARD_K if stall_type == "hard" else SOFT_K
            fault_time = int(expected_step_time * 0.30)  # fires 30% into the step (existing convention)

            run_step(n, fault_node_id, fault_time, results_dir)

            step_time = k * expected_step_time
            completion_ratio = 1.0 / k

            t_fault = t_cursor + fault_time
            collective_id = f"gpt2_step{step}"
            fault_rows.append({"collective_instance_id": collective_id, "t_fault": t_fault})
            stall_rows.append({
                "collective_instance_id": collective_id,
                "t_stall_detected": t_fault + step_time * 0.10,
                "stall_type": stall_type,
            })
        else:
            log_path = run_step(n, None, 0, results_dir)
            measured = parse_jct(log_path)
            step_time = measured if measured is not None else expected_step_time
            completion_ratio = 1.0

        for node_id in nodes_this_step:
            is_faulted_node = node_id == fault_node_id
            node_throughput = 0.0 if is_faulted_node else NOMINAL_WORK_UNITS_PER_STEP / step_time
            step_rows.append({
                "step": step,
                "t_step_start": t_cursor,
                "step_time": step_time,
                "completion_ratio": completion_ratio,
                "node_throughput": node_throughput,
                "node_id": node_id,
            })

        t_cursor += step_time

        if stall_type == "hard":
            active_nodes.remove(fault_node_id)

    return fault_rows, stall_rows, step_rows


def write_seed_outputs(seed: int, fault_rows: list[dict], stall_rows: list[dict], step_rows: list[dict]) -> str:
    out_dir = os.path.join(RUNS_DIR, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "fault_events.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["collective_instance_id", "t_fault"])
        writer.writeheader()
        writer.writerows(fault_rows)

    with open(os.path.join(out_dir, "stall_events.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["collective_instance_id", "t_stall_detected", "stall_type"])
        writer.writeheader()
        writer.writerows(stall_rows)

    with open(os.path.join(out_dir, "step_metrics.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["step", "t_step_start", "step_time", "completion_ratio", "node_throughput", "node_id"]
        )
        writer.writeheader()
        writer.writerows(step_rows)

    return out_dir


def main() -> int:
    os.makedirs(RUNS_DIR, exist_ok=True)

    print("=" * 70)
    print("  Sprint 5 — Seeded MTBE Fault-Injection Sweep (Aggressive, 25 node-hrs)")
    print("=" * 70)

    print("\n[1/4] Applying C++ fault injection patch + rebuilding...")
    apply_patch()
    rebuild()

    print("\n[2/4] Measuring baseline expected_step_time (4 NPU, no fault)...")
    expected_step_time = measure_expected_step_time()
    print(f"      expected_step_time = {expected_step_time:,.0f} cycles")

    print(f"\n[3/4] Running seeded sweep for seeds {list(SEEDS)}...")
    for seed in SEEDS:
        fault_rows, stall_rows, step_rows = simulate_seed(seed, expected_step_time)
        out_dir = write_seed_outputs(seed, fault_rows, stall_rows, step_rows)
        print(f"      seed {seed}: {len(fault_rows)} fault event(s), {len(step_rows)} step_metrics rows -> {out_dir}")

    print("\n[4/4] Reverting C++ patch + rebuilding original ASTRA-sim...")
    revert_patch()
    rebuild()

    print("\n[done] Seeded sweep complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
