"""
Sprint 4 — Final Results Report
TC-03: GSP Fault Injection · JCT Recovery Penalty · NCCL Stall Duration

Reads all result files produced by run_fault_injection.py and run_c_injection.py
and prints a single clean summary of all Sprint 4 metrics.

Usage:
    conda run -n p903 python pod_a_pipeline/sprint4_results.py
"""

import csv
import json
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_BASELINE  = os.path.join(SCRIPT_DIR, "results",              "log.log")
LOG_FAILURE   = os.path.join(SCRIPT_DIR, "results_with_failure", "log.log")
LOG_FAULT_RUN = os.path.join(SCRIPT_DIR, "results_fault_run",    "log.log")
CONTRACT_JSON = os.path.join(SCRIPT_DIR, "results",              "observability_contract.json")
FAULT_CSV     = os.path.join(SCRIPT_DIR, "results_fault_run",    "fault_events.csv")
STALL_CSV     = os.path.join(SCRIPT_DIR, "stall_events.csv")


def parse_wall_times(log_path):
    times = []
    if not os.path.exists(log_path):
        return times
    with open(log_path) as f:
        for line in f:
            m = re.search(r"Wall time:\s*(\d+)", line)
            if m:
                times.append(int(m.group(1)))
    return times


def read_csv_value(path, id_col, val_col, row_id):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get(id_col) == row_id:
                return float(row[val_col])
    return None


def avg(lst):
    return sum(lst) / len(lst) if lst else None


# ── Collect data ───────────────────────────────────────────────────────────────

baseline_times = parse_wall_times(LOG_BASELINE)
failure_times  = parse_wall_times(LOG_FAILURE)

jct_baseline = avg(baseline_times)
jct_failure  = avg(failure_times)

contract = {}
if os.path.exists(CONTRACT_JSON):
    with open(CONTRACT_JSON) as f:
        contract = json.load(f)

fault_t0       = read_csv_value(FAULT_CSV, "collective_instance_id", "t_fault",        "gpt2")
stall_detected = read_csv_value(STALL_CSV, "collective_instance_id", "t_stall_detected", "gpt2")

# ── Print report ──────────────────────────────────────────────────────────────

print()
print("=" * 72)
print("  Sprint 4 — TC-03 Final Results")
print("  GSP Error (XID 119) · Single-fault demo · Ring topology")
print("  CE903 Group Project 3 · Managing Network for LLM Training")
print("=" * 72)

print()
print("  Simulation Setup")
print("  " + "-" * 68)
print(f"  Topology          : Ring")
print(f"  Bandwidth         : 50 GB/s")
print(f"  Latency           : 500 ns")
print(f"  Baseline NPUs     : 4")
print(f"  Failure NPUs      : 3  (NPU {contract.get('fault_node_id', 0)} removed)")
print(f"  Recovery window   : {contract.get('recovery_window_hrs', 0.3)} hrs  (UIUC/IBM average)")

print()
print("  JCT Recovery Penalty  (FR-G2, TC-03)")
print("  " + "-" * 68)
if jct_baseline and jct_failure:
    penalty = jct_failure - jct_baseline
    print(f"  Baseline JCT      : {jct_baseline:>16,.0f}  cycles")
    print(f"  Failure  JCT      : {jct_failure:>16,.0f}  cycles")
    print(f"  Penalty           : {penalty:>16,.0f}  cycles  ({penalty / jct_baseline * 100:.1f}% overhead)")
else:
    print("  [missing] Run run_fault_injection.py first.")

print()
print("  NCCL Stall Duration  (FR-N2)")
print("  " + "-" * 68)
if fault_t0 is not None and stall_detected is not None:
    stall_duration = stall_detected - fault_t0
    print(f"  Fault fired (T0)  : {fault_t0:>16,.0f}  cycles  ← C++ injection")
    print(f"  Stall detected    : {stall_detected:>16,.0f}  cycles  (T0 + Comm×k_hard)")
    print(f"  Stall duration    : {stall_duration:>16,.0f}  cycles")
elif fault_t0 is not None:
    print(f"  Fault fired (T0)  : {fault_t0:>16,.0f}  cycles  ← C++ injection")
    print(f"  Stall detected    : {'N/A':>16}  (stall_events.csv missing)")
else:
    print("  [missing] Run run_c_injection.py to generate fault_events.csv.")

print()
print("  Cluster Impact")
print("  " + "-" * 68)
print(f"  Fault node        : NPU {contract.get('fault_node_id', 0)}")
print(f"  Nodes affected    : {contract.get('nodes_affected_pct', 25.0):.1f}%")
print(f"  Reboot events     : {contract.get('reboot_events_per_epoch', 1)} per epoch")

print()
print("  Source Files")
print("  " + "-" * 68)
print(f"  Baseline log      : results/log.log")
print(f"  Failure log       : results_with_failure/log.log")
print(f"  C++ injection log : results_fault_run/log.log")
print(f"  Fault events      : results_fault_run/fault_events.csv")
print(f"  Contract          : results/observability_contract.json")
print()
print("=" * 72)
