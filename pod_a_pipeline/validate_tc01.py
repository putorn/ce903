"""
Pod A — TC-01 Validation
Reads the simulation log and checks whether the full trace-to-sim pipeline
completed correctly for all NPUs.

TC-01 pass criteria:
  1. err.log is empty — no simulation errors.
  2. All NPUs reported a Wall time (simulation completed for every NPU).
  3. Wall time > Comm time — compute phases (forward, backward, optimizer) added
     time on top of the AllReduce, confirming the dependency graph was respected.
  4. All NPUs have identical Wall time — ring AllReduce is symmetric.

Metrics recorded (for the Sprint 3 report):
  - JCT (Job Completion Time) = Wall time of any NPU (all equal).
  - Comm time = time spent in the gradient AllReduce collective.
  - Compute time = JCT − Comm time (time in forward + backward + optimizer).
  - Comm fraction = Comm time / JCT (how much of the step is communication).
"""

import os
import re
import sys

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR  = os.path.join(SCRIPT_DIR, "results")
LOG_PATH    = os.path.join(RESULT_DIR, "log.log")
ERR_PATH    = os.path.join(RESULT_DIR, "err.log")

# ── Parse log ─────────────────────────────────────────────────────────────────

def parse_log(path: str) -> dict:
    """Extract per-NPU Wall time, Comm time, and GPU time from the simulation log."""
    wall, comm, gpu = {}, {}, {}

    with open(path) as f:
        for line in f:
            m = re.search(r"sys\[(\d+)\].*Wall time:\s*(\d+)", line)
            if m:
                wall[int(m.group(1))] = int(m.group(2))

            m = re.search(r"sys\[(\d+)\].*Comm time:\s*(\d+)", line)
            if m:
                comm[int(m.group(1))] = int(m.group(2))

            # GPU time = total compute cycles (forward + backward + optimizer).
            m = re.search(r"sys\[(\d+)\].*GPU time:\s*(\d+)", line)
            if m:
                gpu[int(m.group(1))] = int(m.group(2))

    return {"wall": wall, "comm": comm, "gpu": gpu}


# ── TC-01 checks ──────────────────────────────────────────────────────────────

def check(label: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  {status}  {label}" + (f"  ({detail})" if detail else ""))
    return condition


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Pod A — TC-01 Pipeline Validation")
    print("  Trace: synthetic GPT-2 124M, 4 NPUs, Ring | 50 GB/s")
    print("=" * 65)

    if not os.path.exists(LOG_PATH):
        print(f"\n  ERROR: {LOG_PATH} not found.")
        print("  Run run_pipeline.sh first.")
        sys.exit(1)

    # ── Check 1: no errors ────────────────────────────────────────────────────
    errors = ""
    if os.path.exists(ERR_PATH):
        with open(ERR_PATH) as f:
            errors = f.read().strip()

    data = parse_log(LOG_PATH)
    wall = data["wall"]
    comm = data["comm"]

    print()

    all_passed = True
    all_passed &= check("err.log is empty (no simulation errors)",
                        errors == "",
                        errors[:80] if errors else "")

    # ── Check 2: exactly 4 simulation NPUs (0–3) are present ─────────────────
    # We check for NPUs 0–3 specifically rather than total count, because
    # ASTRA-sim appends to its log across runs. Clearing the log before each
    # run (done in run_pipeline.sh) is the primary guard, but this check is
    # robust to any leftover entries from other experiments.
    expected_npus = {0, 1, 2, 3}
    found_npus    = set(wall.keys())
    all_passed &= check("All 4 NPUs (0–3) present in log",
                        expected_npus.issubset(found_npus),
                        f"missing: {expected_npus - found_npus}" if not expected_npus.issubset(found_npus) else "")

    # Restrict metrics to only our 4 NPUs (ignore any stale entries).
    wall = {k: v for k, v in wall.items() if k in expected_npus}
    comm = {k: v for k, v in comm.items() if k in expected_npus}
    gpu  = {k: v for k, v in data["gpu"].items() if k in expected_npus}

    if not wall:
        print("\n  Cannot continue — no Wall time data found.")
        sys.exit(1)

    # ── Check 3: Wall time > Comm time (compute was simulated) ────────────────
    # If Wall == Comm for all NPUs, ASTRA-sim ignored the compute nodes.
    npu0_wall = wall.get(0, 0)
    npu0_comm = comm.get(0, 0)
    all_passed &= check("Wall time > Comm time (compute nodes modelled)",
                        npu0_wall > npu0_comm,
                        f"wall={npu0_wall}, comm={npu0_comm}")

    # ── Check 4: symmetric — all NPUs same Wall time ──────────────────────────
    unique_walls = set(wall.values())
    all_passed &= check("All NPUs identical Wall time (ring symmetric)",
                        len(unique_walls) == 1,
                        f"values: {sorted(wall.values())}")

    # ── Metrics summary ───────────────────────────────────────────────────────
    print()
    print("  Metrics (NPU 0):")
    jct          = npu0_wall
    comm_time    = npu0_comm
    gpu_time     = gpu.get(0, 0)
    compute_time = jct - comm_time
    comm_frac    = (comm_time / jct * 100) if jct else 0

    print(f"    JCT (Wall time)   : {jct:>12} cycles")
    print(f"    GPU time (compute): {gpu_time:>12} cycles  ({gpu_time/jct*100:.1f}% of JCT)")
    print(f"    Comm time         : {comm_time:>12} cycles  ({comm_frac:.1f}% of JCT)")
    print(f"    Exposed comm      : {compute_time:>12} cycles  (wall - compute)")

    print()
    print("  All NPUs (0–3):")
    for npu_id in sorted(wall.keys()):
        w = wall[npu_id]
        c = comm.get(npu_id, 0)
        g = gpu.get(npu_id, 0)
        print(f"    NPU {npu_id}: wall={w:>12}  gpu={g:>12}  comm={c:>12}")

    # ── TC-01 verdict ─────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    if all_passed:
        print("  ✓ TC-01 PASSED — pipeline complete, metrics recorded correctly.")
    else:
        print("  ✗ TC-01 FAILED — see check details above.")
    print("=" * 65)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
