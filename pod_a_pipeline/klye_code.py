"""
Pod B — JCT Recovery Penalty and NCCL Stall Duration Harness

Critical metric:
    JCT Recovery Penalty = JCT_with_failure - JCT_baseline

This follows Navami's definition:
    measure baseline Job Completion Time,
    measure Job Completion Time under failure/recovery,
    report the difference.

This harness works before a fault injector exists:
    if no failure results are available, use baseline results as a self-check.
    The expected recovery penalty is then 0.

Stretch metric:
    NCCL Stall Duration, FR-N2 = t_stall_detected - t_fault

Expected directory layout:

    pod_b_traffic/
        results/
            1MB/
                4npus/log.log
                4npus/err.log
                16npus/log.log
                ...
            256MB/
            512MB/

Optional future failure layout:

    pod_b_traffic/
        results_with_failure/
            1MB/
                4npus/log.log
                ...
            256MB/
            512MB/

Optional future fault/stall event files:

    pod_b_traffic/
        fault_events.csv
        stall_events.csv

fault_events.csv columns:
    collective_instance_id,t_fault

stall_events.csv columns:
    collective_instance_id,t_stall_detected

Example:
    collective_instance_id,t_fault
    256MB_16npus,12345

    collective_instance_id,t_stall_detected
    256MB_16npus,12500
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


# ── Config ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_BASELINE_DIR = os.path.join(SCRIPT_DIR, "results")
DEFAULT_FAILURE_DIR = os.path.join(SCRIPT_DIR, "results_with_failure")

DEFAULT_FAULT_EVENTS = os.path.join(SCRIPT_DIR, "fault_events.csv")
DEFAULT_STALL_EVENTS = os.path.join(SCRIPT_DIR, "stall_events.csv")



# ── Basic types ────────────────────────────────────────────────────────────────

class RunType(Enum):
    BASELINE = "baseline"
    WITH_FAILURE = "with_failure"


@dataclass(frozen=True)
class JobTiming:
    """
    Per-job timing within a run.

    For current ASTRA-sim logs, we often only have completion-like timing fields
    such as Wall time or Comm time. In that case, this harness treats t_start as
    zero and t_end as the parsed completion duration.
    """

    job_id: str
    t_start: float
    t_end: float

    @property
    def jct(self) -> float:
        return self.t_end - self.t_start


@dataclass(frozen=True)
class RunSummary:
    """
    One simulator run.

    jobs is keyed by a stable job_id, for example:
        256MB_16npus
    """

    run_id: str
    run_type: RunType
    jobs: Dict[str, JobTiming]


@dataclass(frozen=True)
class RecoveryPenalty:
    """
    Navami-style JCT Recovery Penalty for a single job.
    """

    job_id: str
    jct_baseline: float
    jct_with_failure: float
    penalty: float


@dataclass(frozen=True)
class StallToFault:
    """
    NCCL Stall Duration / FR-N2 for a single collective instance.
    """

    collective_instance_id: str
    t_fault: float
    t_stall_detected: float

    @property
    def stall_duration(self) -> float:
        return self.t_stall_detected - self.t_fault


# ── Log parsing ────────────────────────────────────────────────────────────────

def parse_first_number(pattern: str, text: str) -> Optional[float]:
    """
    Return the first numeric capture group for a regex pattern.
    """

    match = re.search(pattern, text)
    if not match:
        return None
    return float(match.group(1))


def parse_log_jct(log_path: str) -> Optional[float]:
    """
    Extract a JCT-like value from one ASTRA-sim log.

    Priority:
        1. Explicit JCT fields, if future logs include them.
        2. Wall time, because it represents end-to-end completion duration.
        3. Comm time, as a fallback for communication-only experiments.

    Current baseline logs from Pod B usually include lines like:
        Wall time: 12345
        Comm time: 6789
    """

    if not os.path.exists(log_path):
        return None

    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()

    explicit_jct = parse_first_number(
        r"(?:JCT|Job Completion Time|Job completion time)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        text,
    )
    if explicit_jct is not None:
        return explicit_jct

    wall_time = parse_first_number(
        r"Wall time:\s*([0-9]+(?:\.[0-9]+)?)",
        text,
    )
    if wall_time is not None:
        return wall_time

    comm_time = parse_first_number(
        r"Comm time:\s*([0-9]+(?:\.[0-9]+)?)",
        text,
    )
    if comm_time is not None:
        return comm_time

    return None


def load_run_summary(results_dir: str, run_id: str, run_type: RunType) -> RunSummary:
    """
    Load a run summary from results_dir/log.log (flat single-run format).
    """

    jobs: Dict[str, JobTiming] = {}

    log_path = os.path.join(results_dir, "log.log")
    jct = parse_log_jct(log_path)

    if jct is not None:
        jobs["gpt2"] = JobTiming(job_id="gpt2", t_start=0.0, t_end=jct)

    return RunSummary(run_id=run_id, run_type=run_type, jobs=jobs)


# ── JCT Recovery Penalty harness ───────────────────────────────────────────────

class JCTRecoveryHarness:
    """
    Given:
        - one baseline run
        - one with-failure run

    Compute:
        JCT Recovery Penalty = JCT_with_failure - JCT_baseline
    """

    def __init__(self, baseline: RunSummary, with_failure: RunSummary):
        if baseline.run_type != RunType.BASELINE:
            raise ValueError("baseline RunSummary must have run_type=BASELINE")
        if with_failure.run_type != RunType.WITH_FAILURE:
            raise ValueError("with_failure RunSummary must have run_type=WITH_FAILURE")

        self.baseline = baseline
        self.with_failure = with_failure

    def compute_penalties(self) -> Dict[str, RecoveryPenalty]:
        penalties: Dict[str, RecoveryPenalty] = {}

        for job_id, base_job in self.baseline.jobs.items():
            failure_job = self.with_failure.jobs.get(job_id)

            if failure_job is None:
                continue

            jct_baseline = base_job.jct
            jct_with_failure = failure_job.jct

            penalties[job_id] = RecoveryPenalty(
                job_id=job_id,
                jct_baseline=jct_baseline,
                jct_with_failure=jct_with_failure,
                penalty=jct_with_failure - jct_baseline,
            )

        return penalties


# ── NCCL Stall Duration / FR-N2 harness ───────────────────────────────────────

def load_event_times_csv(path: str, id_column: str, time_column: str) -> Dict[str, float]:
    """
    Load event timestamps from a CSV file.

    Returns:
        event_id -> timestamp
    """

    if not os.path.exists(path):
        return {}

    events: Dict[str, float] = {}

    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            event_id = row.get(id_column)
            timestamp = row.get(time_column)

            if not event_id or timestamp is None or timestamp == "":
                continue

            events[event_id] = float(timestamp)

    return events


class NCCLStallDurationHarness:
    """
    Compute FR-N2:
        NCCL Stall Duration = t_stall_detected - t_fault
    """

    def __init__(
        self,
        fault_times: Dict[str, float],
        stall_detected_times: Dict[str, float],
    ):
        self.fault_times = fault_times
        self.stall_detected_times = stall_detected_times

    def compute_stall_durations(self) -> Dict[str, StallToFault]:
        results: Dict[str, StallToFault] = {}

        for collective_instance_id, t_fault in self.fault_times.items():
            t_stall_detected = self.stall_detected_times.get(collective_instance_id)

            if t_stall_detected is None:
                continue

            results[collective_instance_id] = StallToFault(
                collective_instance_id=collective_instance_id,
                t_fault=t_fault,
                t_stall_detected=t_stall_detected,
            )

        return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_recovery_penalty_report(penalties: Dict[str, RecoveryPenalty]) -> None:
    print()
    print("=" * 88)
    print("  JCT Recovery Penalty")
    print("  Definition: JCT_with_failure - JCT_baseline")
    print("=" * 88)

    if not penalties:
        print("  No comparable baseline/failure jobs found.")
        print("=" * 88)
        return

    print()
    print(f"  {'Job':<18}  {'Baseline JCT':>16}  {'Failure JCT':>16}  {'Penalty':>16}")
    print("  " + "-" * 82)

    total_penalty = 0.0

    for job_id in sorted(penalties):
        row = penalties[job_id]
        total_penalty += row.penalty

        print(
            f"  {row.job_id:<18}  "
            f"{row.jct_baseline:>16.3f}  "
            f"{row.jct_with_failure:>16.3f}  "
            f"{row.penalty:>16.3f}"
        )

    average_penalty = total_penalty / len(penalties)

    print("  " + "-" * 82)
    print(f"  {'Average':<18}  {'':>16}  {'':>16}  {average_penalty:>16.3f}")
    print()
    print(f"  Comparable jobs: {len(penalties)}")
    print("=" * 88)


def print_stall_duration_report(stall_durations: Dict[str, StallToFault]) -> None:
    print()
    print("=" * 88)
    print("  NCCL Stall Duration / FR-N2")
    print("  Definition: t_stall_detected - t_fault")
    print("=" * 88)

    if not stall_durations:
        print("  No fault/stall pairs found yet.")
        print("  This is expected before fault injection and stall detection are wired in.")
        print("=" * 88)
        return

    print()
    print(
        f"  {'Collective Instance':<28}  "
        f"{'t_fault':>14}  "
        f"{'t_stall_detected':>18}  "
        f"{'Stall Duration':>18}"
    )
    print("  " + "-" * 82)

    total_duration = 0.0

    for collective_instance_id in sorted(stall_durations):
        row = stall_durations[collective_instance_id]
        total_duration += row.stall_duration

        print(
            f"  {row.collective_instance_id:<28}  "
            f"{row.t_fault:>14.3f}  "
            f"{row.t_stall_detected:>18.3f}  "
            f"{row.stall_duration:>18.3f}"
        )

    average_duration = total_duration / len(stall_durations)

    print("  " + "-" * 82)
    print(f"  {'Average':<28}  {'':>14}  {'':>18}  {average_duration:>18.3f}")
    print()
    print(f"  Matched fault/stall pairs: {len(stall_durations)}")
    print("=" * 88)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure JCT Recovery Penalty and optional NCCL Stall Duration.",
    )

    parser.add_argument(
        "--baseline-dir",
        default=DEFAULT_BASELINE_DIR,
        help="Directory containing baseline results. Default: pod_b_traffic/results",
    )

    parser.add_argument(
        "--failure-dir",
        default=DEFAULT_FAILURE_DIR,
        help="Directory containing with-failure results. Default: pod_b_traffic/results_with_failure",
    )

    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Compare baseline results against themselves. Expected penalty is zero.",
    )

    parser.add_argument(
        "--fault-events",
        default=DEFAULT_FAULT_EVENTS,
        help="CSV containing collective_instance_id,t_fault.",
    )

    parser.add_argument(
        "--stall-events",
        default=DEFAULT_STALL_EVENTS,
        help="CSV containing collective_instance_id,t_stall_detected.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    baseline = load_run_summary(
        results_dir=args.baseline_dir,
        run_id="baseline",
        run_type=RunType.BASELINE,
    )

    if not baseline.jobs:
        print(f"[recovery] ERROR: no baseline jobs found in {args.baseline_dir}")
        print("[recovery] Run pod_b_traffic/run_baseline.sh first.")
        return 1

    if args.self_check or not os.path.exists(args.failure_dir):
        if not args.self_check:
            print(
                f"[recovery] No failure directory found at {args.failure_dir}; "
                "using baseline as a self-check."
            )

        with_failure = RunSummary(
            run_id="baseline_self_check",
            run_type=RunType.WITH_FAILURE,
            jobs=baseline.jobs,
        )
    else:
        with_failure = load_run_summary(
            results_dir=args.failure_dir,
            run_id="with_failure",
            run_type=RunType.WITH_FAILURE,
        )

    recovery_harness = JCTRecoveryHarness(
        baseline=baseline,
        with_failure=with_failure,
    )

    penalties = recovery_harness.compute_penalties()
    print_recovery_penalty_report(penalties)

    fault_times = load_event_times_csv(
        path=args.fault_events,
        id_column="collective_instance_id",
        time_column="t_fault",
    )

    stall_detected_times = load_event_times_csv(
        path=args.stall_events,
        id_column="collective_instance_id",
        time_column="t_stall_detected",
    )

    stall_harness = NCCLStallDurationHarness(
        fault_times=fault_times,
        stall_detected_times=stall_detected_times,
    )

    stall_durations = stall_harness.compute_stall_durations()
    print_stall_duration_report(stall_durations)

    return 0


if __name__ == "__main__":
    sys.exit(main())