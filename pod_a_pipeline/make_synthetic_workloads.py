"""
pod_a_pipeline/make_synthetic_workloads.py

Generates synthetic .et workload files for when real GPT-2 RunPod traces
are unavailable. Used to unblock tc04/train_lstm.py locally.

Signal characteristics come from Kyle's Sprint 5 Lane A feasibility report:
  - Inter-burst intervals follow AR(1) with φ = 0.90 (strong periodicity)
  - Coefficient of variation typically < 5% under baseline 4-NPU conditions
  - Failure traces: mean interval rises when 4→3 NPU topology change kicks in,
    jitter spikes during the recovery window, then settles to a new steady state

The files are saved as numpy timestamp arrays with a .et extension.
burst_extractor.extract_intervals() handles this format automatically
when the Chakra protobuf library is not installed.

Run once before train_lstm.py:
    python pod_a_pipeline/make_synthetic_workloads.py
"""

import numpy as np
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)


# --- 1. Signal parameters (from Kyle's feasibility report) ---

MEAN_BASELINE_NS = 10_000_000   # ~10ms inter-burst interval, 4-NPU ring
STD_BASELINE_NS  =    500_000   # CV ≈ 5%
AR_PHI           =       0.90   # lag-1 autocorrelation target

MEAN_FAILURE_NS  = 13_000_000   # longer ring steps once the 3-NPU path takes over
STD_FAILURE_NS   =  1_200_000   # noticeably higher jitter during recovery

N_INTERVALS       = 600   # intervals per file — plenty for the WINDOW=8 LSTM
N_FILES_BASELINE  = 3
N_FILES_FAILURE   = 2


# --- 2. Signal generators ---

def ar1_series(n: int, mean: float, std: float, phi: float) -> np.ndarray:
    """AR(1) process with the given mean, std, and autocorrelation coefficient."""
    # innovation std derived from the target marginal std and AR(1) variance formula
    innovation_std = std * np.sqrt(1 - phi ** 2)
    x = np.empty(n)
    x[0] = mean
    for i in range(1, n):
        x[i] = mean + phi * (x[i - 1] - mean) + rng.normal(0, innovation_std)
    # clip to a sensible range — negative intervals are physically impossible
    return np.clip(x, mean * 0.5, mean * 2.5)


def make_failure_series(n: int) -> np.ndarray:
    """
    Simulate the TC-03 failure pattern: stable baseline → stall → 3-NPU steady state.

    60% of the trace is normal 4-NPU baseline. A short recovery window follows
    with the observed TC-03 stall of 1,031,879 cycles added to mean. The rest
    settles at the longer 3-NPU interval.
    """
    split    = int(n * 0.60)
    recovery = 20   # intervals of elevated jitter immediately post-failure

    baseline = ar1_series(split, MEAN_BASELINE_NS, STD_BASELINE_NS, AR_PHI)

    # stall transient — 1,031,879 cycles measured in TC-03
    stall = rng.normal(MEAN_BASELINE_NS + 1_031_879, STD_FAILURE_NS * 2, recovery)
    stall = np.clip(stall, 0, None)

    steady_failure = ar1_series(
        n - split - recovery,
        MEAN_FAILURE_NS, STD_FAILURE_NS, AR_PHI
    )

    return np.concatenate([baseline, stall, steady_failure])


def intervals_to_timestamps(intervals: np.ndarray, t0: float = 0.0) -> np.ndarray:
    """Convert inter-burst intervals to cumulative ALL_REDUCE start timestamps."""
    return t0 + np.concatenate([[0.0], np.cumsum(intervals[:-1])])


# --- 3. File writer ---

def write_et(path: Path, timestamps: np.ndarray) -> None:
    """
    Save a timestamp array as a numpy .npy file, then rename it to .et.
    burst_extractor.extract_intervals() picks this up via np.load() fallback.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # np.save always appends .npy — write to a temp name then rename
    tmp = path.with_suffix(".npy")
    np.save(str(tmp), timestamps.astype(np.float64))
    tmp.rename(path)
    print(f"  wrote {path}  ({len(timestamps)} timestamps)")


# --- 4. Main ---

def main():
    print("Generating baseline workloads …")
    base = Path("pod_a_pipeline/workloads")
    for i in range(N_FILES_BASELINE):
        intervals  = ar1_series(N_INTERVALS, MEAN_BASELINE_NS, STD_BASELINE_NS, AR_PHI)
        timestamps = intervals_to_timestamps(intervals)
        write_et(base / f"all_reduce_4npus_256MB.{i}.et", timestamps)

    print("\nGenerating failure workloads …")
    fail = Path("pod_a_pipeline/workloads_failure")
    for i in range(N_FILES_FAILURE):
        intervals  = make_failure_series(N_INTERVALS)
        timestamps = intervals_to_timestamps(intervals)
        write_et(fail / f"all_reduce_4npus_256MB_fail.{i}.et", timestamps)

    print("\nDone. Run next:")
    print("  python tc04/train_lstm.py")
    print("  python tc04/generate_routing.py")


if __name__ == "__main__":
    main()
