"""
pod_a_pipeline/burst_extractor.py

Parses Chakra .et execution trace files and returns the inter-burst interval
series used by tc04/train_lstm.py and tc04/generate_routing.py.

The key function is extract_intervals(), which train_lstm.py calls directly:
    from pod_a_pipeline.burst_extractor import extract_intervals
    series.extend(extract_intervals(str(f)))

It tries the Chakra protobuf parser first. If that is not installed, it falls
back to the numpy format written by make_synthetic_workloads.py, so the LSTM
pipeline runs either way.

extract_bursts() is Kyle's original CSV-writing helper, kept for compatibility.
"""

import glob
import csv
from typing import List

# --- 1. Chakra protobuf parser ---
# Two import paths because the chakra package reorganised between versions.
# If neither works, _parse_trace is set to None and the numpy fallback takes over.

try:
    from chakra.et_def.et_def_pb2 import ExecutionTrace

    def _parse_trace(et_path: str):
        trace = ExecutionTrace()
        with open(et_path, "rb") as f:
            trace.ParseFromString(f.read())
        return trace.nodes

except ImportError:
    try:
        from chakra.et_def import ExecutionTrace

        def _parse_trace(et_path: str):
            trace = ExecutionTrace()
            with open(et_path, "rb") as f:
                trace.ParseFromString(f.read())
            return trace.nodes

    except ImportError:
        _parse_trace = None


# --- 2. Numpy fallback for synthetic workloads ---
# make_synthetic_workloads.py saves timestamps as numpy arrays with a .et
# extension. This tries np.load() when the Chakra path fails or returns nothing.

def _try_numpy_load(et_path: str) -> List[float]:
    """Load timestamps saved by make_synthetic_workloads.py and return intervals."""
    import numpy as np
    try:
        timestamps = np.load(et_path, allow_pickle=False).astype(np.float64)
        if timestamps.ndim != 1 or len(timestamps) < 2:
            return []
        return list(np.sort(timestamps)[1:] - np.sort(timestamps)[:-1])
    except Exception:
        return []


def _parse_events(et_path: str) -> List[dict]:
    """Return ALL_REDUCE events from a single .et file as a list of dicts."""
    if _parse_trace is None:
        raise ImportError(
            "chakra package not found — install with: pip install chakra"
        )

    events = []
    for node in _parse_trace(et_path):
        ct = getattr(node, "collective_type", None)
        ct_str = str(ct).upper() if ct is not None else ""
        # collective_type is an integer enum (0) in most chakra builds,
        # but some versions expose it as the string "ALL_REDUCE"
        is_all_reduce = ct_str in ("ALL_REDUCE", "0") or ct == 0
        if is_all_reduce:
            events.append({
                "t_start_cycles": float(
                    node.start_micros if hasattr(node, "start_micros")
                    else node.start_time
                ),
                "t_end_cycles": float(
                    node.end_micros if hasattr(node, "end_micros")
                    else node.end_time
                ),
            })
    return events


# --- 3. Public API ---

def extract_intervals(et_file_path: str) -> List[float]:
    """
    Parse one .et file and return the inter-burst interval series.

    Δᵢ = t_{i+1} − t_i across sorted ALL_REDUCE start timestamps.
    Returns an empty list if the file has fewer than two events.
    """
    # try the real Chakra parser first
    if _parse_trace is not None:
        try:
            events = _parse_events(et_file_path)
            if events:
                timestamps = sorted(e["t_start_cycles"] for e in events)
                return [timestamps[i + 1] - timestamps[i]
                        for i in range(len(timestamps) - 1)]
        except Exception:
            pass

    # fall back to the numpy synthetic format
    return _try_numpy_load(et_file_path)


def extract_bursts(prefix: str, out_csv: str) -> None:
    """
    Kyle's original helper: write ALL_REDUCE events across prefix.*.et files
    to a CSV. Not called by the LSTM pipeline — kept for compatibility.
    """
    rows = []
    for npu_file in sorted(glob.glob(f"{prefix}.*.et")):
        npu_id = int(npu_file.split(".")[-2])
        events = _parse_events(npu_file)
        for ev in events:
            rows.append({
                "npu":            npu_id,
                "t_start_cycles": ev["t_start_cycles"],
                "t_end_cycles":   ev["t_end_cycles"],
            })

    if not rows:
        print(f"No ALL_REDUCE events found for prefix: {prefix}")
        return

    rows.sort(key=lambda r: r["t_start_cycles"])
    for i, r in enumerate(rows):
        r["burst_id"] = i

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["burst_id", "npu", "t_start_cycles", "t_end_cycles"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Extracted {len(rows)} bursts to {out_csv}")


if __name__ == "__main__":
    # pointing to the Pod A standard and failure workloads
    extract_bursts(
        prefix="pod_a_pipeline/workloads/all_reduce_4npus_256MB",
        out_csv="pod_a_pipeline/bursts_4npu_256MB.csv",
    )
    extract_bursts(
        prefix="pod_a_pipeline/workloads_failure/all_reduce_4npus_256MB_fail",
        out_csv="pod_a_pipeline/bursts_4npu_256MB_failure.csv",
    )
