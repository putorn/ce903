import glob
import csv
from chakra.et_def import ExecutionTrace  # adjust import to actual package

def extract_bursts(prefix, out_csv):
    rows = []
    for npu_file in sorted(glob.glob(f"{prefix}.*.et")):
        npu_id = int(npu_file.split(".")[-2])
        trace = ExecutionTrace()
        with open(npu_file, "rb") as f:
            trace.ParseFromString(f.read())
        for ev in trace.events:
            if ev.collective_type == "ALL_REDUCE":
                rows.append({
                    "npu": npu_id,
                    "t_start_cycles": ev.start_time,
                    "t_end_cycles": ev.end_time,
                })
    rows.sort(key=lambda r: r["t_start_cycles"])
    for i, r in enumerate(rows):
        r["burst_id"] = i
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    extract_bursts("pod_b_traffic/workloads/all_reduce_4npus_256MB", "Lane A directory/bursts_4npu_256MB.csv")
