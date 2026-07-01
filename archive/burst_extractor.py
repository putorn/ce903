import glob
import csv
from chakra.et_def import ExecutionTrace  # Adjusted to match package def

def extract_bursts(prefix, out_csv):
    rows = []
    # Match the execution trace files 
    for npu_file in sorted(glob.glob(f"{prefix}.*.et")):
        npu_id = int(npu_file.split(".")[-2])
        trace = ExecutionTrace()
        with open(npu_file, "rb") as f:
            trace.ParseFromString(f.read())
        
        for ev in trace.events:
            # Filtering for collective communication bursts
            if ev.collective_type == "ALL_REDUCE":
                rows.append({
                    "npu": npu_id,
                    "t_start_cycles": ev.start_time,
                    "t_end_cycles": ev.end_time,
                })
                
    if not rows:
        print(f"No All-Reduce events found for prefix: {prefix}")
        return

    rows.sort(key=lambda r: r["t_start_cycles"])
    for i, r in enumerate(rows):
        r["burst_id"] = i
        
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Successfully extracted bursts to {out_csv}")

if __name__ == "__main__":
    # Pointing to Pod A standard workloads
    extract_bursts(
        prefix="pod_a_pipeline/workloads/all_reduce_4npus_256MB", 
        out_csv="pod_a_pipeline/bursts_4npu_256MB.csv"
    )
    
    # Pointing to Pod A failure scenario workloads
    extract_bursts(
        prefix="pod_a_pipeline/workloads_failure/all_reduce_4npus_256MB_fail", 
        out_csv="pod_a_pipeline/bursts_4npu_256MB_failure.csv"
    )
