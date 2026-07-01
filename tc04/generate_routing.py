"""
tc04/generate_routing.py  —  Evaluate the trained LSTM on TC-03 failure traces
                              and produce ASTRA-sim system configs for TC-04.

Reads the model and normalisation stats from train_lstm.py, runs it against
the failure workloads, then writes two drop-in system JSON configs:
  tc04/system_ecmp.json       — ECMP baseline, Ring_4chunks defaults unchanged
  tc04/system_predictor.json  — LSTM-informed overrides applied

Improvements over Kyle's original:
  - routing_algorithm / path_weight replaced with real ASTRA-sim parameters
    (preferred-dataset-splits, active-chunks-per-dimension, scheduling-policy)
  - failure-trace MAE and effective horizon computed and saved
  - system JSONs are complete drop-ins for --system-configuration

Run order:
    1. python pod_a_pipeline/make_synthetic_workloads.py
    2. python tc04/train_lstm.py
    3. python tc04/generate_routing.py   (this script)
    4. Run ASTRA-sim twice — see printed instructions at end.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
from pathlib import Path
from train_lstm import BurstLSTM, WINDOW, HORIZON, HORIZON_THRESHOLD, SEED


# --- 1. Paths ---

MODEL_PATH        = Path("tc04/lstm_burst_predictor.pt")
METRICS_PATH      = Path("tc04/training_metrics.json")
FAILURE_TRACE_DIR = Path("pod_a_pipeline/workloads_failure")
BASE_SYSTEM_CFG   = Path("astra-sim/examples/system/native_collectives/Ring_4chunks.json")
OUT_DIR           = Path("tc04")

torch.manual_seed(SEED)
np.random.seed(SEED)


# --- 2. Helpers ---

def load_norm_stats():
    """Pull the mean/std saved by train_lstm.py — needed to denormalise predictions."""
    with open(METRICS_PATH) as f:
        m = json.load(f)
    return m["normalisation"]["mean_ns"], m["normalisation"]["std_ns"]


def load_failure_series():
    """Load failure trace .et files — same interface as the baseline loader."""
    from pod_a_pipeline.burst_extractor import extract_intervals
    files = sorted(FAILURE_TRACE_DIR.glob("*.et"))
    if not files:
        raise FileNotFoundError(f"No .et files in {FAILURE_TRACE_DIR}")
    series = []
    for f in files:
        series.extend(extract_intervals(str(f)))
    return np.array(series, dtype=np.float64)


# --- 3. Evaluation on failure traces ---

def evaluate_on_failure(model, series, mean, std):
    """
    Slide the model over the failure series and collect predictions.
    Returns overall relative MAE, effective horizon, per-step MAE list,
    and the raw predictions in ns for use in routing decisions.
    """
    norm       = (series - mean) / std
    preds_norm = []
    actuals    = []

    model.eval()
    with torch.no_grad():
        for i in range(len(norm) - WINDOW - HORIZON):
            x    = torch.tensor(norm[i:i+WINDOW],
                                dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            pred = model(x).numpy()[0]
            preds_norm.append(pred)
            actuals.append(norm[i + WINDOW : i + WINDOW + HORIZON])

    preds_norm = np.array(preds_norm)
    actuals    = np.array(actuals)
    preds_ns   = preds_norm * std + mean
    actuals_ns = actuals    * std + mean

    # overall relative MAE across all windows and steps
    rel_mae_pct = (np.abs(preds_ns - actuals_ns).mean() /
                   np.abs(actuals_ns).mean() * 100)

    # per-step breakdown — useful for seeing where accuracy degrades
    per_step_abs  = np.abs(preds_ns  - actuals_ns).sum(axis=0)
    per_step_true = np.abs(actuals_ns).sum(axis=0)
    per_step_frac = per_step_abs / per_step_true

    # consecutive steps from step 0 that stay below the threshold
    eff_horizon = 0
    for frac in per_step_frac:
        if frac < HORIZON_THRESHOLD:
            eff_horizon += 1
        else:
            break

    return rel_mae_pct, eff_horizon, (per_step_frac * 100).tolist(), preds_ns


# --- 4. Routing parameter selection ---

def choose_routing_params(preds_ns):
    """
    Map LSTM predictions to ASTRA-sim system config overrides.

    Parameters that actually affect scheduling in the analytical backend:
      preferred-dataset-splits    — chunk granularity for AllReduce
      active-chunks-per-dimension — in-flight chunk parallelism
      scheduling-policy           — LIFO (default) vs FIFO

    If more than 20% of prediction windows show intervals below the 20th
    percentile, the ring is under high burst density and finer chunking helps
    pipeline around congestion in the degraded 3-NPU topology.
    """
    burst_scores           = preds_ns.mean(axis=1)
    low_interval_threshold = np.percentile(burst_scores, 20)
    high_burst_fraction    = (burst_scores < low_interval_threshold).mean()

    if high_burst_fraction > 0.20:
        # high burst density — finer splits, more in-flight parallelism
        return {
            "preferred-dataset-splits":    8,
            "active-chunks-per-dimension": 2,
            "scheduling-policy":           "FIFO",
        }
    else:
        # normal load — keep ECMP defaults unchanged
        return {
            "preferred-dataset-splits":    4,
            "active-chunks-per-dimension": 1,
            "scheduling-policy":           "LIFO",
        }


# --- 5. Main ---

def main():
    OUT_DIR.mkdir(exist_ok=True)

    # normalisation stats from the training run
    mean, std = load_norm_stats()

    # load the best checkpoint saved by train_lstm.py
    model = BurstLSTM()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    series = load_failure_series()
    print(f"Loaded {len(series)} intervals from failure traces.")

    rel_mae_pct, eff_horizon, per_step_pct, preds_ns = evaluate_on_failure(
        model, series, mean, std
    )

    # save failure-trace metrics alongside the training metrics
    failure_metrics = {
        "seed":                    SEED,
        "failure_trace_dir":       str(FAILURE_TRACE_DIR),
        "topology_shift":          "4→3 NPU (TC-03 failure pattern)",
        "rel_mae_pct":             round(rel_mae_pct, 4),
        "effective_horizon":       eff_horizon,
        "per_step_rel_mae_pct":    [round(v, 4) for v in per_step_pct],
        "acceptance_criteria": {
            "mae_lt_10pct":         rel_mae_pct < 10.0,
            "horizon_gte_2_bursts": eff_horizon >= 2,
        },
    }
    with open(OUT_DIR / "failure_metrics.json", "w") as f:
        json.dump(failure_metrics, f, indent=4)

    # load the base Ring_4chunks config — ECMP version is unchanged from this
    if not BASE_SYSTEM_CFG.exists():
        raise FileNotFoundError(f"Base system config not found: {BASE_SYSTEM_CFG}")
    with open(BASE_SYSTEM_CFG) as f:
        base_cfg = json.load(f)

    with open(OUT_DIR / "system_ecmp.json", "w") as f:
        json.dump(base_cfg, f, indent=4)

    # predictor version applies the LSTM-derived overrides on top of the base
    routing_params = choose_routing_params(preds_ns)
    predictor_cfg  = dict(base_cfg)
    predictor_cfg.update(routing_params)
    with open(OUT_DIR / "system_predictor.json", "w") as f:
        json.dump(predictor_cfg, f, indent=4)

    print(f"\n── TC-04 Failure-Trace Evaluation ───────────────────────")
    print(f"Failure trace MAE   : {rel_mae_pct:.2f}%  (pass if < 10%)")
    print(f"Effective horizon   : {eff_horizon} bursts  (pass if ≥ 2)")
    print(f"MAE criterion       : {'PASS' if rel_mae_pct < 10.0 else 'FAIL'}")
    print(f"Horizon criterion   : {'PASS' if eff_horizon >= 2 else 'FAIL'}")
    print(f"\nRouting overrides applied to predictor config:")
    for k, v in routing_params.items():
        print(f"  {k}: {v}  (was: {base_cfg.get(k, 'N/A')})")
    print(f"\nSaved → {OUT_DIR}/system_ecmp.json")
    print(f"Saved → {OUT_DIR}/system_predictor.json")
    print(f"Saved → {OUT_DIR}/failure_metrics.json")
    print(f"""
── Next step: run ASTRA-sim twice with seed 42, TC-03 failure pattern ─────────
  Run 1 (ECMP baseline):
    --system-configuration=tc04/system_ecmp.json
    --network-configuration=<your TC-03 failure network config>

  Run 2 (Predictor-informed):
    --system-configuration=tc04/system_predictor.json
    --network-configuration=<your TC-03 failure network config>

  Record workload_finished_time from log/log.log for each run.
  That is your median JCT. Negative delta = improvement over ECMP.
""")


if __name__ == "__main__":
    main()
