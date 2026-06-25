import json
import torch
import numpy as np
from pathlib import Path
from train_lstm import BurstLSTM, WINDOW, HORIZON
from pod_a_pipeline.burst_extractor import extract_intervals

MODEL_PATH = "tc04/lstm_burst_predictor.pt"
FAILURE_TRACE_DIR = "pod_a_pipeline/workloads_failure"
OUTPUT_CONFIG = "pod_a_pipeline/configs/network/routing_predictor.json"

def load_failure_series():
    base = Path(FAILURE_TRACE_DIR)
    files = sorted(base.glob("*.et"))
    series = []
    for f in files:
        series.extend(extract_intervals(str(f)))
    return np.array(series)

def predict_bursts(model, series):
    preds = []
    for i in range(len(series) - WINDOW - HORIZON):
        x = torch.tensor(series[i:i+WINDOW]).float().unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            pred = model(x).numpy()[0]
        preds.append(pred)
    return np.array(preds)

def choose_routing_profile(preds):
    # Simple heuristic: if predicted burst > threshold, bias routing away from congested path
    burst_scores = preds.mean(axis=1)
    high_burst = burst_scores > np.percentile(burst_scores, 80)
    weight = 0.7 if high_burst.mean() > 0.2 else 0.5
    return {"routing_algorithm": "weighted", "path_weight": weight}

def main():
    model = BurstLSTM()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    series = load_failure_series()
    preds = predict_bursts(model, series)
    routing = choose_routing_profile(preds)

    with open(OUTPUT_CONFIG, "w") as f:
        json.dump(routing, f, indent=4)

    print(f"Generated routing config → {OUTPUT_CONFIG}")

if __name__ == "__main__":
    main()
