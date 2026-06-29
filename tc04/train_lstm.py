"""
tc04/train_lstm.py  —  Burst-timing LSTM training for TC-04

Trains a small LSTM on the inter-burst interval series from Pod A workload
traces, then saves the best checkpoint and key metrics for the report.

Improvements over Kyle's original:
  - seed fixed to 42 for reproducibility
  - z-score normalisation (raw ns values are ~1e7, unusable for MSE without it)
  - best-model checkpoint saved at lowest val MAE, not just final epoch
  - effective prediction horizon computed alongside MAE
  - all metrics written to tc04/training_metrics.json

Run order:
    1. python pod_a_pipeline/make_synthetic_workloads.py   (generate .et files)
    2. python tc04/train_lstm.py                           (this script)
    3. python tc04/generate_routing.py
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


# --- 1. Reproducibility and hyperparameters ---

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

WINDOW    = 8      # how many past intervals the model sees
HORIZON   = 3      # how many future intervals it predicts
HIDDEN    = 32
EPOCHS    = 20
BATCH     = 64
LR        = 1e-3
# 15% relative error per step is the threshold for counting a burst as "predicted"
HORIZON_THRESHOLD = 0.15


# --- 2. Dataset ---

class BurstDataset(Dataset):
    """Sliding-window dataset over a normalised interval series."""

    def __init__(self, series, window=WINDOW, horizon=HORIZON):
        self.series  = series
        self.window  = window
        self.horizon = horizon

    def __len__(self):
        return len(self.series) - self.window - self.horizon

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.window]
        y = self.series[idx + self.window : idx + self.window + self.horizon]
        return (torch.tensor(x, dtype=torch.float32).unsqueeze(-1),
                torch.tensor(y, dtype=torch.float32))


# --- 3. Model ---

class BurstLSTM(nn.Module):
    """Single-layer LSTM → linear head for multi-step burst prediction."""

    def __init__(self, hidden=HIDDEN, horizon=HORIZON):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fc   = nn.Linear(hidden, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # take the last hidden state


# --- 4. Data loading ---

def load_series():
    """Load all baseline .et files and concatenate their interval series."""
    from pod_a_pipeline.burst_extractor import extract_intervals
    base  = Path("pod_a_pipeline/workloads")
    files = sorted(base.glob("*.et"))
    if not files:
        raise FileNotFoundError(f"No .et files found in {base} — run make_synthetic_workloads.py first")
    series = []
    for f in files:
        series.extend(extract_intervals(str(f)))
    return np.array(series, dtype=np.float64)


# --- 5. Effective horizon ---

def compute_effective_horizon(model, val_dl, mean, std):
    """
    Count consecutive prediction steps where relative MAE stays below
    HORIZON_THRESHOLD. Stops at the first step that fails.
    Returns (horizon_int, per_step_pct_list).
    """
    per_step_abs  = np.zeros(HORIZON)
    per_step_true = np.zeros(HORIZON)

    model.eval()
    with torch.no_grad():
        for x, y in val_dl:
            pred_ns = model(x).numpy() * std + mean
            y_ns    = y.numpy()        * std + mean
            per_step_abs  += np.abs(pred_ns - y_ns).sum(axis=0)
            per_step_true += np.abs(y_ns).sum(axis=0)

    rel_frac = per_step_abs / per_step_true

    # consecutive steps from step 0 that pass the threshold
    consecutive = 0
    for r in rel_frac:
        if r < HORIZON_THRESHOLD:
            consecutive += 1
        else:
            break

    return consecutive, (rel_frac * 100).tolist()


# --- 6. Training loop ---

def main():
    raw = load_series()
    print(f"Loaded {len(raw)} inter-burst intervals from training traces.")

    # z-score normalisation — keeps MSE numerically stable at the ~1e7 ns scale
    mean, std = float(raw.mean()), float(raw.std())
    if std == 0:
        raise ValueError("Series has zero variance — check burst_extractor output.")
    series = (raw - mean) / std

    split    = int(0.8 * len(series))
    train_ds = BurstDataset(series[:split])
    val_ds   = BurstDataset(series[split:])
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH)

    model   = BurstLSTM()
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_mae   = float("inf")
    best_epoch = -1
    out_dir    = Path("tc04")
    out_dir.mkdir(exist_ok=True)

    for epoch in range(EPOCHS):
        # train pass
        model.train()
        for x, y in train_dl:
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # validation pass — compute relative MAE in original ns units
        model.eval()
        abs_errs, true_vals = [], []
        with torch.no_grad():
            for x, y in val_dl:
                pred_ns = model(x).numpy() * std + mean
                y_ns    = y.numpy()        * std + mean
                abs_errs.append(np.abs(pred_ns - y_ns))
                true_vals.append(np.abs(y_ns))

        rel_mae = (np.concatenate(abs_errs).mean() /
                   np.concatenate(true_vals).mean() * 100)

        print(f"Epoch {epoch+1:2d}: val relative MAE = {rel_mae:.2f}%")

        # save the best checkpoint rather than just the final epoch
        if rel_mae < best_mae:
            best_mae   = rel_mae
            best_epoch = epoch + 1
            torch.save(model.state_dict(), out_dir / "lstm_burst_predictor.pt")

    # reload best checkpoint before computing horizon
    model.load_state_dict(torch.load(out_dir / "lstm_burst_predictor.pt"))
    eff_horizon, per_step_pct = compute_effective_horizon(model, val_dl, mean, std)

    metrics = {
        "seed":                  SEED,
        "best_epoch":            best_epoch,
        "val_relative_mae_pct":  round(best_mae, 4),
        "effective_horizon":     eff_horizon,
        "per_step_rel_mae_pct":  [round(v, 4) for v in per_step_pct],
        "normalisation":         {"mean_ns": mean, "std_ns": std},
        "acceptance_criteria": {
            "mae_lt_10pct":         best_mae < 10.0,
            "horizon_gte_2_bursts": eff_horizon >= 2,
        },
    }
    with open(out_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n── TC-04 Training Summary ───────────────────────────────")
    print(f"Best epoch        : {best_epoch}")
    print(f"Val relative MAE  : {best_mae:.2f}%  (pass if < 10%)")
    print(f"Effective horizon : {eff_horizon} bursts  (pass if ≥ 2)")
    print(f"MAE criterion     : {'PASS' if best_mae < 10.0 else 'FAIL'}")
    print(f"Horizon criterion : {'PASS' if eff_horizon >= 2 else 'FAIL'}")
    print(f"Model saved       → {out_dir}/lstm_burst_predictor.pt")
    print(f"Metrics saved     → {out_dir}/training_metrics.json")


if __name__ == "__main__":
    main()
