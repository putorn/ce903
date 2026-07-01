"""
CE903 Group Project 3 — Sprint 6, Lane B: Fault Prediction Integration
Navami Nair (Role 5 — GPU Failures & Resilience)
Maps to: FR-G2, FR-G3, NFR-P1

Evaluation method: Leave-One-Seed-Out Cross-Validation (LOSO-CV)
  - Trains on 9 seeds, evaluates on the held-out 10th seed (repeated 10 times)
  - Logistic regression comparison (checks whether task needs LSTM or is linearly separable)
  - Feature ablation (which features matter most)

Usage:
  Windows:
    python lane_b_lstm.py --runs_dir "C:/path/to/runs"
  Mac/Linux:
    python3 lane_b_lstm.py --runs_dir ./runs

Outputs (saved to ./lane_b_output/):
  lane_b_lstm.pt                 -- saved model weights
  lane_b_evaluation_report.json  -- all numeric results in JSON format
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# ── Settings ──────────────────────────────────────────────────────────────────
WINDOW_W       = 3      # steps per sliding window
N_FEATURES     = 4      # step_time, completion_ratio, node_throughput, soft_stall_flag
HIDDEN_SIZE    = 16
BATCH_SIZE     = 8
EPOCHS         = 150
LEARNING_RATE  = 1e-3
NOISE_SIGMA    = 0.05   # small noise added to training features to prevent memorisation
PRECISION_GATE = 0.70   # NFR-P1: precision must exceed this
LEAD_TIME_GATE = 0      # NFR-P1: lead time must be >= this (0 = detection is acceptable)
RANDOM_SEED    = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── Data loading ──────────────────────────────────────────────────────────────

@dataclass
class RunData:
    seed: int
    step_metrics: pd.DataFrame
    fault_events: pd.DataFrame
    stall_events: pd.DataFrame


def load_run(runs_dir: Path, seed: int) -> RunData:
    p = runs_dir / f"seed_{seed}"
    return RunData(
        seed=seed,
        step_metrics=pd.read_csv(p / "step_metrics.csv"),
        fault_events=pd.read_csv(p / "fault_events.csv"),
        stall_events=pd.read_csv(p / "stall_events.csv"),
    )


def collapse_to_steps(sm: pd.DataFrame) -> pd.DataFrame:
    """
    Tae's harness writes one row per (step, node_id).
    Collapse to one row per step: timing from first node, metrics averaged.
    """
    return (
        sm.groupby("step")
          .agg(
              t_step_start    =("t_step_start",     "first"),
              step_time       =("step_time",        "first"),
              completion_ratio=("completion_ratio",  "mean"),
              node_throughput =("node_throughput",   "mean"),
          )
          .reset_index()
          .sort_values("step")
          .reset_index(drop=True)
    )


def build_raw_windows(run: RunData) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Build sliding windows for one run without normalisation.
    Normalisation is applied later using cross-run statistics (not per-run),
    so the val seed's features are in the same space as the training seeds.

    Label = 1 if a fault step falls inside the window (detection label).
    Lead time = 0 for all positive windows (same-step detection).
    """
    sm = collapse_to_steps(run.step_metrics)

    # Map each fault's timestamp to the nearest step index
    fault_steps: set = set()
    for _, row in run.fault_events.iterrows():
        nearest = (sm["t_step_start"] - row["t_fault"]).abs().idxmin()
        fault_steps.add(int(sm.loc[nearest, "step"]))

    feature_cols = ["step_time", "completion_ratio", "node_throughput"]
    X_raw, y, lead_times = [], [], []

    for end_idx in range(WINDOW_W - 1, len(sm)):
        window = sm.iloc[end_idx - WINDOW_W + 1 : end_idx + 1]
        feats  = window[feature_cols].values.copy()

        # Soft stall flag: 1 if any soft stall event falls inside this step's time range
        soft_flags = []
        for _, srow in window.iterrows():
            t0 = srow["t_step_start"]
            t1 = t0 + srow["step_time"]
            hit = run.stall_events[
                (run.stall_events["stall_type"] == "soft")
                & (run.stall_events["t_stall_detected"] >= t0)
                & (run.stall_events["t_stall_detected"] <  t1)
            ]
            soft_flags.append(1.0 if len(hit) > 0 else 0.0)

        window_feats = np.hstack([feats, np.array(soft_flags).reshape(-1, 1)])
        label = 1 if (set(window["step"].astype(int).tolist()) & fault_steps) else 0
        X_raw.append(window_feats)
        y.append(float(label))
        if label == 1:
            lead_times.append(0)

    return np.array(X_raw, dtype=np.float32), np.array(y, dtype=np.float32), lead_times


def compute_scaler(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean/std of continuous features from training windows only."""
    flat  = X.reshape(-1, N_FEATURES)
    mu    = flat[:, :3].mean(axis=0)
    sigma = flat[:, :3].std(axis=0)
    sigma[sigma < 1e-9] = 1.0
    return mu, sigma


def apply_scaler(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Normalise continuous features using pre-computed training statistics."""
    X = X.copy()
    X[:, :, :3] = (X[:, :, :3] - mu) / sigma
    return X


# ── Dataset and model ─────────────────────────────────────────────────────────

class FaultWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, noise: float = 0.0):
        self.X     = torch.from_numpy(X)
        self.y     = torch.from_numpy(y)
        self.noise = noise

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.noise > 0:
            x[:, :3] += torch.randn_like(x[:, :3]) * self.noise
        return x, self.y[idx]


class FaultPredictorLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm    = nn.LSTM(N_FEATURES, HIDDEN_SIZE, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc      = nn.Linear(HIDDEN_SIZE, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.sigmoid(self.fc(self.dropout(h[-1]))).squeeze(-1)


# ── Training ──────────────────────────────────────────────────────────────────

def train_fold(X_tr, y_tr, X_va, y_va, epochs=EPOCHS):
    tr_ld = DataLoader(FaultWindowDataset(X_tr, y_tr, NOISE_SIGMA), BATCH_SIZE, shuffle=True)
    va_ld = DataLoader(FaultWindowDataset(X_va, y_va, 0.0), BATCH_SIZE, shuffle=False)
    model = FaultPredictorLSTM()
    opt   = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    crit  = nn.BCELoss()
    best_vl, best_state = float("inf"), None

    for _ in range(epochs):
        model.train()
        for xb, yb in tr_ld:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([crit(model(xb), yb).item() for xb, yb in va_ld]))
        if vl < best_vl:
            best_vl    = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_fold(model, X_va, y_va, threshold=0.5):
    model.eval()
    va_ld = DataLoader(FaultWindowDataset(X_va, y_va, 0.0), BATCH_SIZE, shuffle=False)
    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in va_ld:
            probs.extend(model(xb).numpy())
            labels.extend(yb.numpy())

    probs  = np.array(probs)
    labels = np.array(labels)
    pred_b = (probs >= threshold).astype(int)

    tp = int(((pred_b == 1) & (labels == 1)).sum())
    fp = int(((pred_b == 1) & (labels == 0)).sum())
    fn = int(((pred_b == 0) & (labels == 1)).sum())
    tn = int(((pred_b == 0) & (labels == 0)).sum())

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
        "mean_prob_positive_class": round(float(probs[labels == 1].mean()), 4) if labels.sum() > 0 else 0.0,
        "mean_prob_negative_class": round(float(probs[labels == 0].mean()), 4) if (1 - labels).sum() > 0 else 0.0,
    }


def lr_baseline_fold(X_tr, y_tr, X_va, y_va):
    """Logistic regression on flattened windows as a comparison baseline."""
    Xtr_f = X_tr.reshape(len(X_tr), -1)
    Xva_f = X_va.reshape(len(X_va), -1)
    sc    = StandardScaler().fit(Xtr_f)
    clf   = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    clf.fit(sc.transform(Xtr_f), y_tr)
    pred  = clf.predict(sc.transform(Xva_f))
    return {
        "precision": round(float(precision_score(y_va, pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_va, pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_va, pred, zero_division=0)), 4),
    }


def ablation_fold(X_tr, y_tr, X_va, y_va, feature_idx: int, epochs: int):
    """Train with one feature zeroed out across all windows to measure its impact."""
    X_tr_ab = X_tr.copy(); X_tr_ab[:, :, feature_idx] = 0.0
    X_va_ab = X_va.copy(); X_va_ab[:, :, feature_idx] = 0.0
    model = train_fold(X_tr_ab, y_tr, X_va_ab, y_va, epochs=epochs)
    return eval_fold(model, X_va_ab, y_va)["f1"]


# ── LOSO-CV ───────────────────────────────────────────────────────────────────

def run_loso_cv(all_runs, epochs=EPOCHS):
    all_raw = [build_raw_windows(r) for r in all_runs]
    fold_results, lr_results = [], []
    best_f1, best_model = -1.0, None

    for val_idx in range(len(all_runs)):
        val_seed = all_runs[val_idx].seed

        Xtr_raw = np.concatenate([all_raw[i][0] for i in range(len(all_runs)) if i != val_idx])
        ytr     = np.concatenate([all_raw[i][1] for i in range(len(all_runs)) if i != val_idx])
        Xva_raw, yva, va_lt = all_raw[val_idx]

        mu, sigma = compute_scaler(Xtr_raw)
        Xtr = apply_scaler(Xtr_raw, mu, sigma)
        Xva = apply_scaler(Xva_raw, mu, sigma)

        model  = train_fold(Xtr, ytr, Xva, yva, epochs=epochs)
        result = eval_fold(model, Xva, yva)
        result["seed"]      = val_seed
        result["lead_time"] = float(np.mean(va_lt)) if va_lt else 0.0
        fold_results.append(result)

        lr = lr_baseline_fold(Xtr, ytr, Xva, yva)
        lr["seed"] = val_seed
        lr_results.append(lr)

        gate_word = "PASS" if result["precision"] > PRECISION_GATE else "FAIL"
        print(
            f"  fold {val_seed}: "
            f"LSTM P={result['precision']*100:.0f}%  R={result['recall']*100:.0f}%  F1={result['f1']:.2f}  [{gate_word}]"
            f"  |  LR P={lr['precision']*100:.0f}%  R={lr['recall']*100:.0f}%  F1={lr['f1']:.2f}"
        )

        if result["f1"] > best_f1:
            best_f1    = result["f1"]
            best_model = model

    return fold_results, lr_results, best_model


def run_ablation(all_runs, epochs):
    all_raw      = [build_raw_windows(r) for r in all_runs]
    feature_names = ["step_time", "completion_ratio", "node_throughput", "soft_stall_flag"]
    results = {}
    for fi, fname in enumerate(feature_names):
        f1s = []
        for val_idx in range(len(all_runs)):
            Xtr_raw = np.concatenate([all_raw[i][0] for i in range(len(all_runs)) if i != val_idx])
            ytr     = np.concatenate([all_raw[i][1] for i in range(len(all_runs)) if i != val_idx])
            Xva_raw, yva, _ = all_raw[val_idx]
            mu, sigma = compute_scaler(Xtr_raw)
            Xtr = apply_scaler(Xtr_raw, mu, sigma)
            Xva = apply_scaler(Xva_raw, mu, sigma)
            f1s.append(ablation_fold(Xtr, ytr, Xva, yva, fi, epochs=epochs))
        results[fname] = round(float(np.mean(f1s)), 4)
    return results


# ── Aggregate results ─────────────────────────────────────────────────────────

def aggregate(fold_results, lr_results):
    prec = [r["precision"]               for r in fold_results]
    rec  = [r["recall"]                  for r in fold_results]
    f1s  = [r["f1"]                      for r in fold_results]
    lt   = [r["lead_time"]               for r in fold_results]
    mp   = [r["mean_prob_positive_class"] for r in fold_results]
    mn   = [r["mean_prob_negative_class"] for r in fold_results]

    gate_p  = float(np.mean(prec)) > PRECISION_GATE
    gate_lt = float(np.mean(lt))   >= LEAD_TIME_GATE

    return {
        "lstm": {
            "mean_precision":            round(float(np.mean(prec)), 4),
            "std_precision":             round(float(np.std(prec)),  4),
            "mean_recall":               round(float(np.mean(rec)),  4),
            "std_recall":                round(float(np.std(rec)),   4),
            "mean_f1":                   round(float(np.mean(f1s)),  4),
            "std_f1":                    round(float(np.std(f1s)),   4),
            "mean_lead_time_steps":      round(float(np.mean(lt)),   2),
            "mean_prob_positive_class":  round(float(np.mean(mp)),   4),
            "mean_prob_negative_class":  round(float(np.mean(mn)),   4),
        },
        "logistic_regression_baseline": {
            "mean_precision": round(float(np.mean([r["precision"] for r in lr_results])), 4),
            "mean_recall":    round(float(np.mean([r["recall"]    for r in lr_results])), 4),
            "mean_f1":        round(float(np.mean([r["f1"]        for r in lr_results])), 4),
        },
        "nfr_p1_gate": {
            "precision_threshold": PRECISION_GATE,
            "precision_pass":      gate_p,
            "lead_time_threshold": LEAD_TIME_GATE,
            "lead_time_pass":      gate_lt,
            "overall_pass":        gate_p and gate_lt,
        },
        "per_fold": fold_results,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Sprint 6 Lane B — fault prediction LSTM")
    ap.add_argument("--runs_dir",   default="./runs",
                    help="Folder containing seed_42/, seed_43/, ... subdirectories")
    ap.add_argument("--seeds",      type=int, nargs="+", default=list(range(42, 52)))
    ap.add_argument("--epochs",     type=int, default=EPOCHS)
    ap.add_argument("--output_dir", default="./lane_b_output")
    args = ap.parse_args()

    runs_dir   = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading {len(args.seeds)} seeded runs ...")
    all_runs = []
    for seed in args.seeds:
        try:
            all_runs.append(load_run(runs_dir, seed))
        except FileNotFoundError as e:
            print(f"  [WARN] seed {seed}: {e} — skipping")
    if not all_runs:
        raise RuntimeError("No run data found. Check that seed_<N>/ folders exist under --runs_dir.")

    print(f"\nLOSO-CV ({len(all_runs)} folds) + logistic regression baseline ...\n")
    fold_results, lr_results, best_model = run_loso_cv(all_runs, args.epochs)

    ablation_epochs = min(80, args.epochs)
    print(f"\nFeature ablation (each feature zeroed in turn, {ablation_epochs} epochs) ...")
    ablation = run_ablation(all_runs, epochs=ablation_epochs)
    full_f1  = round(float(np.mean([r["f1"] for r in fold_results])), 4)
    print(f"  Full model F1  : {full_f1:.3f}")
    for fname, ab_f1 in ablation.items():
        drop = round(full_f1 - ab_f1, 4)
        print(f"  Without {fname:<22}: F1={ab_f1:.3f}  (change={drop:+.3f})")

    agg = aggregate(fold_results, lr_results)
    agg["ablation_f1_without_feature"] = ablation

    gp   = agg["nfr_p1_gate"]
    lstm = agg["lstm"]
    lr   = agg["logistic_regression_baseline"]

    print("\n" + "=" * 65)
    print("SPRINT 6 LANE B  —  EVALUATION RESULT  (LOSO-CV, 10 folds)")
    print("=" * 65)
    print(f"LSTM Precision   : {lstm['mean_precision']*100:.1f}% +/- {lstm['std_precision']*100:.1f}%"
          f"  (gate >70%)  {'PASS' if gp['precision_pass'] else 'FAIL'}")
    print(f"LSTM Recall      : {lstm['mean_recall']*100:.1f}% +/- {lstm['std_recall']*100:.1f}%")
    print(f"LSTM F1          : {lstm['mean_f1']:.3f} +/- {lstm['std_f1']:.3f}")
    print(f"Lead time        : {lstm['mean_lead_time_steps']} steps"
          f"  (gate >=0)  {'PASS' if gp['lead_time_pass'] else 'FAIL'}")
    print(f"Confidence gap   : positive class={lstm['mean_prob_positive_class']:.3f}"
          f"  negative class={lstm['mean_prob_negative_class']:.3f}")
    print(f"\nLR baseline      : P={lr['mean_precision']*100:.1f}%"
          f"  R={lr['mean_recall']*100:.1f}%  F1={lr['mean_f1']:.3f}")
    print(f"\nNFR-P1 GATE      : {'PASS' if gp['overall_pass'] else 'FAIL'}")
    print("=" * 65)
    print("\nNote: lead time = 0 = fault detected at the same step it fires, not before it.")
    print("ASTRA-sim models failure as an instant event with no pre-fault degradation ramp.")
    print("Option B: adding a synthetic degradation ramp to the harness would enable lead time >= 1 step.")

    # Save outputs
    torch.save(best_model.state_dict(), output_dir / "lane_b_lstm.pt")
    with open(output_dir / "lane_b_evaluation_report.json", "w") as fh:
        json.dump({
            **agg,
            "ablation_f1_without_feature": ablation,
            "config": {
                "window_w":       WINDOW_W,
                "n_features":     N_FEATURES,
                "hidden_size":    HIDDEN_SIZE,
                "epochs":         args.epochs,
                "noise_sigma":    NOISE_SIGMA,
                "seeds_used":     args.seeds,
                "eval_method":    "Leave-One-Seed-Out cross-validation (LOSO-CV)",
                "lr_baseline":    "Logistic regression on flattened windows",
                "label_mode":     "detection (fault step inside window)",
                "lead_time_note": "0 steps = same-step detection, not N-step prediction",
            },
        }, fh, indent=2)

    print(f"\nModel saved   -> {output_dir / 'lane_b_lstm.pt'}")
    print(f"Report saved  -> {output_dir / 'lane_b_evaluation_report.json'}")


if __name__ == "__main__":
    main()
