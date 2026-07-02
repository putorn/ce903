# TC-04 — Predictive Routing vs ECMP: Results

Run with `bash tc04/run_tc04.sh` (or via `run_regression_sweep.sh`) from the repo root.

| Metric | Value |
|---|---|
| ECMP JCT | 94,012,616 cycles |
| Predictor JCT | 93,333,832 cycles |
| Delta (predictor − ECMP) | −678,784 cycles (−0.72%) |
| Date run | 2026-07-02 |
| Run by | Tae |

Numbers match the report exactly (ECMP 94,012,616 / predictor 93,333,832 / −0.72%).
