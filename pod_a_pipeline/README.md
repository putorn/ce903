# pod_a_pipeline

Provides the burst interval extractor used by `tc04/train_lstm.py` and `tc04/generate_routing.py`, plus a script to generate synthetic `.et` workload files when real Chakra traces aren't available.

## Quick start

Run this once before anything in tc04:

```bash
python pod_a_pipeline/make_synthetic_workloads.py
```

That creates the `.et` files in `pod_a_pipeline/workloads/` and `pod_a_pipeline/workloads_failure/`. Then run tc04 as normal:

```bash
python tc04/train_lstm.py
python tc04/generate_routing.py
```

## What's in here

- `burst_extractor.py` — parses `.et` files and returns inter-burst interval series. Tries the Chakra protobuf parser first; falls back to numpy format if Chakra isn't installed.
- `make_synthetic_workloads.py` — generates AR(1) synthetic traces matching the signal characteristics from Kyle's feasibility report (CV < 5%, φ = 0.90). Failure traces include the TC-03 stall pattern (1,031,879 cycle transient, 4→3 NPU mean shift).

## Notes

- `.et` files are in `.gitignore` — regenerate them locally with `make_synthetic_workloads.py`, don't commit them.
- The trained model (`lstm_burst_predictor.pt`) is an output of `train_lstm.py`, not a dependency — it will be created on first run.
