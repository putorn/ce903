# archive/

Earlier, standalone versions of scripts that were superseded during the project as
the pipeline was consolidated. They are kept here for provenance rather than deleted.

- `burst_extractor.py` — early root-level burst extractor. Superseded by
  `pod_a_pipeline/burst_extractor.py` (which exposes `extract_intervals`).
- `LSTMburst_timing_training_script.py` — early burst-timing LSTM trainer.
  Superseded by `tc04/train_lstm.py` (seeded, normalised, horizon-3).
- `predict_and_generate_routing.py` — early routing generator. Superseded by
  `tc04/generate_routing.py`.

The live pipeline lives in `pod_a_pipeline/` and `tc04/`. Nothing in the project
depends on the files in this folder.
