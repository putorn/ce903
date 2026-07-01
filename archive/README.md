archive/

Earlier, standalone versions of scripts that were superseded during the project as the pipeline was consolidated. They are kept here for provenance rather than deleted.

burst_extractor.py — early root-level burst extractor. Superseded by pod_a_pipeline/burst_extractor.py (which exposes extract_intervals).
LSTMburst_timing_training_script.py — early burst-timing LSTM trainer. Superseded by tc04/train_lstm.py (seeded, normalised, horizon-3).
LTSM Predictor prototype.py — early Lane A burst-timing LSTM prototype (reads analysis/bursts_4npu_256MB.csv). Superseded by tc04/train_lstm.py.
predict_and_generate_routing.py — early routing generator. Superseded by tc04/generate_routing.py.
run_tc04_experiment.sh — early TC-04 wrapper script, superseded by tc04/run_tc04.sh. Calls LSTMburst_timing_training_script.py and predict_and_generate_routing.py by their old root-level paths, so it will not run as-is — kept for provenance only, not a working script.

The live pipeline lives in pod_a_pipeline/ and tc04/. Nothing in the project depends on the files in this folder.
