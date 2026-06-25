#!/bin/bash


set -e

echo "=== TC-04: Training LSTM ==="
python LTSMburst_timing_training_script.py

echo "=== TC-04: Generating routing config ==="
python predict_and_generate_routing.py

echo "=== TC-04: Running ECMP baseline (TC-03, seed 42) ==="
python pod_a_pipeline/run_fault_injection.py --seed 42 --routing ecmp

echo "=== TC-04: Running predictor-informed routing (TC-03, seed 42) ==="
python pod_a_pipeline/run_fault_injection.py --seed 42 --routing predictor

echo "=== TC-04 complete. Extract median JCT from: pod_a_pipeline/results_fault_run/ ==="
