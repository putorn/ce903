#!/bin/bash
# run_regression_sweep.sh — one-command TC-01–04 regression sweep. Run from repo root.
RESULTS=()
section() { echo; echo "══════════════════════════════════════"; echo " $1"; echo "══════════════════════════════════════"; }

section "TC-01 — Trace-to-Sim Pipeline"
(cd pod_a_pipeline && bash run_pipeline.sh) && RESULTS+=("TC-01: PASS") || RESULTS+=("TC-01: FAIL — see output above")

section "TC-02 — Ring AllReduce Baseline Sweep"
(cd pod_b_traffic && bash generate_workloads.sh && bash run_baseline.sh) && RESULTS+=("TC-02: PASS") || RESULTS+=("TC-02: FAIL — see output above")

section "TC-03 — Fault Injection & JCT Recovery Penalty"
(cd pod_a_pipeline && conda run -n p903 python run_fault_injection.py && conda run -n p903 python recovery_metrics.py) && RESULTS+=("TC-03: PASS") || RESULTS+=("TC-03: FAIL — see output above")

section "TC-04 — Predictive Routing vs ECMP Baseline"
(python pod_a_pipeline/make_synthetic_workloads.py && python tc04/train_lstm.py && python tc04/generate_routing.py && bash tc04/run_tc04.sh) && RESULTS+=("TC-04: PASS — copy the two JCT numbers above into tc04/TC04_results.md") || RESULTS+=("TC-04: FAIL — see output above")

section "Regression Sweep Summary"
printf '%s\n' "${RESULTS[@]}"
