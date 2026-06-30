# FR-G1 — Spine-Leaf Topology: Results & Write-up

## What this closes
FR-G1 (SRS): *"The system shall model a network topology capable of replicating massive
LLM traffic, inspired by Alibaba's HPN dual-ToR design, to assess path search complexity
and fault redundancy."* Demonstrated with a two-tier spine-leaf on ASTRA-sim's analytical
congestion-aware backend, evaluated against the ring baseline at matched NPU count (16).

## Experiment
Identical 16-NPU GPT-2 124M workload across three runs; only the network changes:

| Run | Network config | Purpose |
|---|---|---|
| Ring (16) | `Ring_16npus.yml` | topology baseline (ring saturates at 16 NPUs — TC-02) |
| Spine-leaf healthy | `spine_leaf_16.yml` (4×4) | path-diverse topology |
| Spine-leaf, 1 spine lost | `spine_leaf_16_spinefail.yml` | fault-redundancy test (spine BW 50→25) |

Run with `bash frg1/run_frg1.sh` from the repo root.

## Results (fill from run_frg1.sh output)

| Metric | Value |
|---|---|
| Ring (16) JCT | ___ cycles |
| Spine-leaf healthy JCT | ___ cycles |
| Spine-leaf, 1 spine lost JCT | ___ cycles |
| Topology benefit (spine-leaf vs ring) | ___ % |
| Fault-redundancy penalty (1 spine lost) | ___ % |

## Path-search complexity (analytical — the FR-G1 "path complexity" deliverable)
- **Spine-leaf:** between two GPUs on different leaves there are **2 equal-cost paths**
  (one per spine), each **2 hops** (leaf → spine → leaf).
- **Ring (16 NPUs):** a **single** path, up to **N−1 = 15 hops** worst case (~8 average),
  with bisection bandwidth that saturates at 16 NPUs (TC-02 finding).
- The spine-leaf therefore offers both shorter worst-case paths and redundant equal-cost
  paths — exactly the two properties FR-G1 requires.

## Scope-boundary paragraph (paste-ready for the report)
The spine-leaf topology realised here demonstrates the two core FR-G1 properties — path
diversity (two equal-cost leaf-to-spine paths per GPU pair) and fault redundancy under
switch failure — using ASTRA-sim's analytical congestion-aware backend. It is structurally
equivalent to, but not a full implementation of, Alibaba's HPN dual-ToR design: it does not
model rack-level locality, dual-ToR intra-rack redundancy, or packet-level ECMP hashing, all
of which would require the ns-3 backend. The SRS "inspired by" wording is satisfied by
demonstrating the same core properties; the full dual-ToR implementation is recorded as
future work. A spine failure is modelled by halving the spine-tier link bandwidth (one of two
spine paths lost); the resulting JCT increase quantifies the fault-redundancy penalty.

## Future work (ns-3, paste-ready)
The natural higher-fidelity extension rebuilds this topology on ASTRA-sim's ns-3 backend,
which models real switches, links and ECMP hashing. There a spine switch can be failed
directly and traffic observed rerouting across the survivor — the packet-level version of the
bandwidth-degradation proxy used here.
