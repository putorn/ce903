# FR-G1 — Topology Comparison: Results & Write-up

## What this closes

FR-G1 (SRS): "The system shall model a network topology capable of replicating massive
LLM traffic, inspired by Alibaba's HPN dual-ToR design, to assess path search complexity
and fault redundancy." ASTRA-sim's analytical congestion-aware backend only supports
single-dimension topologies, so the two-tier spine-leaf design (`run_frg1.sh`) can't run
on it directly. This is demonstrated instead with a single-tier switch fabric
(`run_frg1_fallback.sh`) against the ring baseline, at matched NPU count (16).

## Experiment

Identical 16-NPU GPT-2 124M workload across three runs; only the network changes:

| Run | Network config | Purpose |
|---|---|---|
| Ring (16) | `Ring_16npus.yml` | topology baseline (ring saturates at 16 NPUs — TC-02) |
| Switch fabric, healthy | `switch_16.yml` | every NPU one hop via the shared switch |
| Switch fabric, degraded | `switch_16_fail.yml` | fault proxy — switch bandwidth halved |

Run with `bash frg1/run_frg1_fallback.sh` from the repo root.

(`run_frg1.sh` — the full two-tier spine-leaf version — is kept alongside this for the
ns-3 extension; it will not run on the analytical backend used here.)

## Results

| Metric | Value |
|---|---|
| Ring (16) JCT | 99,937,920 cycles |
| Switch fabric, healthy JCT | 118,927,680 cycles |
| Switch fabric, degraded JCT | 156,428,160 cycles |
| Topology benefit (switch fabric vs ring) | +19% (worse — ring wins) |
| Degradation penalty (switch bandwidth halved) | +31.5% vs healthy switch fabric |

## Path-search complexity (analytical)

- **Switch fabric (16 NPUs):** every NPU pair is one hop via the shared switch — a
  single path, not a redundant one. There is no second switch, so the "degraded" run
  models a bandwidth-halving proxy for partial failure, not a reroute onto a surviving
  equal-cost path.
- **Ring (16 NPUs):** a single path, up to N−1 = 15 hops worst case (~8 average), with
  bisection bandwidth that saturates at 16 NPUs (TC-02 finding).
- The switch fabric therefore offers a shorter worst-case path than the ring, but not
  the equal-cost path *redundancy* that a true two-tier spine-leaf (two spines) would —
  that property is untested here and is exactly what the ns-3 extension would add.

## Scope-boundary paragraph (paste-ready for the report)

The switch fabric realised here demonstrates one FR-G1 property — shorter worst-case
path length than the ring — using ASTRA-sim's analytical congestion-aware backend, which
only supports single-dimension topologies. It does not demonstrate true path redundancy
(equal-cost multi-path routing across two spines), rack-level locality, dual-ToR
intra-rack redundancy, or packet-level ECMP hashing — all of which would require the
ns-3 backend and the full two-tier spine-leaf design in `run_frg1.sh`. The ring
outperforming the switch fabric at this scale (+19%) is a confirmation that ring
AllReduce remains bandwidth-optimal for this workload, not a failure of the topology
design. The switch-degradation test (bandwidth halved) is a proxy for partial failure,
standing in for the redundancy test a second spine would allow.

## Future work (ns-3, paste-ready)

The natural higher-fidelity extension rebuilds the full two-tier spine-leaf topology
(`run_frg1.sh`'s configs) on ASTRA-sim's ns-3 backend, which models real switches, links,
and ECMP hashing across multiple dimensions. There, a spine switch can be failed directly
and traffic observed rerouting across the surviving spine — the true path-redundancy test
that the single-tier fallback used here cannot perform.
