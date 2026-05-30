"""
Pod A — Synthetic GPT-2 124M Training Step Trace Generator

Creates one Chakra .et file per NPU that represents a single GPT-2 training step:
  Node 0  gpt2_forward_pass    COMP_NODE      25,000 μs  (25 ms)
  Node 1  gpt2_backward_pass   COMP_NODE      50,000 μs  (50 ms)  ← waits for node 0
  Node 2  gradient_allreduce   COMM_COLL_NODE 512 MB FP32           ← waits for node 1
  Node 3  optimizer_step       COMP_NODE       5,000 μs  ( 5 ms)  ← waits for node 2

Compute timing rationale:
  - GPT-2 124M steady-state step time on a single GPU is typically 80–120 ms.
  - Backward pass is approximately 2× the forward pass (standard for autograd).
  - Values here are representative for analytical simulation — actual profiler
    timings from Kyle's pipeline (Pod A integration) will replace these later.

AllReduce size rationale (from Sprint 3 collective spec):
  - GPT-2 124M has 124,439,808 parameters.
  - FP32 gradient sync: 124,439,808 × 4 bytes = 497,759,232 bytes ≈ 512 MB.
  - This is the confirmed analytical synthesis value (no live GPU cluster required).

Each NPU receives the same logical graph. ASTRA-sim loads one file per NPU:
  workloads/gpt2_step.0.et, workloads/gpt2_step.1.et, ..., gpt2_step.N-1.et

Usage:
  conda run -n p903 python generate_synthetic_trace.py --npus 4 --output workloads
"""

import argparse
import os
import sys


def build_trace(astra_sim_dir: str, npus: int, output_dir: str) -> None:
    # Add astra-sim to path so Chakra's internal imports resolve.
    sys.path.insert(0, astra_sim_dir)

    from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import (
        GlobalMetadata, COMP_NODE, COMM_COLL_NODE, ALL_REDUCE,
    )
    from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import (
        AttributeProto as ChakraAttr,
        Node as ChakraNode,
    )
    from extern.graph_frontend.chakra.src.third_party.utils.protolib import (
        encodeMessage as encode_message,
    )

    os.makedirs(output_dir, exist_ok=True)

    # AllReduce size: GPT-2 124M FP32 gradients (124,439,808 params × 4 bytes).
    ALLREDUCE_BYTES = 512 * 1024 * 1024  # 536,870,912 bytes

    # Compute timings in microseconds (μs). These match a representative steady-state
    # GPT-2 124M step; backward is ~2× forward as expected for autograd.
    FORWARD_MICROS  = 25_000  #  25 ms
    BACKWARD_MICROS = 50_000  #  50 ms
    OPTIMIZER_MICROS = 5_000  #   5 ms

    for npu_id in range(npus):
        filepath = os.path.join(output_dir, f"gpt2_step.{npu_id}.et")

        with open(filepath, "wb") as et:
            # File header: Chakra schema version.
            encode_message(et, GlobalMetadata(version="0.0.4"))

            # ── Node 0: Forward pass ──────────────────────────────────────────
            # Simulates the full GPT-2 forward computation (embedding + 12× transformer).
            # No data dependencies — this is the first op in each training step.
            node = ChakraNode()
            node.id   = 0
            node.name = "gpt2_forward_pass"
            node.type = COMP_NODE
            node.duration_micros = FORWARD_MICROS
            node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
            encode_message(et, node)

            # ── Node 1: Backward pass ─────────────────────────────────────────
            # Autograd backward through all 12 transformer layers.
            # Must wait for forward to finish (data_deps = [0]).
            node = ChakraNode()
            node.id   = 1
            node.name = "gpt2_backward_pass"
            node.type = COMP_NODE
            node.duration_micros = BACKWARD_MICROS
            node.data_deps.append(0)  # depends on: forward pass
            node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
            encode_message(et, node)

            # ── Node 2: Gradient AllReduce ────────────────────────────────────
            # Synchronises gradients across all NPUs after the backward pass.
            # Size = 512 MB (GPT-2 124M FP32) from the Sprint 3 collective spec.
            # Must wait for backward to finish (data_deps = [1]).
            node = ChakraNode()
            node.id   = 2
            node.name = "gradient_allreduce_fp32"
            node.type = COMM_COLL_NODE
            node.data_deps.append(1)  # depends on: backward pass
            node.attr.append(ChakraAttr(name="is_cpu_op",  bool_val=False))
            node.attr.append(ChakraAttr(name="comm_type",  int64_val=ALL_REDUCE))
            node.attr.append(ChakraAttr(name="comm_size",  int64_val=ALLREDUCE_BYTES))
            encode_message(et, node)

            # ── Node 3: Optimizer step ────────────────────────────────────────
            # AdamW parameter update — can only run once gradients are synchronised.
            # Must wait for AllReduce to finish (data_deps = [2]).
            node = ChakraNode()
            node.id   = 3
            node.name = "optimizer_step"
            node.type = COMP_NODE
            node.duration_micros = OPTIMIZER_MICROS
            node.data_deps.append(2)  # depends on: gradient allreduce
            node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
            encode_message(et, node)

        print(f"  Generated: {filepath}")

    print(f"\n  Total: {npus} .et files in {output_dir}/")
    print(f"  Trace structure: forward({FORWARD_MICROS//1000}ms) → "
          f"backward({BACKWARD_MICROS//1000}ms) → "
          f"allreduce({ALLREDUCE_BYTES // (1024*1024)}MB FP32) → "
          f"optimizer({OPTIMIZER_MICROS//1000}ms)")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic GPT-2 Chakra trace")
    parser.add_argument("--npus",       type=int, default=4,
                        help="Number of NPUs (default: 4)")
    parser.add_argument("--output",     type=str, default="workloads",
                        help="Output directory for .et files (default: workloads/)")
    parser.add_argument("--astra-sim",  type=str,
                        default=os.path.join(os.path.dirname(__file__), "../astra-sim"),
                        help="Path to astra-sim root (for Chakra imports)")
    args = parser.parse_args()

    astra_sim_dir = os.path.realpath(args.astra_sim)
    output_dir    = os.path.realpath(args.output)

    print(f"[Pod A] Generating synthetic GPT-2 trace")
    print(f"        NPUs      : {args.npus}")
    print(f"        Output    : {output_dir}")
    print(f"        astra-sim : {astra_sim_dir}")
    print()

    build_trace(astra_sim_dir=astra_sim_dir, npus=args.npus, output_dir=output_dir)


if __name__ == "__main__":
    main()
