#!/bin/bash
# RunPod Setup — A100 GPU
# Run this once after starting the pod before anything else.
# Installs all dependencies and clones astra-sim for Chakra tools.
set -e

echo "[setup] ── Installing PyTorch 2.1.2 + CUDA 12.1 ─────────────────────────"
pip install torch==2.1.2+cu121 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "[setup] ── Installing transformers ──────────────────────────────────────"
pip install transformers

echo ""
echo "[setup] ── Installing et_replay (Chakra host trace reader) ──────────────"
pip install "git+https://github.com/facebookresearch/param.git#subdirectory=train/compute/python"

echo ""
echo "[setup] ── Installing HolisticTraceAnalysis (hta) ───────────────────────"
pip install HolisticTraceAnalysis

echo ""
echo "[setup] ── Cloning ASTRA-sim (for Chakra converter tools) ───────────────"
# Only the graph_frontend submodule is needed for conversion — full build not required.
if [ ! -d "astra-sim" ]; then
    git clone --recurse-submodules https://github.com/astra-sim/astra-sim.git
else
    echo "[setup] astra-sim already exists — skipping clone."
fi

echo ""
echo "[setup] ── Patching Chakra schema version list ───────────────────────────"
# PyTorch 2.1.2 generates schema 1.1.1-chakra.0.0.4 which is newer than
# what the bundled Chakra 0.0.4 converter supports. This patch adds it.
python runpod/roshan_converter/patch_chakra.py

echo ""
echo "[setup] ── Verifying GPU ────────────────────────────────────────────────"
python -c "
import torch
print(f'  CUDA available : {torch.cuda.is_available()}')
print(f'  GPU            : {torch.cuda.get_device_name(0)}')
print(f'  VRAM           : {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB')
print(f'  PyTorch        : {torch.__version__}')
"

echo ""
echo "[setup] Done. Run in order:"
echo "  1. python runpod/kyle_profiler/run_profiler.py"
echo "  2. bash   runpod/roshan_converter/convert.sh"
echo "  3. Download workloads/gpt2.*.et → pod_a_pipeline/workloads/"
