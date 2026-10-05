#!/usr/bin/env bash
# Serve the base model + your LoRA adapter with vLLM (needs an NVIDIA GPU, e.g. Colab T4 or a cloud GPU).
# The server speaks the OpenAI-compatible chat API on port 8000; predict_local.py --backend vllm calls it.
#
#   pip install vllm
#   bash serve_vllm.sh checkpoints/qwen-1.5b-lora
set -euo pipefail

ADAPTER="${1:-checkpoints/qwen-1.5b-lora}"
BASE="${BASE:-Qwen/Qwen2.5-1.5B-Instruct}"

vllm serve "$BASE" \
  --enable-lora \
  --lora-modules clinical-lora="$ADAPTER" \
  --max-model-len 2048 \
  --dtype half \
  --port 8000
