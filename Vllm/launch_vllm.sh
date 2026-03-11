#!/usr/bin/env bash
# =============================================================================
# launch_vllm.sh — Start the Qwen3.5-9B vLLM OpenAI-compatible server.
# Run this BEFORE starting the FastAPI service (main.py).
#
# VRAM requirements (BF16):
#   24 GB GPU (A10G / RTX 4090) → use MAX_MODEL_LEN=8192   (safe)
#   40 GB GPU (A100 40GB)       → use MAX_MODEL_LEN=32768  (comfortable)
#   80 GB GPU (A100 80GB/H100)  → use MAX_MODEL_LEN=32768  (no constraints)
#
# For photo validation each prompt has 1 image + ~1400 token prompt + ~2048
# token output, so 8192 context is more than sufficient.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if present
[ -f .venv/bin/activate ] && source .venv/bin/activate

# Use local weights if available, otherwise HuggingFace hub ID
MODEL_PATH="./models/Qwen3.5-9B"
if [ ! -d "$MODEL_PATH" ] || [ -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]; then
    MODEL_PATH="Qwen/Qwen3.5-9B"
fi

# Auto-detect available VRAM and pick a safe max_model_len
VRAM_GB=$(python3 - <<'PYEOF' 2>/dev/null || echo "0"
import subprocess, re
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
    text=True
).strip().split("\n")[0]
print(int(out) // 1024)
PYEOF
)

if [ -z "${MAX_MODEL_LEN:-}" ]; then
    # Photo validation needs ~2000 tokens max — 4096 is sufficient for all GPUs.
    # Keeping this low saves ~17 GB of KV cache VRAM vs 32768.
    MAX_MODEL_LEN=4096
fi

echo "[$(date +'%H:%M:%S')] GPU VRAM detected: ${VRAM_GB} GB — using max_model_len=${MAX_MODEL_LEN}"
echo "[$(date +'%H:%M:%S')] Starting vLLM server with model: $MODEL_PATH"

vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT:-8000}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
    --dtype "${VLLM_DTYPE:-auto}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
    --served-model-name "Qwen/Qwen3.5-9B"
