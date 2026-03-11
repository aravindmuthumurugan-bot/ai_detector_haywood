#!/usr/bin/env bash
# =============================================================================
# setup.sh — Qwen3.5-9B vLLM Photo Validation Service
# Installs all GPU dependencies (CUDA 12.4 / cuDNN) and downloads model weights.
# Run once before starting the service.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }

log "======================================================"
log " Qwen3.5-9B vLLM Photo Validation Service — Setup"
log "======================================================"

# -----------------------------------------------------------------------------
# 1. GPU check
# -----------------------------------------------------------------------------
log "Checking for NVIDIA GPU..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. An NVIDIA GPU with drivers is required."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
log "GPU detected."

# -----------------------------------------------------------------------------
# 2. System dependencies
# -----------------------------------------------------------------------------
log "Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    python3-pip python3-dev python3-venv \
    git wget curl \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    ffmpeg  # needed by some torch vision ops

# -----------------------------------------------------------------------------
# 3. Python virtual environment (optional but recommended)
# -----------------------------------------------------------------------------
if [ ! -d ".venv" ]; then
    log "Creating Python virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate
log "Using Python: $(python --version)"

# -----------------------------------------------------------------------------
# 4. Upgrade pip and base tools
# -----------------------------------------------------------------------------
pip install --upgrade pip setuptools wheel --quiet

# -----------------------------------------------------------------------------
# 5. Install PyTorch with CUDA 12.4 (matches existing infra: CUDA 12.4.1 + cuDNN)
# -----------------------------------------------------------------------------
log "Installing PyTorch 2.4 with CUDA 12.4..."
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124 --quiet

# Verify CUDA
python - <<'PYEOF'
import torch
print(f"PyTorch {torch.__version__}  |  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
else:
    print("WARNING: CUDA not available — vLLM requires GPU")
PYEOF

# -----------------------------------------------------------------------------
# 6. Install vLLM (CUDA 12.4 compatible — vLLM >=0.6.3 ships cu124 wheels)
# -----------------------------------------------------------------------------
log "Installing vLLM..."
pip install "vllm>=0.6.3" --quiet

# -----------------------------------------------------------------------------
# 7. Install remaining Python dependencies
# -----------------------------------------------------------------------------
log "Installing service dependencies..."
pip install -r requirements.txt --quiet

# -----------------------------------------------------------------------------
# 8. Download Qwen3.5-9B model weights
# -----------------------------------------------------------------------------
MODEL_ID="Qwen/Qwen3.5-9B"
MODEL_DIR="./models/Qwen3.5-9B"

log "Checking model weights at ${MODEL_DIR}..."
mkdir -p models

python - <<PYEOF
import os, sys
from pathlib import Path

model_dir = Path("${MODEL_DIR}")
existing = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
if existing:
    print(f"  Model already present ({len(existing)} weight files). Skipping download.")
    sys.exit(0)

print(f"  Downloading Qwen/Qwen3.5-9B (~18 GB, this may take a while)...")
try:
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="Qwen/Qwen3.5-9B",
        local_dir="${MODEL_DIR}",
        ignore_patterns=["*.msgpack", "flax*", "tf_*", "rust_*"],
    )
    print("  Download complete!")
except Exception as e:
    print(f"  ERROR downloading model: {e}")
    print("  Try manually: huggingface-cli download Qwen/Qwen3.5-9B --local-dir ${MODEL_DIR}")
    sys.exit(1)
PYEOF

# -----------------------------------------------------------------------------
# 9. Quick smoke test
# -----------------------------------------------------------------------------
log "Running quick import check..."
python - <<'PYEOF'
import fastapi, uvicorn, PIL, pydantic, openai
print("  All imports OK.")
PYEOF

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
log ""
log "======================================================"
log " Setup complete!"
log "======================================================"
log ""
log " ┌─ STEP 1: Start the vLLM inference server"
log " │   source .venv/bin/activate"
log " │   vllm serve ./models/Qwen3.5-9B \\"
log " │       --host 0.0.0.0 --port 8000 \\"
log " │       --max-model-len 32768 \\"
log " │       --gpu-memory-utilization 0.85 \\"
log " │       --dtype auto"
log " │"
log " │   # Or use the helper script:"
log " │   bash launch_vllm.sh"
log " │"
log " ├─ STEP 2: Start the FastAPI validation service (new terminal)"
log " │   source .venv/bin/activate"
log " │   uvicorn main:app --host 0.0.0.0 --port 8001 --workers 1"
log " │"
log " ├─ Health checks"
log " │   curl http://localhost:8000/health   # vLLM server"
log " │   curl http://localhost:8001/health   # validation API"
log " │"
log " └─ API docs"
log "     http://localhost:8001/docs"
log ""
