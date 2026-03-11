import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_NAME = os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-9B")
MODEL_LOCAL_PATH = os.getenv("MODEL_LOCAL_PATH", str(BASE_DIR / "models" / "Qwen3.5-9B"))
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "true").lower() == "true"


def get_model_path() -> str:
    """Return local path if weights exist, else HuggingFace model ID."""
    local = Path(MODEL_LOCAL_PATH)
    if USE_LOCAL_MODEL and local.exists() and any(local.iterdir()):
        return str(local)
    return MODEL_NAME


# ---------------------------------------------------------------------------
# vLLM OpenAI-compatible server (separate process)
# ---------------------------------------------------------------------------
# URL of the running vLLM serve instance.
# Default: localhost:8000 (started by setup.sh / run manually).
VLLM_SERVER_URL = os.getenv("VLLM_SERVER_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "not-needed")


# ---------------------------------------------------------------------------
# vLLM serve launch parameters (used by setup.sh / launch_vllm.sh)
# ---------------------------------------------------------------------------
# Photo validation needs ~2000 tokens max (image + prompt + output).
# 4096 is sufficient and reduces KV cache VRAM by ~17 GB vs 32768.
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "4096"))
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.60"))
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
DTYPE = os.getenv("VLLM_DTYPE", "auto")
ENFORCE_EAGER = os.getenv("ENFORCE_EAGER", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.05"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "2048"))
TOP_P = float(os.getenv("TOP_P", "0.9"))

# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))

# ---------------------------------------------------------------------------
# Validation limits
# ---------------------------------------------------------------------------
MAX_PHOTOS = 10
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
