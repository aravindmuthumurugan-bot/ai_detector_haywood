"""
AI Image Detector — Pure ONNX Runtime GPU (CUDA)
No PyTorch / TensorFlow required at inference.

Usage:
    Step 1 — export model once:
        python gpu_test.py --export

    Step 2 — run detection:
        python gpu_test.py
"""

import os
import sys
import json
import site
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

# ─── CUDA lib discovery (must run before onnxruntime is imported) ─────────────
# Matches the LD_LIBRARY_PATH pattern from Dockerfile.base:
#   nvidia/cudnn/lib, nvidia/cublas/lib, nvidia/cuda_runtime/lib
def _setup_cuda_libs():
    nvidia_pkgs = ["nvidia/cudnn/lib", "nvidia/cublas/lib", "nvidia/cuda_runtime/lib"]
    extra = []
    for sp in site.getsitepackages():
        for pkg in nvidia_pkgs:
            p = os.path.join(sp, pkg)
            if os.path.isdir(p):
                extra.append(p)
    if extra:
        current = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(extra) + (":" + current if current else "")

_setup_cuda_libs()

import onnxruntime as ort

# ─── Configuration ────────────────────────────────────────────────────────────

ONNX_MODEL_DIR     = "./model_onnx"
MATRI_ID           = "M12345"
REJECTION_THRESHOLD = 0.65

IMAGE_PATHS = [
    "Ai_images/Saloni.png",
    "Ai_images/Varun.png",
    "Ai_images/Anjali.png",
    "Ai_images/Priya (2).png",
    "Ai_images/Amit with family.png",
    "Ai_images/Gemini_Generated_Image_oxh2e9oxh2e9oxh2.png",
    "Real_images/IMG_20220814_174830.jpg",
    "Real_images/CHR4226900_Mrl620_TB_4.jpg",
    "Real_images/image (2).png",
]

# ──────────────────────────────────────────────────────────────────────────────


def export_model():
    print("Exporting haywoodsloan/ai-image-detector-deploy to ONNX (via torch.onnx)...")
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError:
        print("ERROR: Run first → pip install transformers torch")
        sys.exit(1)

    MODEL_NAME = "haywoodsloan/ai-image-detector-deploy"
    os.makedirs(ONNX_MODEL_DIR, exist_ok=True)

    print("Loading model weights...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model     = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()

    # Save processor config and model config (needed at inference)
    processor.save_pretrained(ONNX_MODEL_DIR)
    model.config.to_json_file(os.path.join(ONNX_MODEL_DIR, "config.json"))

    # Dummy input — matches the center-crop size from preprocessor_config.json
    crop_size   = processor.crop_size.get("height", 224) if hasattr(processor, "crop_size") and isinstance(processor.crop_size, dict) else 224
    dummy_input = torch.randn(1, 3, crop_size, crop_size)

    onnx_path = os.path.join(ONNX_MODEL_DIR, "model.onnx")
    print(f"Exporting to {onnx_path} ...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "logits":       {0: "batch_size"},
        },
    )

    print(f"\n✓ Model exported to '{ONNX_MODEL_DIR}/'")
    print("  Now run: python gpu_test.py")


# ─── Preprocessing (pure numpy + PIL, no torch) ───────────────────────────────

def load_config():
    with open(os.path.join(ONNX_MODEL_DIR, "preprocessor_config.json")) as f:
        return json.load(f)


def preprocess(image_path: str, config: dict) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")

    # Resize size
    size = config.get("size", {})
    if isinstance(size, dict):
        resize_to = size.get("shortest_edge", size.get("height", 256))
    else:
        resize_to = int(size)

    # Crop size
    crop_cfg = config.get("crop_size", {})
    if isinstance(crop_cfg, dict):
        crop_size = crop_cfg.get("height", 224)
    else:
        crop_size = int(crop_cfg) if crop_cfg else 224

    mean = np.array(config.get("image_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std  = np.array(config.get("image_std",  [0.229, 0.224, 0.225]), dtype=np.float32)

    # Resize (shortest edge)
    w, h  = image.size
    scale = resize_to / min(w, h)
    image = image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

    # Center crop
    w, h = image.size
    left = (w - crop_size) // 2
    top  = (h - crop_size) // 2
    image = image.crop((left, top, left + crop_size, top + crop_size))

    # Normalize → CHW → batch dim
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, 0).astype(np.float32)


# ─── ONNX Runtime session (CUDA) ──────────────────────────────────────────────

def create_session() -> ort.InferenceSession:
    model_path = os.path.join(ONNX_MODEL_DIR, "model.onnx")

    cuda_options = {
        "device_id": 0,
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "DEFAULT",
    }

    session = ort.InferenceSession(
        model_path,
        providers=[("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"],
    )

    active = session.get_providers()[0]
    print(f"ONNX Runtime : {ort.__version__}")
    print(f"Provider     : {active}")

    if active != "CUDAExecutionProvider":
        print("WARNING: CUDA not available, running on CPU.")

    return session


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def get_id2label() -> dict:
    config_path = os.path.join(ONNX_MODEL_DIR, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    return cfg.get("id2label", {"0": "ai", "1": "real"})


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI Image Detector — ONNX Runtime GPU")
    parser.add_argument("--export", action="store_true", help="Export model to ONNX (one-time)")
    args = parser.parse_args()

    if args.export:
        export_model()
        return

    # Check model exists
    onnx_file = os.path.join(ONNX_MODEL_DIR, "model.onnx")
    if not os.path.exists(onnx_file):
        print("ONNX model not found. Run first:\n  python gpu_test.py --export")
        sys.exit(1)

    config   = load_config()
    id2label = get_id2label()
    session  = create_session()

    input_name = session.get_inputs()[0].name

    print(f"\nMatri ID : {MATRI_ID}")
    print("=" * 60)

    rejected = []

    for path in IMAGE_PATHS:
        filename = Path(path).name

        input_tensor = preprocess(path, config)
        logits       = session.run(None, {input_name: input_tensor})[0][0]
        probs        = softmax(logits)

        scores    = {id2label[str(i)]: float(probs[i]) for i in range(len(probs))}
        ai_prob   = scores.get("ai", scores.get("artificial", 0.0))
        real_prob = scores.get("real", scores.get("human", 1.0 - ai_prob))

        is_rejected = ai_prob >= REJECTION_THRESHOLD
        status = "REJECTED" if is_rejected else "ACCEPTED"

        print(f"File      : {filename}")
        print(f"Verdict   : {status}")
        print(f"AI Prob   : {ai_prob:.4f}")
        print(f"Real Prob : {real_prob:.4f}")
        print("-" * 60)

        if is_rejected:
            rejected.append(filename)

    print("\nSUMMARY")
    print("=" * 60)
    print(f"Matri ID        : {MATRI_ID}")
    print(f"Total images    : {len(IMAGE_PATHS)}")
    print(f"Rejected        : {len(rejected)}")
    print(f"Status          : {'REJECTED' if rejected else 'ACCEPTED'}")
    if rejected:
        print(f"Rejected reason : {len(rejected)} AI-generated image(s) detected")
        print("Rejected files  :")
        for f in rejected:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
