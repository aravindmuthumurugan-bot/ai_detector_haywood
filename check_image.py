from transformers import pipeline, AutoImageProcessor
from PIL import Image

# ─── Set your values here ─────────────────────────────────────────────────────

MATRI_ID   = "M12345"
IMAGE_PATH = r"Ai_images\Anjali.png"   # ← change this to your image path

# ─── Config ───────────────────────────────────────────────────────────────────

MODEL_PATH          = "./model_haywood"
REJECTION_THRESHOLD = 0.65

# ─── Load & Run ───────────────────────────────────────────────────────────────

processor  = AutoImageProcessor.from_pretrained(MODEL_PATH)
classifier = pipeline(
    "image-classification",
    model=MODEL_PATH,
    image_processor=processor,
    device=-1,  # CPU; change to 0 for GPU
)

image   = Image.open(IMAGE_PATH).convert("RGB")
results = classifier(image)

scores    = {r["label"]: r["score"] for r in results}
ai_prob   = scores.get("ai", scores.get("artificial", 0.0))
real_prob = scores.get("real", scores.get("human", 1.0 - ai_prob))

is_rejected = ai_prob >= REJECTION_THRESHOLD

print("=" * 50)
print(f"Matri ID  : {MATRI_ID}")
print(f"Image     : {IMAGE_PATH}")
print(f"Verdict   : {'REJECTED' if is_rejected else 'ACCEPTED'}")
print(f"AI Prob   : {ai_prob:.4f}")
print(f"Real Prob : {real_prob:.4f}")
print(f"Reason    : {'AI-generated image detected' if is_rejected else 'Image looks real'}")
print("=" * 50)
