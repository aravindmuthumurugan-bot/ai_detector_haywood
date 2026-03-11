"""
main.py — FastAPI Photo Validation Service (Qwen3.5-9B via vLLM).

Prerequisites
-------------
1. Run setup.sh  (installs deps + downloads Qwen3.5-9B weights)
2. Start vLLM:   bash launch_vllm.sh            (port 8000)
3. Start this:   uvicorn main:app --port 8001    (port 8001)

Endpoint
--------
POST /validate
  Form fields:
    unique_id  (str)           — caller-supplied user / profile ID
    gender     (str)           — "M" or "F"
    age        (int)           — declared age of the profile owner
    prompt     (str, optional) — extra platform-specific rules for the model
  File field:
    photos     (1–10 files)    — JPEG / PNG / WEBP / BMP images to validate
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from checks import ALL_CHECK_IDS
from config import LOG_LEVEL, MAX_FILE_SIZE_MB, MAX_PHOTOS, PORT, SUPPORTED_FORMATS, HOST
from models import (
    ErrorResponse,
    PhotoValidationResult,
    ValidationResponse,
    ValidationSummary,
    ViolationItem,
)
from vllm_engine import QwenVLEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("photo_validation")

# ---------------------------------------------------------------------------
# App & engine lifecycle
# ---------------------------------------------------------------------------
engine: Optional[QwenVLEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("Initialising Qwen3.5-9B engine (connecting to vLLM server)...")
    engine = QwenVLEngine()
    await engine.initialize()
    logger.info("Engine ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="AI Photo Validation Service — Qwen3.5 vLLM",
    description=(
        "Validates user-uploaded photos against 53 platform policy checks "
        "using Qwen3.5-9B running on vLLM."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health", tags=["system"])
async def health():
    return {"status": "healthy", "model": "Qwen/Qwen3.5-9B", "engine_ready": engine is not None}


# ---------------------------------------------------------------------------
# Main validation endpoint
# ---------------------------------------------------------------------------


@app.post(
    "/validate",
    response_model=ValidationResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["validation"],
    summary="Validate up to 10 photos for policy violations",
)
async def validate_photos(
    unique_id: str = Form(
        ...,
        description="Unique user / profile identifier (e.g. matri ID)",
    ),
    gender: str = Form(
        ...,
        description="Profile owner's gender — 'M' or 'F'",
    ),
    age: int = Form(
        ...,
        description="Profile owner's declared age",
    ),
    prompt: Optional[str] = Form(
        None,
        description=(
            "Additional validation guidelines or platform-specific rules to pass "
            "to the model (e.g. 'reject photos showing alcohol even if subtle')."
        ),
    ),
    photos: List[UploadFile] = File(
        ...,
        description="Photos to validate. Upload 1–10 images (JPEG / PNG / WEBP / BMP).",
    ),
):
    request_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not initialised. Please wait and retry.",
        )

    if not 1 <= len(photos) <= MAX_PHOTOS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Upload between 1 and {MAX_PHOTOS} photos. Got {len(photos)}.",
        )

    gender_norm = gender.strip().upper()
    if gender_norm not in ("M", "F", "MALE", "FEMALE"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="'gender' must be 'M' or 'F'.",
        )

    if not 1 <= age <= 120:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="'age' must be between 1 and 120.",
        )

    # ------------------------------------------------------------------
    # Read & validate uploaded files
    # ------------------------------------------------------------------
    photo_data: List[tuple] = []  # (filename, bytes)

    for idx, photo in enumerate(photos, start=1):
        filename = photo.filename or f"photo_{idx}"
        ext = Path(filename).suffix.lower()

        if ext not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Photo {idx} ('{filename}'): unsupported format '{ext}'. "
                    f"Supported: {sorted(SUPPORTED_FORMATS)}"
                ),
            )

        content = await photo.read()
        max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024

        if len(content) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Photo {idx} ('{filename}') is {len(content) // (1024*1024)} MB — "
                    f"exceeds the {MAX_FILE_SIZE_MB} MB limit."
                ),
            )

        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Photo {idx} ('{filename}') is empty.",
            )

        photo_data.append((filename, content))

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------
    logger.info(
        "Validating %d photo(s) for unique_id=%s gender=%s age=%d",
        len(photo_data),
        unique_id,
        gender_norm,
        age,
    )

    image_bytes_list = [d[1] for d in photo_data]

    raw_results = await engine.validate_batch(
        image_bytes_list=image_bytes_list,
        gender=gender_norm,
        age=age,
        extra_prompt=prompt or "",
    )

    # ------------------------------------------------------------------
    # Build structured response
    # ------------------------------------------------------------------
    results: List[PhotoValidationResult] = []

    for idx, ((filename, _), raw) in enumerate(zip(photo_data, raw_results)):
        result = PhotoValidationResult(
            upload_index=idx,
            filename=filename,
            final_decision=raw["decision"],
            final_reason=raw["reason"],
            face_count=raw["face_count"],
            photo_type=raw["photo_type"],
            age_estimate=raw["age_estimate"],
            gender_detected=raw["gender_detected"],
            violations=[
                ViolationItem(check_id=v["id"], reason=v["why"])
                for v in raw["violations"]
            ],
            uncertain=[
                ViolationItem(check_id=u["id"], reason=u["why"])
                for u in raw["uncertain"]
            ],
            all_checks=raw["all_checks"],
        )
        results.append(result)

    # Summary counts
    approved = sum(1 for r in results if r.final_decision == "APPROVE")
    rejected = sum(1 for r in results if r.final_decision == "REJECT")
    suspended = sum(1 for r in results if r.final_decision == "SUSPEND")

    # First approved individual photo = primary candidate
    primary_index = next(
        (
            r.upload_index
            for r in results
            if r.final_decision == "APPROVE" and r.photo_type == "individual"
        ),
        None,
    )

    total_time = round(time.perf_counter() - request_start, 3)

    logger.info(
        "Completed unique_id=%s — approved=%d rejected=%d suspended=%d  (%.2fs)",
        unique_id,
        approved,
        rejected,
        suspended,
        total_time,
    )

    return ValidationResponse(
        success=True,
        unique_id=unique_id,
        gender=gender_norm,
        age=age,
        total_photos=len(photo_data),
        results=results,
        summary=ValidationSummary(
            total_photos=len(photo_data),
            approved=approved,
            rejected=rejected,
            suspended=suspended,
            primary_photo_index=primary_index,
        ),
        total_processing_time_seconds=total_time,
        model_used="Qwen/Qwen3.5-9B",
    )


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=HOST, port=PORT, workers=1, reload=False)
