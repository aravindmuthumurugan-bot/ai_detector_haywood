"""
models.py — Pydantic request / response schemas for the validation API.
"""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class ViolationItem(BaseModel):
    check_id: str = Field(..., description="Machine-readable check identifier")
    reason: str = Field(..., description="Why this check failed or is uncertain")


class ValidationSummary(BaseModel):
    total_photos: int
    approved: int
    rejected: int
    suspended: int
    primary_photo_index: Optional[int] = Field(
        None,
        description="0-based index of the best approved individual photo (primary)",
    )


# ---------------------------------------------------------------------------
# Per-photo result
# ---------------------------------------------------------------------------

class PhotoValidationResult(BaseModel):
    upload_index: int = Field(
        ..., description="0-based position in the uploaded photos list"
    )
    filename: str

    final_decision: Literal["APPROVE", "REJECT", "SUSPEND"]
    final_reason: str = Field(..., description="Human-readable primary reason")

    face_count: int = Field(0, description="Number of faces detected by the model")
    photo_type: Literal["individual", "group", "no_face", "unknown"] = "unknown"

    age_estimate: Optional[int] = Field(
        None, description="Model's estimate of the subject's age (may be null)"
    )
    gender_detected: Literal["male", "female", "unclear"] = "unclear"

    violations: List[ViolationItem] = Field(
        default_factory=list,
        description="Checks that FAILED — reasons the photo was rejected",
    )
    uncertain: List[ViolationItem] = Field(
        default_factory=list,
        description="Checks where the model was not fully confident",
    )
    all_checks: Dict[str, Literal["PASS", "FAIL", "UNCERTAIN"]] = Field(
        default_factory=dict,
        description="Complete map of every check_id → PASS / FAIL / UNCERTAIN",
    )

    processing_time_seconds: float = Field(
        0.0, description="Time taken to validate this specific photo"
    )


# ---------------------------------------------------------------------------
# Top-level response
# ---------------------------------------------------------------------------

class ValidationResponse(BaseModel):
    success: bool = True
    unique_id: str
    gender: str
    age: int
    total_photos: int

    results: List[PhotoValidationResult]
    summary: ValidationSummary

    total_processing_time_seconds: float
    model_used: str = "Qwen/Qwen3.5-9B"


# ---------------------------------------------------------------------------
# Error response (returned on HTTP 4xx / 5xx)
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    success: bool = False
    error_code: str
    message: str
    unique_id: Optional[str] = None
