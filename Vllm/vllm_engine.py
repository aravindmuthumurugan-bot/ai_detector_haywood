"""
vllm_engine.py — Qwen3.5-9B inference via vLLM's OpenAI-compatible server.

Architecture
------------
vLLM is launched as a separate process (see setup.sh / launch_vllm.sh):
    vllm serve Qwen/Qwen3.5-9B --host 0.0.0.0 --port 8000

This module calls that server using the standard openai Python client.
Images are encoded as base64 data-URLs and sent as `image_url` content blocks
(the same format Qwen3.5 was designed for).

All calls are async-native so FastAPI never blocks.
"""

import asyncio
import base64
import io
import json
import logging
import re
from typing import Any, Dict, List

from openai import AsyncOpenAI
from PIL import Image

from checks import ALL_CHECK_IDS
from config import (
    MAX_NEW_TOKENS,
    MODEL_NAME,
    TEMPERATURE,
    TOP_P,
    VLLM_API_KEY,
    VLLM_SERVER_URL,
)
from prompts import build_system_prompt, build_user_prompt

logger = logging.getLogger(__name__)


class QwenVLEngine:
    """
    Async client wrapper for the vLLM OpenAI-compatible inference server.

    The vLLM server must already be running before the FastAPI app starts.
    Use `setup.sh` or run manually:
        vllm serve Qwen/Qwen3.5-9B --host 0.0.0.0 --port 8000 --max-model-len 32768
    """

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None
        self._model = MODEL_NAME

    # ------------------------------------------------------------------
    # Initialisation  (called from FastAPI lifespan)
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        self._client = AsyncOpenAI(
            base_url=VLLM_SERVER_URL,
            api_key=VLLM_API_KEY,
        )
        # Verify the server is reachable
        try:
            models = await self._client.models.list()
            available = [m.id for m in models.data]
            logger.info("vLLM server reachable. Available models: %s", available)
        except Exception as exc:
            logger.warning(
                "vLLM server not reachable at %s: %s — will retry on first request.",
                VLLM_SERVER_URL,
                exc,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def validate_batch(
        self,
        image_bytes_list: List[bytes],
        gender: str,
        age: int,
        extra_prompt: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Validate a batch of images concurrently.

        All images are dispatched as independent async tasks so the vLLM server
        can process them in parallel.
        Returns results in the same order as the input list.
        """
        tasks = [
            self._validate_single(img_bytes, gender, age, extra_prompt)
            for img_bytes in image_bytes_list
        ]
        return await asyncio.gather(*tasks)

    # ------------------------------------------------------------------
    # Per-image validation
    # ------------------------------------------------------------------

    async def _validate_single(
        self,
        image_bytes: bytes,
        gender: str,
        age: int,
        extra_prompt: str,
    ) -> Dict[str, Any]:
        data_url = self._encode_image(image_bytes)
        if data_url is None:
            return self._error_result("Cannot read/decode image")

        system_prompt = build_system_prompt()
        user_text = build_user_prompt(gender, age, extra_prompt)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                # Qwen3.5 thinking mode — disable for validation (determinism)
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw_text = response.choices[0].message.content or ""
            return self._parse_llm_output(raw_text)

        except Exception as exc:
            logger.error("vLLM call failed: %s", exc, exc_info=True)
            return self._error_result(f"Inference error: {exc}")

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image(data: bytes) -> "str | None":
        """Convert raw image bytes → JPEG base64 data-URL."""
        try:
            img = Image.open(io.BytesIO(data))
            if img.mode not in ("RGB",):
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"
        except Exception as exc:
            logger.warning("Image encode failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> str:
        """Strip markdown fences and return the first JSON object."""
        text = text.strip()
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = text.replace("```", "")
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return text[start:end]
        return text

    def _parse_llm_output(self, raw: str) -> Dict[str, Any]:
        try:
            payload = json.loads(self._extract_json(raw))
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("JSON parse error (%s). Raw: %.300s", exc, raw)
            return self._error_result("Unable to parse model response")

        def _normalise_items(items: list) -> list:
            out = []
            for item in items:
                if isinstance(item, dict):
                    cid = item.get("id") or item.get("check_id") or ""
                    why = item.get("why") or item.get("reason") or ""
                    if cid:
                        out.append({"id": str(cid), "why": str(why)})
            return out

        violations = _normalise_items(payload.get("violations", []))
        uncertain = _normalise_items(payload.get("uncertain", []))

        decision = str(payload.get("decision", "REJECT")).upper()
        if decision not in ("APPROVE", "REJECT", "SUSPEND"):
            decision = "REJECT"

        # If violations exist, decision must not be APPROVE
        if violations and decision == "APPROVE":
            decision = "REJECT"

        reason = str(payload.get("reason", "")).strip()
        if not reason and violations:
            reason = violations[0].get("why") or violations[0].get("id", "Policy violation")
        if not reason:
            reason = "All checks passed" if decision == "APPROVE" else "Validation failed"

        face_count = max(0, int(payload.get("face_count", 0)))

        photo_type = str(payload.get("photo_type", "unknown")).lower()
        if photo_type not in ("individual", "group", "no_face"):
            photo_type = "unknown"

        age_est = payload.get("age_estimate")
        try:
            age_est = int(age_est) if age_est is not None else None
        except (TypeError, ValueError):
            age_est = None

        gender_det = str(payload.get("gender_detected", "unclear")).lower()
        if gender_det not in ("male", "female", "unclear"):
            gender_det = "unclear"

        # Build full checks map
        all_checks: Dict[str, str] = {cid: "PASS" for cid in ALL_CHECK_IDS}
        for v in violations:
            cid = v.get("id", "")
            if cid in all_checks:
                all_checks[cid] = "FAIL"
        for u in uncertain:
            cid = u.get("id", "")
            if cid in all_checks and all_checks[cid] == "PASS":
                all_checks[cid] = "UNCERTAIN"

        return {
            "violations": violations,
            "uncertain": uncertain,
            "all_checks": all_checks,
            "face_count": face_count,
            "photo_type": photo_type,
            "age_estimate": age_est,
            "gender_detected": gender_det,
            "decision": decision,
            "reason": reason,
        }

    @staticmethod
    def _error_result(reason: str = "Image could not be processed") -> Dict[str, Any]:
        return {
            "violations": [],
            "uncertain": [],
            "all_checks": {cid: "PASS" for cid in ALL_CHECK_IDS},
            "face_count": 0,
            "photo_type": "unknown",
            "age_estimate": None,
            "gender_detected": "unclear",
            "decision": "REJECT",
            "reason": reason,
        }
