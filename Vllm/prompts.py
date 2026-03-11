"""
prompts.py — Builds the system and user prompts sent to Qwen2.5-VL.

Design goals:
  • Keep total prompt tokens ≤ 1 400 (image tokens use ~640–3 000 depending on
    resolution, leaving enough room within an 8 192-token context window).
  • Ask the model to return ONLY a compact JSON listing violations — not a full
    per-check table — so the output is small and easy to parse.
  • Include all 53 check IDs in a grouped reference list so the model can name
    them accurately.
"""

from checks import ALL_CHECK_IDS


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a strict but fair photo validation AI for a matrimonial and "
    "dating platform (BharatMatrimony / similar). "
    "Your sole job is to detect policy violations in user-uploaded photos. "
    "You must respond with ONLY valid JSON — no explanations outside the JSON. "
    "Be accurate: a false positive harms genuine users; a false negative "
    "harms platform safety."
)


# ---------------------------------------------------------------------------
# Check reference block (injected once into the user prompt)
# ---------------------------------------------------------------------------
_CHECK_REFERENCE = """
AVAILABLE CHECK IDs — use ONLY these exact strings in your JSON:

[NSFW/SEXUAL]
nsfw_explicit, nsfw_suggestive, indoor_swimwear

[HARMFUL]
gore_blood, self_harm, violence_weapons, obscene_gestures, hate_symbols, alcohol_drugs

[AUTHENTICITY]
ai_generated, celebrity_stock, photo_of_photo, photo_of_screen, screenshot,
meme_graphic, collage, photo_of_document

[TEXT/CONTACT violations visible in image]
pii_contact, qr_codes, ads_promotion, logos_watermarks

[OFFENSIVE TEXT visible in image]
abusive_text, vernacular_abuse, sexually_suggestive_text, harassment_text,
caste_discrimination, threatening_text, hate_speech_image, hate_speech_text,
slurs_text, body_shaming_text, coded_abuse, religious_propaganda,
political_propaganda, protest_slogans, banned_orgs, objectionable_memes

[FACE QUALITY / IDENTITY]
no_face, face_covered, face_too_small, face_cropped, multiple_faces,
group_photo, body_parts_only, minors_no_owner, minor_in_adult_profile

[PHOTO QUALITY]
blurry_low_res, wrong_orientation, heavily_filtered, sticker_overlay, objects_only
""".strip()


# ---------------------------------------------------------------------------
# JSON output schema (shown as an example inside the prompt)
# ---------------------------------------------------------------------------
_JSON_SCHEMA = """{
  "violations": [
    {"id": "<check_id>", "why": "<brief reason>"}
  ],
  "uncertain": [
    {"id": "<check_id>", "why": "<brief reason>"}
  ],
  "face_count": <integer 0+>,
  "photo_type": "<individual|group|no_face>",
  "age_estimate": <integer or null>,
  "gender_detected": "<male|female|unclear>",
  "decision": "<APPROVE|REJECT|SUSPEND>",
  "reason": "<primary human-readable reason for the decision>"
}"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    return SYSTEM_PROMPT


def build_user_prompt(gender: str, age: int, extra_rules: str = "") -> str:
    """
    Build the user-turn prompt.

    Parameters
    ----------
    gender      : "M" / "F" / "MALE" / "FEMALE"
    age         : declared age of the profile owner
    extra_rules : optional free-text guidelines added by the caller (product /
                  platform-specific rules sent via the `prompt` API parameter)
    """
    gender_label = "Male" if gender.upper().startswith("M") else "Female"

    extra_section = ""
    if extra_rules and extra_rules.strip():
        extra_section = (
            f"\n\nADDITIONAL PLATFORM RULES (apply these too):\n{extra_rules.strip()}"
        )

    prompt = f"""Analyze the uploaded photo for a {gender_label} profile owner aged {age}.{extra_section}

{_CHECK_REFERENCE}

RULES:
• List ONLY checks that FAIL (violations) or are UNCERTAIN — omit passing checks.
• decision = REJECT   if one or more violations exist.
• decision = SUSPEND  only when the sole violation is minor_in_adult_profile
  (face appears to be a child despite an adult profile age).
• decision = APPROVE  if no violations and no UNCERTAIN flags.
• For minor_in_adult_profile: consider the declared age ({age}) — flag only if
  the person in the photo looks clearly under 16 regardless of declared age.
• For group_photo / multiple_faces: flag only if you genuinely cannot tell who
  the primary person is.
• For ai_generated: flag if face/image has telltale synthetic artefacts.

Return ONLY the following JSON (no markdown, no extra text):
{_JSON_SCHEMA}"""

    return prompt
