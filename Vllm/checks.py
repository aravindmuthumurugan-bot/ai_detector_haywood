"""
checks.py — Defines all 53 photo-validation checks (plus existing-code checks).

Each check has:
  - id       : machine-readable key returned in API responses
  - name     : short human-readable label
  - severity : REJECT | SUSPEND
  - category : grouping for display / filtering
  - description : detailed text sent to the LLM inside the prompt

Mapping to the 53-point checklist supplied by the product team:
  Check  1  → nsfw_explicit
  Check  2  → nsfw_suggestive
  Check  3  → pii_contact
  Check  4  → qr_codes
  Check  5  → ads_promotion
  Check  6  → logos_watermarks
  Check  7  → screenshot
  Check  8  → meme_graphic
  Check  9  → celebrity_stock
  Check 10  → ai_generated
  Check 11  → (cross-profile duplicate — requires DB, out of scope for this service)
  Check 12  → (within-profile duplicate — removed per product decision)
  Check 13  → photo_of_screen
  Check 14  → photo_of_document
  Check 15  → objects_only
  Check 16  → no_face
  Check 17  → multiple_faces
  Check 18  → group_photo
  Check 19  → minors_no_owner
  Check 20  → minor_in_adult_profile
  Check 21  → face_covered
  Check 22  → face_cropped
  Check 23  → face_too_small
  Check 24  → body_parts_only
  Check 25  → collage
  Check 26  → sticker_overlay
  Check 27  → heavily_filtered
  Check 28  → blurry_low_res
  Check 29  → wrong_orientation
  Check 30  → face_too_small  (CCTV-style — same check)
  Check 31  → alcohol_drugs
  Check 32  → violence_weapons
  Check 33  → obscene_gestures
  Check 34  → hate_symbols
  Check 35  → hate_speech_image
  Check 36  → political_propaganda
  Check 37  → religious_insults
  Check 38  → abusive_text
  Check 39  → slurs_text
  Check 40  → vernacular_abuse
  Check 41  → sexually_suggestive_text
  Check 42  → harassment_text
  Check 43  → caste_discrimination
  Check 44  → threatening_text
  Check 45  → objectionable_memes
  Check 46  → body_shaming_text
  Check 47  → coded_abuse
  Check 48  → religious_propaganda
  Check 49  → protest_slogans
  Check 50  → banned_orgs
  Check 51  → indoor_swimwear
  Check 52  → gore_blood
  Check 53  → self_harm

Additional checks from existing ML pipeline (NSFW, CLIP, PII, orientation, blur):
  photo_of_photo, enhancement_filters — folded into existing IDs above.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CheckDef:
    name: str
    severity: str        # "REJECT" | "SUSPEND"
    category: str
    description: str     # Sent verbatim to the LLM


# ---------------------------------------------------------------------------
# Master check registry (ordered: severity groups → alphabetical within group)
# ---------------------------------------------------------------------------
ALL_CHECKS: Dict[str, CheckDef] = {

    # =========================================================================
    # NSFW / Sexual content
    # =========================================================================
    "nsfw_explicit": CheckDef(
        name="Explicit Nudity / Sexual Activity",
        severity="REJECT",
        category="nsfw",
        description=(
            "Explicit nudity, visible genitalia, sexual intercourse or sexual activity "
            "of any kind is present in the image."
        ),
    ),
    "nsfw_suggestive": CheckDef(
        name="Sexually Suggestive / Partial Nudity",
        severity="REJECT",
        category="nsfw",
        description=(
            "Sexually suggestive poses, lingerie or underwear worn as the main outfit, "
            "or partial nudity that is not appropriate for a matrimonial platform."
        ),
    ),
    "indoor_swimwear": CheckDef(
        name="Indoor Swimwear / Shirtless",
        severity="REJECT",
        category="nsfw",
        description=(
            "Person is wearing bikini, swimwear, underwear, or is shirtless in an "
            "indoor setting not at a pool, beach or sports facility."
        ),
    ),

    # =========================================================================
    # Harmful / violent / dangerous content
    # =========================================================================
    "gore_blood": CheckDef(
        name="Gore / Blood / Injury",
        severity="REJECT",
        category="harmful",
        description=(
            "Image contains gore, blood, visible injuries, bodily fluids, corpses, "
            "dead animals, or hunting kills."
        ),
    ),
    "self_harm": CheckDef(
        name="Self-Harm / Suicide Depiction",
        severity="REJECT",
        category="harmful",
        description=(
            "Image promotes, glorifies or depicts self-harm, cutting, or suicide."
        ),
    ),
    "violence_weapons": CheckDef(
        name="Violence / Weapons",
        severity="REJECT",
        category="harmful",
        description=(
            "Image shows weapons (firearms, knives, blunt objects used threateningly), "
            "acts of violence, or threatening poses directed at others."
        ),
    ),
    "obscene_gestures": CheckDef(
        name="Obscene / Sexually Suggestive Gestures",
        severity="REJECT",
        category="harmful",
        description=(
            "Person is making obscene gestures, sexually suggestive hand signs, or "
            "vulgar body-language poses."
        ),
    ),
    "hate_symbols": CheckDef(
        name="Hate Symbols / Extremist Logos",
        severity="REJECT",
        category="harmful",
        description=(
            "Image contains hate symbols, extremist organisation logos, Nazi symbols, "
            "or any imagery associated with hate groups."
        ),
    ),
    "alcohol_drugs": CheckDef(
        name="Alcohol / Drugs / Smoking",
        severity="REJECT",
        category="harmful",
        description=(
            "Visible consumption of alcohol, drugs or tobacco/smoking; paraphernalia "
            "clearly associated with substance abuse."
        ),
    ),

    # =========================================================================
    # Photo authenticity
    # =========================================================================
    "ai_generated": CheckDef(
        name="AI-Generated / Deepfake / Synthetic",
        severity="REJECT",
        category="authenticity",
        description=(
            "Face or full image appears AI-generated, deepfake, CGI or synthetically "
            "created — not a real photograph of a real person."
        ),
    ),
    "celebrity_stock": CheckDef(
        name="Celebrity / Stock Photo / Downloaded Image",
        severity="REJECT",
        category="authenticity",
        description=(
            "Image appears to be a photo of a celebrity, well-known public figure, "
            "influencer, or a stock photo / promotional image downloaded from the internet."
        ),
    ),
    "photo_of_photo": CheckDef(
        name="Photo of a Printed Photo",
        severity="REJECT",
        category="authenticity",
        description=(
            "Image is a re-photograph of a printed photograph, a physical photo album "
            "page, or a photograph laid on a surface."
        ),
    ),
    "photo_of_screen": CheckDef(
        name="Photo of a Digital Screen",
        severity="REJECT",
        category="authenticity",
        description=(
            "Image is a photograph taken of a laptop screen, mobile display, TV, "
            "computer monitor or any other digital display showing content."
        ),
    ),
    "screenshot": CheckDef(
        name="Screenshot / App Interface",
        severity="REJECT",
        category="authenticity",
        description=(
            "Image is a screenshot of a chat application, social media profile "
            "(Instagram, Facebook, etc.), dating app, or any software interface."
        ),
    ),
    "meme_graphic": CheckDef(
        name="Meme / Quote Card / Graphic",
        severity="REJECT",
        category="authenticity",
        description=(
            "Image is a meme, quote card, motivational poster, infographic, "
            "illustration or graphic — not a photograph of a real person."
        ),
    ),
    "collage": CheckDef(
        name="Collage / Multi-Photo Frame",
        severity="REJECT",
        category="authenticity",
        description=(
            "Image is a collage combining multiple separate photographs or frames "
            "arranged together in a single image."
        ),
    ),
    "photo_of_document": CheckDef(
        name="Document Photo (ID / Certificate)",
        severity="REJECT",
        category="authenticity",
        description=(
            "Image shows an Aadhaar card, passport, PAN card, voter ID, driving "
            "licence, birth certificate or any other official document."
        ),
    ),

    # =========================================================================
    # Text / contact violations (OCR-detectable by VLM)
    # =========================================================================
    "pii_contact": CheckDef(
        name="Personal Contact / Social Media Info",
        severity="REJECT",
        category="text_violations",
        description=(
            "Visible phone numbers, email addresses, Instagram handles, Facebook IDs, "
            "Twitter/X handles, WhatsApp, Telegram, or Snapchat contact details in image."
        ),
    ),
    "qr_codes": CheckDef(
        name="QR Codes / Payment Links",
        severity="REJECT",
        category="text_violations",
        description=(
            "QR codes, UPI payment links, barcodes, Telegram or WhatsApp invite "
            "links, or any other scannable code visible in the image."
        ),
    ),
    "ads_promotion": CheckDef(
        name="Website URLs / Ads / Brand Promotion",
        severity="REJECT",
        category="text_violations",
        description=(
            "Website URLs, brand slogans, promotional text, advertisement overlays, "
            "or product endorsement content visible in the image."
        ),
    ),
    "logos_watermarks": CheckDef(
        name="Logos / Watermarks / Studio Branding",
        severity="REJECT",
        category="text_violations",
        description=(
            "Studio logos, copyright watermarks, brand logos, or photographer "
            "branding overlaid on the photo."
        ),
    ),

    # =========================================================================
    # Offensive / hate-speech text (OCR-detectable by VLM)
    # =========================================================================
    "abusive_text": CheckDef(
        name="Abusive / Vulgar / Obscene Text",
        severity="REJECT",
        category="offensive_text",
        description=(
            "Abusive, vulgar or obscene language is visible as text within the image "
            "(English or any language)."
        ),
    ),
    "vernacular_abuse": CheckDef(
        name="Vernacular Abusive Words",
        severity="REJECT",
        category="offensive_text",
        description=(
            "Abusive words in Indian regional languages (Tamil, Hindi, Telugu, "
            "Kannada, Malayalam, Marathi, Bengali, Gujarati, Punjabi, etc.) visible "
            "as text in the image."
        ),
    ),
    "sexually_suggestive_text": CheckDef(
        name="Sexually Suggestive / Vulgar Text",
        severity="REJECT",
        category="offensive_text",
        description=(
            "Sexually suggestive phrases, double-meaning vulgar text or innuendo "
            "visible as text within the image."
        ),
    ),
    "harassment_text": CheckDef(
        name="Harassment / Insult Text",
        severity="REJECT",
        category="offensive_text",
        description=(
            "Harassment, insults or demeaning content targeting women, men or any "
            "gender group visible as text in the image."
        ),
    ),
    "caste_discrimination": CheckDef(
        name="Caste Superiority / Discrimination Text",
        severity="REJECT",
        category="offensive_text",
        description=(
            "Text promoting caste superiority, caste discrimination, caste-based "
            "hatred or caste-targeted slurs visible in the image."
        ),
    ),
    "threatening_text": CheckDef(
        name="Threatening / Intimidation Text",
        severity="REJECT",
        category="offensive_text",
        description=(
            "Threatening language, intimidation messages, or calls for violence "
            "visible as text within the image."
        ),
    ),
    "hate_speech_image": CheckDef(
        name="Hate-Speech Imagery",
        severity="REJECT",
        category="offensive_text",
        description=(
            "Image visually promotes hatred against a caste, religion, race, region "
            "or community (beyond text alone — e.g. propaganda imagery, caricatures)."
        ),
    ),
    "hate_speech_text": CheckDef(
        name="Hate-Speech Text",
        severity="REJECT",
        category="offensive_text",
        description=(
            "Text visible in the image promotes hatred against caste, religion, "
            "race, ethnicity or community."
        ),
    ),
    "slurs_text": CheckDef(
        name="Slurs (Caste / Religion / Gender / Region)",
        severity="REJECT",
        category="offensive_text",
        description=(
            "Slurs targeting caste, religion, gender, region or ethnicity visible "
            "as text in the image."
        ),
    ),
    "body_shaming_text": CheckDef(
        name="Body Shaming / Skin-Colour Insult Text",
        severity="REJECT",
        category="offensive_text",
        description=(
            "Body-shaming content or skin-colour insults visible as text within the image."
        ),
    ),
    "coded_abuse": CheckDef(
        name="Coded / Emoji-Disguised Abuse",
        severity="REJECT",
        category="offensive_text",
        description=(
            "Creatively spelled or emoji-coded abusive words designed to bypass "
            "text filters visible in the image."
        ),
    ),
    "religious_propaganda": CheckDef(
        name="Religious Propaganda / Anti-Religion / Desecration",
        severity="REJECT",
        category="offensive_text",
        description=(
            "Religious conversion propaganda, anti-religion messaging, desecration "
            "of religious symbols, or religiously offensive content visible in image."
        ),
    ),
    "political_propaganda": CheckDef(
        name="Political Propaganda / Campaign Banners",
        severity="REJECT",
        category="offensive_text",
        description=(
            "Political party propaganda, election campaign banners, political "
            "slogans or campaign materials visible in the image."
        ),
    ),
    "protest_slogans": CheckDef(
        name="Protest Slogans / Inflammatory Banners",
        severity="REJECT",
        category="offensive_text",
        description=(
            "Protest slogans, inflammatory banners or incitement content visible "
            "in the image."
        ),
    ),
    "banned_orgs": CheckDef(
        name="Banned Organisation Content",
        severity="REJECT",
        category="offensive_text",
        description=(
            "Content associated with banned organisations, extremist movements "
            "or terrorist organisations visible in the image."
        ),
    ),
    "objectionable_memes": CheckDef(
        name="Objectionable / Community-Targeting Memes",
        severity="REJECT",
        category="offensive_text",
        description=(
            "Objectionable memes or insulting captions that target communities, "
            "castes, religions or groups visible in the image."
        ),
    ),

    # =========================================================================
    # Face quality & identity
    # =========================================================================
    "no_face": CheckDef(
        name="No Human Face Detected",
        severity="REJECT",
        category="face_quality",
        description=(
            "No human face is visible or detectable in the image."
        ),
    ),
    "face_covered": CheckDef(
        name="Face Covered / Hidden",
        severity="REJECT",
        category="face_quality",
        description=(
            "The person's face is heavily covered or obscured by sunglasses, helmet, "
            "surgical mask, full-face niqab/veil, scarf, or any other covering that "
            "makes identity verification impossible."
        ),
    ),
    "face_too_small": CheckDef(
        name="Face Too Small / Distant (CCTV-style)",
        severity="REJECT",
        category="face_quality",
        description=(
            "The face occupies a very small area of the frame — CCTV-style distant "
            "shot — making identity verification impossible."
        ),
    ),
    "face_cropped": CheckDef(
        name="Face Partially Cut Off",
        severity="REJECT",
        category="face_quality",
        description=(
            "The face is partially cut off or not fully visible within the image "
            "frame (e.g. top of head, chin or sides missing significantly)."
        ),
    ),
    "multiple_faces": CheckDef(
        name="Multiple Faces — Primary Cannot Be Identified",
        severity="REJECT",
        category="face_quality",
        description=(
            "Multiple faces are present but the profile owner cannot be clearly "
            "identified as the primary subject."
        ),
    ),
    "group_photo": CheckDef(
        name="Group Photo — Owner Not Identifiable",
        severity="REJECT",
        category="face_quality",
        description=(
            "This is a group photo and the profile owner cannot be clearly "
            "identified from among the group."
        ),
    ),
    "body_parts_only": CheckDef(
        name="Body Parts Only (No Face)",
        severity="REJECT",
        category="face_quality",
        description=(
            "Only body parts (torso, hands, legs, back, etc.) are visible — "
            "no face present."
        ),
    ),
    "minors_no_owner": CheckDef(
        name="Only Children / Minors — No Profile Owner",
        severity="REJECT",
        category="face_quality",
        description=(
            "The photo contains only children or minors with the profile owner "
            "not present in the image."
        ),
    ),
    "minor_in_adult_profile": CheckDef(
        name="Minor Detected in Adult Profile",
        severity="SUSPEND",
        category="face_quality",
        description=(
            "The primary subject of the photo appears to be a child or minor "
            "(estimated age well below adulthood), but the profile is registered "
            "as an adult. Flag for human review."
        ),
    ),

    # =========================================================================
    # Photo quality
    # =========================================================================
    "blurry_low_res": CheckDef(
        name="Blurry / Extremely Low Resolution",
        severity="REJECT",
        category="quality",
        description=(
            "The photo is extremely blurry, heavily pixelated or of such low "
            "resolution that identity cannot be verified."
        ),
    ),
    "wrong_orientation": CheckDef(
        name="Incorrect Orientation",
        severity="REJECT",
        category="quality",
        description=(
            "The photo is upside down, rotated 90°, or incorrectly oriented such "
            "that the subject is not right-side up."
        ),
    ),
    "heavily_filtered": CheckDef(
        name="Heavily Filtered / Beautified / Cartoon Style",
        severity="REJECT",
        category="quality",
        description=(
            "Image is heavily filtered, over-beautified to an unrecognisable degree, "
            "or rendered in a cartoon, anime, Ghibli or illustration style that "
            "distorts or replaces real facial features."
        ),
    ),
    "sticker_overlay": CheckDef(
        name="Stickers / Emojis / Overlays on Face",
        severity="REJECT",
        category="quality",
        description=(
            "Stickers, emojis, decorative borders, or overlays are covering or "
            "significantly obscuring the face."
        ),
    ),
    "objects_only": CheckDef(
        name="Objects / Scenery / Pets Only — No Person",
        severity="REJECT",
        category="quality",
        description=(
            "The image contains only objects, scenery, pets, vehicles or places — "
            "no person is present."
        ),
    ),
}

# Ordered list of all check IDs (used to build the all_checks map in responses)
ALL_CHECK_IDS: List[str] = list(ALL_CHECKS.keys())
