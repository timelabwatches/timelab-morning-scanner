"""
TIMELAB · Vinted Vision Layer
─────────────────────────────
Sends a listing's primary photo to Google Gemini Flash 2.0 to validate that
the alert is for a real, complete watch matching the declared brand. This is
the LAST line of defense against false positives that survive text-based
filtering: replicas, parts-only listings, and brand mismatches.

Architecture:
- Called from vinted_scanner.py AFTER all text filters and AFTER target+econ
  filtering pick the top-N candidates (typically 5).
- Single API call per candidate (one image, ~600 token output).
- Returns a VisionVerdict that callers can use to BLOCK, FLAG, or PASS the
  alert. On any error or missing API key, returns "skip" verdict — the caller
  treats this as "no verdict" and falls through to the existing behavior.
- Total daily cost on Gemini's free tier: $0 (15 req/min, 1500/day, well above
  our ~5-15 alerts/day).

Failure modes (all non-blocking):
- No GEMINI_API_KEY in env → skip
- Photo URL not present → skip
- Photo download timeout/error → skip
- Gemini API error (rate limit, server error) → skip
- Response parsing failure → skip

Configuration via env vars:
- GEMINI_API_KEY        (required for vision to be active)
- VT_VISION_ENABLED     (default 1; set 0 to disable cleanly)
- VT_VISION_MODEL       (default "gemini-2.0-flash")
- VT_VISION_TIMEOUT     (default 15 seconds)
"""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")
# Default to Gemini Flash-Lite (cheapest reliable Vision model on Google's
# tier as of May 2026). Override via VT_VISION_MODEL env var if needed.
GEMINI_MODEL      = os.environ.get("VT_VISION_MODEL", "gemini-flash-lite-latest")
VT_VISION_ENABLED = os.environ.get("VT_VISION_ENABLED", "1") == "1"
VT_VISION_TIMEOUT = int(os.environ.get("VT_VISION_TIMEOUT", "15"))

GEMINI_ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

MAX_IMAGE_BYTES = 4_000_000  # 4MB safety cap; Vinted thumbnails are ~50-300KB


# ─────────────────────────────────────────────
# RESULT TYPE
# ─────────────────────────────────────────────

@dataclass
class VisionVerdict:
    """Outcome of analyzing a listing's photo."""
    verdict: str = "skip"            # ok | replica | parts | wrong_brand | uncertain | skip
    confidence: str = "low"          # high | medium | low
    detected_brand: str = ""
    detected_model: str = ""
    condition_visual: str = ""       # nuevo | muy bueno | bueno | aceptable
    notes: str = ""                  # short Spanish description
    red_flags: List[str] = field(default_factory=list)
    error: Optional[str] = None      # populated when verdict=="skip" with reason

    @property
    def is_blocking(self) -> bool:
        """True if this verdict should cause the alert to be discarded."""
        return self.verdict in ("replica", "parts", "wrong_brand")

    @property
    def is_flagged(self) -> bool:
        """
        True if alert should be marked for manual review.
        Includes both 'uncertain' verdicts AND 'ok' verdicts that came back
        with red_flags — Vision says it's authentic but something is off
        (e.g., the model on the photo doesn't match the matched target).
        """
        return self.verdict == "uncertain" or (
            self.verdict == "ok" and bool(self.red_flags)
        )

    @property
    def is_clean(self) -> bool:
        """True if Vision validated the listing as a genuine watch with no flags."""
        return self.verdict == "ok" and not self.red_flags

    @property
    def was_skipped(self) -> bool:
        """True if Vision didn't run (no key, error, etc) — caller should ignore."""
        return self.verdict == "skip"


# ─────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────

def _build_prompt(brand_hint: str, target_id: str, model_hint: str, title: str) -> str:
    """Build the analysis prompt sent to Gemini alongside the image."""
    return f"""You are a watch arbitrage analyst reviewing a Vinted listing photo.

LISTING CONTEXT:
- Declared brand: {brand_hint or "unknown"}
- Listing title: {title}
- Matched target: {target_id}
- Target family/model expectation: {model_hint or "generic"}

YOUR TASK:
Analyze the photo and determine if this is a complete, authentic watch matching
the declared brand, suitable for resale arbitrage on Catawiki.

WATCH OUT FOR (in order of importance):
1. PARTS-ONLY → photo shows just a strap, empty box/case, crystal, dial alone,
   movement, or other components. NOT a full assembled watch.
2. REPLICAS → telltale signs: poor logo quality, wrong dial layout, suspicious
   markings, off proportions, mismatched details vs known authentic watches.
3. WRONG BRAND → the watch shown is clearly a different brand than declared.
4. CONDITION MISMATCH → visible damage, redial, heavy wear inconsistent with
   the stated condition.

RESPOND WITH ONLY A VALID JSON OBJECT — no markdown, no preamble, no commentary:

{{
  "verdict": "ok" | "replica" | "parts" | "wrong_brand" | "uncertain",
  "confidence": "high" | "medium" | "low",
  "detected_brand": "<brand visible in the photo>",
  "detected_model": "<model/family if identifiable, else empty>",
  "condition_visual": "nuevo" | "muy bueno" | "bueno" | "aceptable",
  "notes": "<one short sentence in Spanish describing what you see>",
  "red_flags": ["<short flag 1>", "<short flag 2>"]
}}

DECISION GUIDELINES:
- Use "ok" only if reasonably confident: complete authentic watch, correct brand
- Use "uncertain" if photo is blurry, ambiguous, partially obscured, or you
  cannot determine authenticity from what's visible
- Use "replica" only when you see clear counterfeiting signs
- Use "parts" if the photo clearly shows ONLY a component, not a complete watch
- Use "wrong_brand" if a different brand is unambiguously visible
- Be CONSERVATIVE: when in doubt, prefer "uncertain" over "ok"
- "red_flags" should be empty array if verdict is "ok" with high confidence
"""


# ─────────────────────────────────────────────
# IMAGE DOWNLOAD
# ─────────────────────────────────────────────

def _download_image(url: str) -> Optional[tuple]:
    """
    Downloads an image, returns (bytes, mime_type) or None on failure.
    Aborts if image exceeds MAX_IMAGE_BYTES.
    """
    try:
        r = requests.get(url, timeout=10, stream=True)
        if r.status_code != 200:
            logger.warning("[VISION] image fetch status=%s url=%s", r.status_code, url[:80])
            return None

        content = bytearray()
        for chunk in r.iter_content(chunk_size=16384):
            content.extend(chunk)
            if len(content) > MAX_IMAGE_BYTES:
                logger.warning("[VISION] image too large (>%dB) url=%s", MAX_IMAGE_BYTES, url[:80])
                return None

        # Sniff mime from magic bytes
        mime = "image/jpeg"
        if content[:4] == b"\x89PNG":
            mime = "image/png"
        elif content[:6] in (b"GIF87a", b"GIF89a"):
            mime = "image/gif"
        elif content[:4] == b"RIFF" and content[8:12] == b"WEBP":
            mime = "image/webp"
        # default jpeg

        return (bytes(content), mime)
    except requests.Timeout:
        logger.warning("[VISION] image fetch timeout url=%s", url[:80])
        return None
    except Exception as e:
        logger.warning("[VISION] image fetch error url=%s err=%s", url[:80], e)
        return None


# ─────────────────────────────────────────────
# GEMINI CALL
# ─────────────────────────────────────────────

def analyze_listing_photo(
    photo_url: str,
    brand_hint: str = "",
    target_id: str = "",
    model_hint: str = "",
    title: str = "",
) -> VisionVerdict:
    """
    Run Vision analysis on a Vinted listing's primary photo.
    Returns a VisionVerdict. Never raises — failures yield verdict='skip'.
    """
    if not VT_VISION_ENABLED:
        return VisionVerdict(verdict="skip", error="disabled")

    if not GEMINI_API_KEY:
        return VisionVerdict(verdict="skip", error="no_api_key")

    if not photo_url:
        return VisionVerdict(verdict="skip", error="no_photo_url")

    img = _download_image(photo_url)
    if not img:
        return VisionVerdict(verdict="skip", error="image_download_failed")
    img_bytes, mime = img

    img_b64 = base64.b64encode(img_bytes).decode("ascii")
    prompt = _build_prompt(brand_hint, target_id, model_hint, title)

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime, "data": img_b64}},
            ]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.2,
            "maxOutputTokens": 600,
        },
    }

    # Up to 1 retry on 429 with backoff. The free tier of Gemini can be a
    # bit volatile when bursts of multimodal requests come too fast.
    _max_attempts = 2
    for attempt in range(1, _max_attempts + 1):
        try:
            r = requests.post(
                GEMINI_ENDPOINT,
                params={"key": GEMINI_API_KEY},
                json=payload,
                timeout=VT_VISION_TIMEOUT,
                headers={"Content-Type": "application/json"},
            )
        except requests.Timeout:
            logger.warning("[VISION] gemini timeout (attempt %d)", attempt)
            return VisionVerdict(verdict="skip", error="timeout")
        except Exception as e:
            logger.warning("[VISION] gemini request error: %s", e)
            return VisionVerdict(verdict="skip", error=f"request_error:{type(e).__name__}")

        if r.status_code == 429 and attempt < _max_attempts:
            # Rate-limited: wait and retry once
            import time as _t
            wait_s = 8 * attempt
            logger.warning("[VISION] gemini rate-limited (429), waiting %ds and retrying", wait_s)
            _t.sleep(wait_s)
            continue

        if r.status_code == 429:
            logger.warning("[VISION] gemini rate-limited (429) after retry, giving up")
            return VisionVerdict(verdict="skip", error="rate_limited")
        if r.status_code != 200:
            logger.warning("[VISION] gemini status=%s body=%s", r.status_code, r.text[:200])
            return VisionVerdict(verdict="skip", error=f"api_status_{r.status_code}")
        # success
        break

    try:
        data = r.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        result = json.loads(text)
    except Exception as e:
        logger.warning("[VISION] parse error: %s", e)
        return VisionVerdict(verdict="skip", error="parse_failed")

    # Whitelist verdict values
    verdict = result.get("verdict", "uncertain")
    if verdict not in ("ok", "replica", "parts", "wrong_brand", "uncertain"):
        verdict = "uncertain"

    confidence = result.get("confidence", "low")
    if confidence not in ("high", "medium", "low"):
        confidence = "low"

    red_flags = result.get("red_flags", [])
    if not isinstance(red_flags, list):
        red_flags = []
    red_flags = [str(f)[:80] for f in red_flags[:5]]  # cap count and length

    return VisionVerdict(
        verdict=verdict,
        confidence=confidence,
        detected_brand=str(result.get("detected_brand", ""))[:50],
        detected_model=str(result.get("detected_model", ""))[:80],
        condition_visual=str(result.get("condition_visual", ""))[:30],
        notes=str(result.get("notes", ""))[:300],
        red_flags=red_flags,
    )


# ─────────────────────────────────────────────
# MESSAGE FORMATTING HELPER
# ─────────────────────────────────────────────

def format_verdict_for_telegram(v: VisionVerdict) -> str:
    """Render the verdict as a short multi-line block for Telegram messages."""
    if v.was_skipped:
        return ""  # don't clutter messages when vision didn't run

    # OK with flags → show as warning rather than green check, since something
    # the model noticed is off (typically: model on photo doesn't match target)
    if v.verdict == "ok" and v.red_flags:
        icon = "⚠️"
        label = "OK con avisos"
    else:
        icon = {
            "ok":          "✅",
            "uncertain":   "⚠️",
            "replica":     "❌",
            "parts":       "❌",
            "wrong_brand": "❌",
        }.get(v.verdict, "•")
        label = v.verdict.upper()

    lines = [f"   {icon} Vision: {label} (conf={v.confidence})"]
    if v.notes:
        lines.append(f"      {v.notes}")
    if v.red_flags:
        lines.append(f"      🚩 {' | '.join(v.red_flags)}")
    return "\n".join(lines)
