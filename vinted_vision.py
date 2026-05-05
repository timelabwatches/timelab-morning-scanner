"""
TIMELAB · Vinted Vision shim
─────────────────────────────
Backwards-compatibility module. The actual Vision logic now lives in
gemini_vision.py (shared by all scanners). This file just re-exports so
existing imports `from vinted_vision import ...` keep working.

If you're writing new code, import from gemini_vision directly.
"""

from gemini_vision import (  # noqa: F401
    VisionVerdict,
    analyze_listing_photo,
    format_verdict_for_telegram,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    VT_VISION_ENABLED,
    VT_VISION_TIMEOUT,
)
