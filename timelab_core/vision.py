"""Phase-2 multimodal hooks (non-blocking).

This module does not run CV directly in production yet; it ingests optional
precomputed vision annotations (manual or external model output) and converts
those into scanner-friendly structured hints.
"""

import json
from pathlib import Path
from typing import Dict


def load_vision_annotations(path: str = "vision_annotations.json") -> Dict[str, Dict]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items() if isinstance(v, dict)}
    return {}


def summarize_visual_hints(row: Dict) -> Dict[str, bool]:
    row = row or {}
    chrono = bool(row.get("chrono_real") or row.get("subdials_visible"))
    diver = bool(row.get("diver_real") or row.get("diver_bezel"))
    full_set = bool(row.get("full_set_visible") or (row.get("box_visible") and row.get("papers_visible")))
    conflict = bool(row.get("text_image_conflict"))
    return {
        "chrono_real": chrono,
        "diver_real": diver,
        "full_set_visible": full_set,
        "conflict": conflict,
    }