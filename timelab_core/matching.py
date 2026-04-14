import re
from typing import Dict, List

MICROBRAND_OR_HOMAGE_TERMS = {
    "pagani", "pagani design", "addiesdive", "steeldive", "san martin", "benyar", "cadisen",
    "heimdallr", "milifortic", "baltany", "parnis", "invicta pro diver", "homage", "homenaje",
}

MOVEMENT_ONLY_TOKENS = {
    "vk63", "nh35", "nh36", "nh34", "seagull st19", "st1901", "pt5000", "eta clone",
}

MOVEMENT_CONTEXT = {
    "movement", "movimiento", "caliber", "calibre", "uhrwerk", "werk", "solo movimiento",
    "for parts", "parts only", "movement only", "only movement",
}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def keyword_in_title(keyword: str, title: str) -> bool:
    kw = norm(keyword)
    t = norm(title)
    if not kw or not t:
        return False
    if kw in t:
        return True
    compact_kw = re.sub(r"[^a-z0-9]", "", kw)
    compact_title = re.sub(r"[^a-z0-9]", "", t)
    if compact_kw and compact_kw in compact_title:
        return True
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", kw) if tok]
    if len(tokens) >= 2:
        return all(tok in t for tok in tokens)
    return False


def detect_brand_ambiguity(text: str, target_brand: str) -> Dict[str, object]:
    t = norm(text)
    b = norm(target_brand)
    has_brand = b in t
    has_microbrand = any(term in t for term in MICROBRAND_OR_HOMAGE_TERMS)
    has_movement_only = any(term in t for term in MOVEMENT_ONLY_TOKENS)
    has_movement_context = any(term in t for term in MOVEMENT_CONTEXT)

    movement_brand_contamination = (
        has_brand and (has_movement_only or has_movement_context) and has_microbrand
    )

    penalty = 0
    reason_flags: List[str] = []
    if has_microbrand:
        penalty += 18
        reason_flags.append("microbrand_or_homage")
    if has_movement_only:
        penalty += 28
        reason_flags.append("movement_caliber_token")
    if has_movement_context and not has_brand:
        penalty += 20
        reason_flags.append("movement_context_without_watch_brand")
    if movement_brand_contamination:
        penalty += 30
        reason_flags.append("movement_brand_contamination")

    return {
        "has_brand": has_brand,
        "movement_brand_contamination": movement_brand_contamination,
        "penalty": penalty,
        "reason_flags": reason_flags,
    }


def detect_discovery_family(text: str) -> str:
    t = norm(text)
    if "seiko" in t and any(x in t for x in ["chrono", "chronograph", "7t", "6138", "6139"]):
        return "seiko_chronograph"
    if "tissot" in t and any(x in t for x in ["chrono", "chronograph", "prs", "prc", "valjoux", "7750"]):
        return "tissot_chronograph"
    if any(x in t for x in ["valjoux", "lemania", "7750", "chronograph"]):
        return "valjoux_lemania_chronograph"
    if any(x in t for x in ["diver", "skin diver", "sub", "super compressor"]):
        return "vintage_diver"
    if "vintage" in t and "automatic" in t:
        return "vintage_automatic"
    if any(x in t for x in ["lcd", "alarm chrono", "digital chronograph"]):
        return "lcd_alarm_chronograph"
    return ""


def derive_match_confidence(match_score: int, kw_hits: int, ambiguity_penalty: int) -> int:
    conf = int(match_score * 0.75) + min(15, kw_hits * 4) - int(ambiguity_penalty * 0.4)
    return max(0, min(100, conf))
