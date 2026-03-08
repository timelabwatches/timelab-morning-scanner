import re
from typing import Dict, Iterable, Tuple


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def extract_listing_flags(text: str) -> Dict[str, bool]:
    t = norm(text)
    has_box = any(x in t for x in ["box", "caja", "scatola"])
    has_papers = any(x in t for x in ["papers", "papeles", "documenti", "garantia", "garantía", "warranty"])
    has_service = any(x in t for x in ["serviced", "service", "revisado", "revisión", "revision"])
    is_nos = any(x in t for x in ["nos", "new old stock", "unworn"])
    is_full_set = has_box and has_papers or any(x in t for x in ["full set", "full kit", "set completo"])
    return {
        "has_box": has_box,
        "has_papers": has_papers,
        "has_service": has_service,
        "is_nos": is_nos,
        "is_full_set": is_full_set,
    }


def condition_multiplier(condition_score: int) -> float:
    if condition_score >= 15:
        return 1.05
    if condition_score >= 5:
        return 1.03
    if condition_score <= -35:
        return 0.82
    if condition_score <= -20:
        return 0.90
    return 1.0


def should_use_p75(text: str, condition_score: int, triggers: Iterable[str]) -> bool:
    if condition_score < 10:
        return False
    t = norm(text)
    return any(norm(tr) in t for tr in (triggers or []))


def estimate_close_price(
    p50: float,
    p75: float,
    detail_text: str,
    condition_score: int,
    haircut: float,
    triggers: Iterable[str],
) -> Tuple[float, Dict[str, bool]]:
    flags = extract_listing_flags(detail_text)
    base = max(0.0, p50) * haircut
    if p75 > p50 and should_use_p75(detail_text, condition_score, triggers):
        base = p75 * haircut
    base *= condition_multiplier(condition_score)
    if flags["is_full_set"]:
        base *= 1.02
    if flags["is_nos"]:
        base *= 1.03
    return round(base, 2), flags


def liquidity_score(liquidity: str) -> int:
    v = norm(liquidity)
    if v in {"high", "very high", "medium-high"}:
        return 10
    if v in {"medium", "medium high", "medium-low"}:
        return 5
    return 2


def brand_score(risk: str) -> int:
    r = norm(risk)
    if r == "low":
        return 12
    if r == "medium":
        return 7
    return 2


def compute_confidence(
    price_confidence: int,
    match_confidence: int,
    close_estimate_confidence: int,
) -> Dict[str, int]:
    return {
        "price_confidence": max(0, min(100, int(price_confidence))),
        "match_confidence": max(0, min(100, int(match_confidence))),
        "close_estimate_confidence": max(0, min(100, int(close_estimate_confidence))),
    }


def derive_close_estimate_confidence(flags: Dict[str, bool], used_p75: bool, condition_score: int) -> int:
    conf = 58
    if used_p75:
        conf += 6
    if flags.get("is_full_set"):
        conf += 8
    if flags.get("has_service"):
        conf += 8
    if flags.get("is_nos"):
        conf += 7
    if condition_score < -20:
        conf -= 18
    return max(20, min(95, conf))


def compute_opportunity_score(
    net: float,
    roi: float,
    match_score: int,
    condition_score: int,
    brand_points: int,
    liquidity_points: int,
    ambiguity_penalty: int,
) -> int:
    roi_pct = roi * 100.0
    raw = (
        min(40, max(0, int(net / 8)))
        + min(22, max(0, int(roi_pct / 2.2)))
        + min(20, max(0, int(match_score / 5)))
        + min(10, max(0, int((condition_score + 50) / 10)))
        + brand_points
        + liquidity_points
        - min(35, ambiguity_penalty)
    )
    return max(0, min(100, raw))


def bucket_from_score(score: int, is_generic: bool, discovery: bool = False) -> str:
    if discovery:
        return "DISCOVERY_REVIEW"
    if score >= 80 and not is_generic:
        return "SUPER_BUY"
    if score >= 62 and not is_generic:
        return "BUY"
    if score >= 45:
        return "REVIEW"
    return "DISCOVERY_REVIEW"


def explain_bucket(score: int, bucket: str, net: float, roi: float, match_score: int, ambiguity_penalty: int) -> str:
    return (
        f"bucket={bucket} score={score} via net={net:.2f}, roi={roi*100:.1f}%, "
        f"match={match_score}, ambiguity_penalty={ambiguity_penalty}"
    )