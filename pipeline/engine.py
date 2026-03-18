# pipeline/engine.py

from bridge.analyst_adapter import ebay_candidate_to_record
from bridge.analyst_pipeline import analyze_record
from pipeline.filters import reject_reason


def evaluate_candidate(
    candidate: dict,
    comparables: list[dict],
    settings,
) -> dict | None:
    """
    Evaluate one eBay candidate using the TIMELAB analyst pipeline.
    """

    raw_text = candidate.get("raw_text", "")
    location = candidate.get("location", "")

    reason = reject_reason(
        text=raw_text,
        location_text=location,
        eu_only=True,
    )
    if reason is not None:
        return None

    record = ebay_candidate_to_record(candidate)
    analyzed = analyze_record(record)

    return analyzed


def passes_decision_gate(result: dict, settings) -> bool:
    """
    Decide whether an analyzed record should be sent to Telegram.
    """

    decision = result.get("decision")
    expected_case = ((result.get("economics") or {}).get("expected_case") or {})
    net_profit = expected_case.get("net_profit")
    roi = expected_case.get("roi")

    if decision not in {"strong_buy", "buy", "review"}:
        return False

    if net_profit is None or roi is None:
        return False

    if net_profit < settings.min_net_eur:
        return False

    if roi < settings.min_net_roi:
        return False

    return True


def build_alert_payload(result: dict) -> dict:
    """
    Convert analyzed record into Telegram-friendly payload.
    """

    economics = result.get("economics") or {}
    expected_case = economics.get("expected_case") or {}
    auction_estimate = result.get("auction_estimate") or {}

    return {
        "title": result.get("title", ""),
        "price": float(result.get("price", 0.0) or 0.0),
        "shipping": float(result.get("shipping", 0.0) or 0.0),
        "est_close": float(auction_estimate.get("expected_hammer", 0.0) or 0.0),
        "net_profit": float(expected_case.get("net_profit", 0.0) or 0.0),
        "roi": float(expected_case.get("roi", 0.0) or 0.0),
        "target_id": result.get("reference") or result.get("model") or result.get("brand") or "",
        "match_score": int(result.get("score", 0) or 0),
        "match_band": result.get("candidate_class", "weak"),
        "sample_size": int((((result.get("reference_kb_data") or {}).get("price_stats") or {}).get("count", 0)) or 0),
        "p50": float((((result.get("reference_kb_data") or {}).get("price_stats") or {}).get("p50", 0.0)) or 0.0),
        "p75": float((((result.get("reference_kb_data") or {}).get("price_stats") or {}).get("p75", 0.0)) or 0.0),
        "stats_confidence": auction_estimate.get("price_confidence", "low"),
        "location": result.get("location", ""),
        "condition": result.get("condition_text", "") or result.get("condition", ""),
        "category": result.get("category_id", ""),
        "url": result.get("url", ""),
        "decision": result.get("decision", "pass"),
        "decision_reason": result.get("decision_reason", ""),
        "brand": result.get("brand", ""),
        "model": result.get("model", ""),
        "reference": result.get("reference", ""),
        "watch_type": result.get("watch_type", ""),
        "movement_hint": result.get("movement_hint", ""),
        "reference_kb_hit": bool(result.get("reference_kb_hit")),
    }