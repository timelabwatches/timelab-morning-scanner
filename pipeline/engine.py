from pipeline.comparables import get_target_stats
from pipeline.filters import reject_reason
from pipeline.knowledge_base import resolve_listing_identity
from pipeline.valuation import estimate_close_price, estimate_net_profit


def passes_identity_gate(identity: dict, min_match_score: int) -> bool:
    score = int(identity.get("match_score", 0) or 0)
    band = str(identity.get("match_confidence_band", "very_low"))
    ambiguity = bool(identity.get("ambiguity", True))

    if score < min_match_score:
        return False

    if band not in {"high", "medium"}:
        return False

    if ambiguity:
        return False

    return True


def choose_close_estimate(listing_text: str, listing_price: float, stats: dict) -> float:
    p50 = float(stats.get("p50", 0.0) or 0.0)
    p75 = float(stats.get("p75", 0.0) or 0.0)
    sample_size = int(stats.get("sample_size", 0) or 0)

    if sample_size <= 0:
        return round(max(listing_price * 1.30, listing_price + 70), 2)

    return estimate_close_price(
        price_eur=listing_price,
        target_p50=p50,
        target_p75=p75,
        listing_text=listing_text,
    )


def evaluate_candidate(
    candidate: dict,
    models: list[dict],
    comparables: list[dict],
    settings,
) -> dict | None:
    raw_text = candidate.get("raw_text", "")
    location = candidate.get("location", "")

    reason = reject_reason(
        text=raw_text,
        location_text=location,
        eu_only=True,
    )
    if reason is not None:
        return None

    identity = resolve_listing_identity(raw_text, models)
    target_id = identity.get("target_id")
    if not target_id:
        return None

    stats = get_target_stats(target_id, comparables)

    est_close = choose_close_estimate(
        listing_text=raw_text,
        listing_price=float(candidate.get("price", 0.0) or 0.0),
        stats=stats,
    )

    net_profit, roi = estimate_net_profit(
        buy_price=float(candidate.get("price", 0.0) or 0.0),
        shipping_cost=float(candidate.get("shipping", 0.0) or 0.0),
        estimated_close=est_close,
        catwiki_commission=settings.catwiki_commission,
        payment_processing=settings.payment_processing,
        packaging_eur=settings.packaging_eur,
        misc_eur=settings.misc_eur,
        ship_arbitrage_eur=settings.ship_arbitrage_eur,
        effective_tax_rate_on_profit=settings.effective_tax_rate_on_profit,
    )

    return {
        "source": candidate.get("source", ""),
        "listing_id": candidate.get("listing_id", ""),
        "title": candidate.get("title", ""),
        "description": candidate.get("description", ""),
        "condition_text": candidate.get("condition_text", ""),
        "price": float(candidate.get("price", 0.0) or 0.0),
        "shipping": float(candidate.get("shipping", 0.0) or 0.0),
        "location": candidate.get("location", ""),
        "url": candidate.get("url", ""),
        "category_id": candidate.get("category_id", ""),
        "raw_text": raw_text,
        "identity": identity,
        "stats": stats,
        "est_close": est_close,
        "net_profit": net_profit,
        "roi": roi,
    }


def passes_decision_gate(result: dict, settings) -> bool:
    identity = result.get("identity", {})
    stats = result.get("stats", {})

    if not passes_identity_gate(identity, settings.min_match_score):
        return False

    sample_size = int(stats.get("sample_size", 0) or 0)
    if sample_size < 2:
        return False

    net_profit = float(result.get("net_profit", 0.0) or 0.0)
    roi = float(result.get("roi", 0.0) or 0.0)

    if net_profit < settings.min_net_eur:
        return False

    if roi < settings.min_net_roi:
        return False

    return True


def build_alert_payload(result: dict) -> dict:
    identity = result.get("identity", {})
    stats = result.get("stats", {})

    return {
        "title": result.get("title", ""),
        "price": float(result.get("price", 0.0) or 0.0),
        "shipping": float(result.get("shipping", 0.0) or 0.0),
        "est_close": float(result.get("est_close", 0.0) or 0.0),
        "net_profit": float(result.get("net_profit", 0.0) or 0.0),
        "roi": float(result.get("roi", 0.0) or 0.0),
        "target_id": identity.get("target_id", ""),
        "match_score": int(identity.get("match_score", 0) or 0),
        "match_band": identity.get("match_confidence_band", "very_low"),
        "sample_size": int(stats.get("sample_size", 0) or 0),
        "p50": float(stats.get("p50", 0.0) or 0.0),
        "p75": float(stats.get("p75", 0.0) or 0.0),
        "stats_confidence": int(stats.get("confidence_score", 0) or 0),
        "location": result.get("location", ""),
        "condition": result.get("condition_text", ""),
        "category": result.get("category_id", ""),
        "url": result.get("url", ""),
    }