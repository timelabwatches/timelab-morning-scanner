"""
TIMELAB Atelier client.

Sends scraped offers from any scanner (Wallapop, Vinted, eBay, CashConverters,
etc.) to the TIMELAB Atelier via Apps Script webhook.

The Atelier is the secondary destination — Telegram remains the primary alert
channel. This client is fail-safe: if the webhook is down, misconfigured, or
returns an error, the function returns silently without breaking the scanner.

Configuration (env vars):
    ATELIER_WEBHOOK_URL  Apps Script /exec URL
    ATELIER_TOKEN        Shared secret token
"""
import os
import requests
from typing import Any


ATELIER_TIMEOUT_SEC = 10


def _safe_get(obj: Any, key: str, default=None):
    """Get a value from either a dict or an object with attributes."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_offer(raw: dict, source: str) -> dict:
    """
    Normalize a scanner-specific offer dict into the Atelier schema.

    Each scanner has its own structure. This function tries common shapes
    and falls back to safe defaults. Returns a dict ready to POST.
    """
    listing = raw.get("listing")  # Wallapop/Vinted style
    chrono24 = raw.get("chrono24", {}) or {}

    # Try multiple shapes
    title       = _safe_get(listing, "title", "") or raw.get("title", "")
    url         = _safe_get(listing, "url", "")   or raw.get("url", "")
    price_eur   = _safe_get(listing, "price_eur", 0) or raw.get("price", 0) or raw.get("buy_price", 0)
    photo_url   = _safe_get(listing, "photo_url", "") or raw.get("photo_url", "")
    location    = _safe_get(listing, "location", "") or raw.get("location", "")
    condition   = _safe_get(listing, "cond", "") or raw.get("condition", "")
    item_id     = _safe_get(listing, "item_id", "") or raw.get("item_id", "") or raw.get("id", "")

    # Watch identification (built by analyzer pipeline)
    brand       = raw.get("brand", "") or _safe_get(listing, "brand", "")
    model       = raw.get("model", "") or _safe_get(listing, "model", "")
    reference   = raw.get("reference", "") or raw.get("ref", "")

    # Economics
    est_close   = raw.get("close_est", 0) or raw.get("est_close", 0)
    net_profit  = raw.get("net", 0) or raw.get("net_profit", 0)
    roi         = raw.get("roi", 0)
    score       = raw.get("score", 0) or raw.get("match_score", 0)
    decision    = raw.get("bucket", "") or raw.get("decision", "")
    decision_reason = raw.get("decision_reason", "") or _safe_get(raw.get("explain", {}) or {}, "bucket_reason", "")

    # Vision verdict (only on enriched alerts)
    vision = raw.get("vision")
    vision_verdict = ""
    if vision is not None:
        vision_verdict = _safe_get(vision, "verdict", "") or ""

    # Chrono24 alternative if scanner suggested it
    c24_close   = chrono24.get("close_est", 0)
    c24_net     = chrono24.get("net", 0)
    c24_roi     = chrono24.get("roi", 0)
    c24_better  = bool(chrono24.get("suggested", False))

    return {
        "id":              f"{source.lower()}_{item_id}" if item_id else "",
        "source":          source,
        "url":             url,
        "title":           title[:200],
        "brand":           brand,
        "model":           model,
        "reference":       reference,
        "price_eur":       float(price_eur or 0),
        "photo_url":       photo_url,
        "location":        location,
        "condition":       condition,
        "est_close":       float(est_close or 0),
        "net_profit":      float(net_profit or 0),
        "roi":             float(roi or 0),
        "score":           int(score or 0),
        "decision":        decision,
        "decision_reason": decision_reason[:200],
        "vision_verdict":  vision_verdict,
        "c24_close":       float(c24_close or 0),
        "c24_net":         float(c24_net or 0),
        "c24_roi":         float(c24_roi or 0),
        "c24_better":      c24_better,
    }


def send_offers_to_atelier(offers: list, source: str) -> None:
    """
    Send a list of scanner offers to the Atelier webhook.

    Args:
        offers: list of offer dicts (scanner-specific shape)
        source: "Wallapop" | "Vinted" | "eBay" | "CashConverters" | etc.

    Returns silently on any failure. Telegram delivery is unaffected.
    """
    webhook_url = os.getenv("ATELIER_WEBHOOK_URL", "").strip()
    token       = os.getenv("ATELIER_TOKEN", "").strip()

    if not webhook_url or not token:
        return  # Not configured — silently skip

    if not offers:
        return

    payload = {
        "action": "add_offers",
        "token":  token,
        "source": source,
        "offers": [_normalize_offer(o, source) for o in offers],
    }

    try:
        response = requests.post(
            webhook_url,
            json=payload,
            timeout=ATELIER_TIMEOUT_SEC,
        )
        # Log non-OK responses but never raise
        if response.status_code != 200:
            print(
                f"[ATELIER] non-200 response {response.status_code}: "
                f"{response.text[:200]}",
                flush=True,
            )
        else:
            print(
                f"[ATELIER] sent {len(offers)} offers from {source}",
                flush=True,
            )
    except Exception as exc:
        # Never break the scanner because of an Atelier failure
        print(
            f"[ATELIER] send failed: {type(exc).__name__}: {str(exc)[:150]}",
            flush=True,
        )
