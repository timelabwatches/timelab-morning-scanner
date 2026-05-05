"""
TIMELAB — comparables/shadow.py

Shadow-mode wrapper around the comparables engine. Wired into the analyst
pipeline AFTER the existing auction_price_engine so we can compare both
estimates side-by-side without affecting the actual decision flow.

This module does the dimension translation between:
  - analyzer's record dict  (English movement labels, lowercase brand,
                             listing_id, title/raw_text, etc.)
  - comparables engine API  (Spanish movement labels, canonical brand)

Public API:
    shadow_compare(record: dict) -> dict
        Returns the new estimate (so caller can record it for later eval),
        but does NOT mutate `record`.
        Logs INFO line per call. Never raises.

Use:
    # bridge/analyst_pipeline.py
    from comparables.shadow import shadow_compare
    ...
    analyzed = apply_auction_price_engine(analyzed)
    shadow_compare(analyzed)        # ← side-effect-free shadow
    analyzed = apply_profit_engine(analyzed)
"""

from __future__ import annotations

import logging
import sys

from .comparables_engine import estimate_hammer_catawiki
from .enrichers import (
    infer_model_family,
    infer_mechanism_from_refs,
    infer_mechanism_from_family,
)


# Self-configured logger so SHADOW lines reach stdout even when the host
# script (e.g. main.py) does not call logging.basicConfig().
logger = logging.getLogger("timelab.shadow")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False


# Map analyzer's English movement hints → DB Spanish labels
_MOVEMENT_MAP = {
    "automatic":  "Automático",
    "manual":     "Cuerda",
    "manual-wind":"Cuerda",
    "hand-wound": "Cuerda",
    "quartz":     "Cuarzo",
    "kinetic":    "Cuarzo",     # kinetic prices like quartz on Catawiki
    "solar":      "Cuarzo",
    "electronic": "Electrónico",
}

# Brand normalization: analyzer commonly emits lowercase / different spelling
_BRAND_NORMALIZE = {
    "tissot": "Tissot", "seiko": "Seiko", "longines": "Longines",
    "omega": "Omega", "hamilton": "Hamilton", "zenith": "Zenith",
    "tag heuer": "TAG Heuer", "tagheuer": "TAG Heuer", "tag-heuer": "TAG Heuer",
    "junghans": "Junghans", "certina": "Certina", "oris": "Oris",
    "rado": "Rado", "fortis": "Fortis", "cyma": "Cyma",
    "cauny": "Cauny", "citizen": "Citizen", "mido": "Mido",
    "mortima": "Mortima", "dogma": "Dogma",
    "favre-leuba": "Favre-Leuba", "favre leuba": "Favre-Leuba",
    "universal geneve": "Universal Genève",
    "universal genève": "Universal Genève",
    "maurice lacroix": "Maurice Lacroix",
    "baume": "Baume & Mercier",
    "baume & mercier": "Baume & Mercier",
}


def _norm_brand(b: str) -> str:
    if not b:
        return ""
    k = b.strip().lower()
    return _BRAND_NORMALIZE.get(k, b.strip().title())


def _build_text_for_family(record: dict, brand: str, model: str) -> str:
    """Combine the strongest signals to extract model_family from."""
    return " ".join(filter(None, [
        brand,
        model,
        record.get("title"),
        record.get("raw_text"),
        record.get("description"),
    ]))


def shadow_compare(record: dict) -> dict:
    """
    Run the new engine and log a side-by-side vs the existing
    `expected_hammer` field. Returns the new estimate dict. Never raises;
    on any error returns {"reason": "exception", "error": <msg>}.
    """
    try:
        brand = _norm_brand(record.get("brand", ""))
        if not brand:
            return {"reason": "no_brand"}

        # Translate movement
        mvh = (record.get("movement_hint") or "").lower().strip()
        mech = _MOVEMENT_MAP.get(mvh)
        mech_source = "analyzer" if mech else None

        # Chrono flag from watch_type
        watch_type = (record.get("watch_type") or "").lower()
        is_chrono = "chrono" in watch_type

        # Reference(s)
        reference = record.get("reference") or ""
        refs = [reference] if reference else []

        # Family extraction
        model = record.get("model") or ""
        text_for_family = _build_text_for_family(record, brand, model)
        family = infer_model_family(brand, text_for_family)

        # Fallback chain for mechanism — only fires if previous didn't resolve.
        # 1) ref-based (e.g. "7S26" → Automático)
        if not mech and refs:
            ref_mech = infer_mechanism_from_refs(refs, brand_hint=brand)
            if ref_mech:
                mech = ref_mech
                mech_source = "refs"
        # 2) family-based (data-driven first, then industry canonical)
        if not mech and family:
            fam_mech, fam_src = infer_mechanism_from_family(brand, family)
            if fam_mech:
                mech = fam_mech
                mech_source = f"family:{fam_src}"

        new = estimate_hammer_catawiki(
            brand=brand,
            mech=mech,
            is_chrono=is_chrono,
            refs=refs,
            model_family=family,
            auction_tier=None,    # destination auction tier unknown at scan time
        )

        # Compose log line. The old auction_price_engine writes into a sub-dict
        # `record["auction_estimate"]`, not into top-level fields, so we read
        # the old expected_hammer from there.
        item_id = record.get("listing_id") or record.get("id") or record.get("url") or "?"
        mech_label = f"{mech}({mech_source})" if mech else "None"
        old_estimate = record.get("auction_estimate") or {}
        old = old_estimate.get("expected_hammer")
        if old is not None and new.get("expected_hammer") is not None:
            diff = new["expected_hammer"] - old
            pct = (diff / old * 100) if old else 0
            logger.info(
                "[SHADOW] item=%s brand=%s mech=%s chrono=%s family=%s | "
                "old=%.0f€ new=%.0f€ diff=%+.0f€ (%+.1f%%) | "
                "L%s n=%d conf=%s bucket=%s",
                item_id, brand, mech_label, is_chrono, family,
                old, new["expected_hammer"], diff, pct,
                new.get("level_used"), new.get("n", 0),
                new.get("confidence"), new.get("source_bucket"),
            )
        elif new.get("expected_hammer") is not None:
            logger.info(
                "[SHADOW] item=%s brand=%s mech=%s chrono=%s family=%s | "
                "old=NONE new=%.0f€ | L%s n=%d conf=%s bucket=%s",
                item_id, brand, mech_label, is_chrono, family,
                new["expected_hammer"],
                new.get("level_used"), new.get("n", 0),
                new.get("confidence"), new.get("source_bucket"),
            )
        else:
            logger.info(
                "[SHADOW] item=%s brand=%s mech=%s chrono=%s family=%s | "
                "old=%s new=NONE reason=%s",
                item_id, brand, mech_label, is_chrono, family,
                f"{old:.0f}€" if old is not None else "NONE",
                new.get("reason"),
            )
        return new

    except Exception as e:
        logger.warning("[SHADOW] failed: %s", e, exc_info=False)
        return {"reason": "exception", "error": str(e)}


def apply_comparables_engine(record: dict) -> dict:
    """
    PHASE 2 (refined) — promote the comparables engine to actual decisor.

    Decision matrix:
      ┌──────────────────────────┬─────────────────────────────────────────┐
      │ Situation                │ Action                                  │
      ├──────────────────────────┼─────────────────────────────────────────┤
      │ new returns NONE         │ leave record alone (engine has no data) │
      │ legacy returned NONE     │ promote new value (was silent before)   │
      │ both have value, conf=H  │ OVERRIDE — trust real-cierre calibration│
      │ both have value, conf<H  │ defer to legacy (we don't know better)  │
      └──────────────────────────┴─────────────────────────────────────────┘

    Rationale for the high-confidence override: when our bucket has n≥8 real
    cierres, the median is empirically grounded in your historical sales. The
    legacy engine assigns category defaults from target_stats.json that are
    much coarser (e.g., 197€ for any Seiko Prospex, when our 18 cierres show
    p50=145€). At conf=high, prefer the empirical median.

    The promoted dict matches the field names profit_engine reads:
        expected_hammer, conservative_hammer, optimistic_hammer, raw_expected_hammer
    Plus audit fields: source, confidence, level_used, n, bucket, promote_reason,
    and legacy_value (preserves the original old_hammer for traceability).

    Mutates and returns `record`. Never raises.
    """
    new = shadow_compare(record)

    new_hammer = new.get("expected_hammer")
    if new_hammer is None:
        return record  # nothing we can offer

    old_estimate = record.get("auction_estimate") or {}
    old_hammer = old_estimate.get("expected_hammer")
    new_conf = new.get("confidence", "low")

    # Decide whether to write our value
    promote_reason = None
    if old_hammer is None:
        promote_reason = "legacy_returned_none"
    elif new_conf == "high":
        promote_reason = "high_confidence_override"

    if promote_reason is None:
        return record  # defer to legacy

    record["auction_estimate"] = {
        "expected_hammer":     new_hammer,
        "conservative_hammer": new.get("conservative", new_hammer),
        "optimistic_hammer":   new.get("optimistic", new_hammer),
        "raw_expected_hammer": new.get("raw_p50", new_hammer),
        # ↓ Contract fields the decision_engine reads directly. Without these,
        # decision_engine emits 'pass' with reason='no_pricing_data', which
        # silently downgrades any candidate that we override.
        "price_estimate_available": True,
        "price_confidence":         new_conf,   # 'high' | 'medium' | 'low'
        # Audit / provenance (non-functional for decision_engine)
        "source":         "comparables_engine",
        "confidence":     new_conf,
        "level_used":     new.get("level_used"),
        "n":              new.get("n"),
        "bucket":         new.get("source_bucket"),
        "promote_reason": promote_reason,
        "legacy_value":   old_hammer,
    }

    item_id = record.get("listing_id") or record.get("id") or "?"
    if promote_reason == "high_confidence_override":
        logger.info(
            "[COMPARABLES] item=%s OVERRIDE old=%.0f€ → new=%.0f€ "
            "conf=high L%s n=%d bucket=%s",
            item_id, old_hammer, new_hammer,
            new.get("level_used"), new.get("n", 0), new.get("source_bucket"),
        )
    else:
        logger.info(
            "[COMPARABLES] item=%s PROMOTED expected_hammer=%.0f€ p25=%.0f€ p75=%.0f€ "
            "conf=%s L%s n=%d (legacy was None)",
            item_id, new_hammer,
            new.get("conservative", new_hammer),
            new.get("optimistic", new_hammer),
            new_conf, new.get("level_used"), new.get("n", 0),
        )
    return record
