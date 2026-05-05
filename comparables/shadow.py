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
from .enrichers import infer_model_family, infer_mechanism_from_refs


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

        # Upgrade unknown mech via ref lookup
        if not mech and refs:
            mech = infer_mechanism_from_refs(refs, brand_hint=brand)

        new = estimate_hammer_catawiki(
            brand=brand,
            mech=mech,
            is_chrono=is_chrono,
            refs=refs,
            model_family=family,
            auction_tier=None,    # destination auction tier unknown at scan time
        )

        # Compose log line
        item_id = record.get("listing_id") or record.get("id") or record.get("url") or "?"
        old = record.get("expected_hammer")
        if old is not None and new.get("expected_hammer") is not None:
            diff = new["expected_hammer"] - old
            pct = (diff / old * 100) if old else 0
            logger.info(
                "[SHADOW] item=%s brand=%s mech=%s chrono=%s family=%s | "
                "old=%.0f€ new=%.0f€ diff=%+.0f€ (%+.1f%%) | "
                "L%s n=%d conf=%s bucket=%s",
                item_id, brand, mech, is_chrono, family,
                old, new["expected_hammer"], diff, pct,
                new.get("level_used"), new.get("n", 0),
                new.get("confidence"), new.get("source_bucket"),
            )
        elif new.get("expected_hammer") is not None:
            logger.info(
                "[SHADOW] item=%s brand=%s mech=%s chrono=%s family=%s | "
                "old=NONE new=%.0f€ | L%s n=%d conf=%s bucket=%s",
                item_id, brand, mech, is_chrono, family,
                new["expected_hammer"],
                new.get("level_used"), new.get("n", 0),
                new.get("confidence"), new.get("source_bucket"),
            )
        else:
            logger.info(
                "[SHADOW] item=%s brand=%s mech=%s chrono=%s family=%s | "
                "old=%s new=NONE reason=%s",
                item_id, brand, mech, is_chrono, family,
                f"{old:.0f}€" if old is not None else "NONE",
                new.get("reason"),
            )
        return new

    except Exception as e:
        logger.warning("[SHADOW] failed: %s", e, exc_info=False)
        return {"reason": "exception", "error": str(e)}
