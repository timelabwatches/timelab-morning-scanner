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


# ──────────────────────────────────────────────────────────────────────────────
# Override guardrail
# ──────────────────────────────────────────────────────────────────────────────
# Block any override that would cut the legacy estimate by more than 50%.
#
# Rationale: in the CC and secondhand pipelines the legacy value comes from
# `target_list.json["catawiki_estimate"].p50`, which is **curated by hand per
# specific model** (e.g., seiko-prospex-turtle has a per-model p50 ~500€).
# Our comparables engine groups at brand×mech (L3), so for a heterogeneous
# brand like Seiko the L3 median (~145€) mixes Seiko 5 / Presage / Prospex
# and dragging a Prospex curated at 500€ down to 145€ destroys real per-model
# information.
#
# Asymmetric on purpose: only blocks DOWNWARD overrides. Upward overrides
# remain safe because they correct generic legacy defaults using buckets
# calibrated against real cierres (n≥8 unanimous).
#
# Threshold 0.5 chosen empirically: catastrophic Seiko Prospex cases observed
# at ratio ~0.30-0.34; legitimate corrections (Hamilton Ventura 529→340 = 0.64)
# remain above the threshold.
MIN_DOWNWARD_OVERRIDE_RATIO = 0.5


def _should_block_downward_override(legacy, new, conf="low", n=0) -> bool:
    """
    True when the new value is so much lower than legacy that overriding
    would likely discard valuable per-model curation. See above.

    EXCEPTION: when our motor has both conf=high AND n≥12 cierres real,
    we trust it over the legacy curated value. The original guardrail
    protected against coarse buckets (e.g., Seiko Auto L3 with n=18 mixing
    Seiko 5 / Presage / Prospex tiers) overriding fine-grained target
    curation. But buckets like Tissot Cuarzo No-chrono L3 (n=16, unanimous,
    all real cierres of the same category) are MORE accurate than a
    static catawiki_estimate.p50 set by hand. The asymmetric block was
    too eager — we'd rather let strongly-evidenced overrides through
    even when they correct legacy downward.
    """
    if not legacy or legacy <= 0 or not new or new <= 0:
        return False
    if (new / legacy) >= MIN_DOWNWARD_OVERRIDE_RATIO:
        return False  # not a significant downward; allow override
    # Below 0.5x ratio: only block when motor's confidence is weak.
    # When conf=high AND we have ≥12 real cierres, trust the empirical
    # value even if it's much lower than legacy.
    if conf == "high" and n >= 12:
        return False  # trust motor's empirical evidence
    return True  # block — motor doesn't have strong-enough evidence
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


def apply_comparables_engine_cc(
    legacy_close,
    listing,
    target=None,
    vision_data=None,
    detected_movement=None,
):
    """
    PHASE 2 wrapper for the CashConverters scanner.

    Same decision matrix as `apply_comparables_engine` (the eBay one) but
    adapted to CC's flow where:
      - There is no `auction_estimate` dict; CC passes a raw float around.
      - Inputs come from the Listing dataclass + matched target + vision data.
      - Returns the close estimate to use (legacy or new).

    Decision logic:
      - new returns NONE                       → keep legacy
      - legacy is None / 0                     → use new (promote)
      - both have value AND conf=high          → use new (override)
      - both have value AND conf < high        → keep legacy

    Logs via the same `[SHADOW]` line as eBay (diff legacy vs new) plus a
    `[COMPARABLES-CC]` action line when we override or promote. Never raises;
    on any error returns `legacy_close` unchanged.

    Returns: float (close estimate to use) — or None if both engines fail.
    """
    try:
        title = getattr(listing, "title", "") or ""
        cond  = getattr(listing, "cond", "") or ""
        desc  = getattr(listing, "description", "") or ""
        url   = getattr(listing, "url", "") or "?"

        target = target or {}
        vision = vision_data or {}

        # Brand: prefer the matched target's value, fall back to vision
        brand = target.get("brand") or vision.get("brand") or ""

        # Reference: from vision (CC's analyzer doesn't extract refs reliably)
        reference = vision.get("reference") or ""

        # Movement: detected_movement (CC's text-based detector, English) is most reliable.
        # Skip "unknown" sentinel and fall back to vision.
        mvh_raw = (
            detected_movement
            if detected_movement and detected_movement != "unknown"
            else (vision.get("movement") or "")
        )

        # Chrono detection from title (CC has no `watch_type` field like the analyzer)
        title_l = title.lower()
        is_chrono = any(
            t in title_l for t in ("chrono", "chronograph", "cronograph", "cronograf")
        )

        item_id = url.rsplit("/", 1)[-1] if "/" in url else url[:50]

        # Build a record dict shaped like what shadow_compare expects
        record = {
            "listing_id": item_id,
            "brand":         brand.lower() if brand else "",
            "model":         (vision.get("model") or "").lower(),
            "reference":     reference,
            "movement_hint": mvh_raw.lower() if mvh_raw else None,
            "watch_type":    "chronograph" if is_chrono else "",
            "title":         title,
            "raw_text":      f"{title} {cond} {desc}".strip()[:1000],
            # Inject legacy estimate so shadow_compare can log diff vs new
            "auction_estimate": (
                {"expected_hammer": float(legacy_close)}
                if legacy_close and legacy_close > 0 else {}
            ),
        }

        new = shadow_compare(record)
        new_hammer = new.get("expected_hammer")

        if new_hammer is None:
            return legacy_close  # nothing better to offer

        new_conf = new.get("confidence", "low")

        # Promote when legacy returned no value
        if not legacy_close or legacy_close <= 0:
            logger.info(
                "[COMPARABLES-CC] item=%s PROMOTED expected_hammer=%.0f€ conf=%s "
                "L%s n=%d bucket=%s",
                item_id, new_hammer, new_conf,
                new.get("level_used"), new.get("n", 0), new.get("source_bucket"),
            )
            return new_hammer

        # Override when our confidence is high
        if new_conf == "high":
            if _should_block_downward_override(legacy_close, new_hammer, conf=new_conf, n=new.get('n', 0)):
                logger.info(
                    "[COMPARABLES-CC] item=%s SKIP_OVERRIDE_DOWNWARD ratio=%.2f "
                    "old=%.0f€ new=%.0f€ (legacy preserved)",
                    item_id, new_hammer / float(legacy_close),
                    float(legacy_close), new_hammer,
                )
                return legacy_close
            logger.info(
                "[COMPARABLES-CC] item=%s OVERRIDE old=%.0f€ → new=%.0f€ conf=high "
                "L%s n=%d bucket=%s",
                item_id, float(legacy_close), new_hammer,
                new.get("level_used"), new.get("n", 0), new.get("source_bucket"),
            )
            return new_hammer

        # Otherwise defer to the curated target value
        return legacy_close

    except Exception as e:
        logger.warning("[COMPARABLES-CC] failed: %s", e, exc_info=False)
        return legacy_close  # safe fallback — never break the CC pipeline


def apply_comparables_engine_secondhand(
    legacy_close,
    listing,
    target=None,
    movement=None,
    brand_hint=None,
):
    """
    PHASE 2 wrapper for the secondhand scanners (Bilbotruke, RealCash, Locotoo)
    via `scanners/secondhand_core.py`.

    Same decision matrix as the eBay and CC variants:
      - new returns NONE                       → keep legacy
      - legacy is None / 0                     → use new (promote)
      - both have value AND conf=high          → use new (override)
      - both have value AND conf < high        → keep legacy

    The secondhand pipeline differs from CC in two relevant ways:
      - No Claude Vision data (no `vision_data` arg).
      - Movement is already detected by the core's `detect_movement(title, raw_text)`
        and passed in here directly (English: "automatic"/"quartz"/"manual"/"unknown").

    Listing schema expected (from secondhand_core.Listing dataclass):
        .title, .raw_text, .price_eur, .cond, .url, .sku

    Logs via the same `[SHADOW]` line as the others, plus a `[COMPARABLES-2H]`
    action line when we override or promote. Never raises; on any error returns
    `legacy_close` unchanged.

    Returns: float (close estimate to use) — or None if both engines fail.
    """
    try:
        title    = getattr(listing, "title", "") or ""
        raw_text = getattr(listing, "raw_text", "") or ""
        cond     = getattr(listing, "cond", "") or ""
        url      = getattr(listing, "url", "") or ""
        sku      = getattr(listing, "sku", "") or ""

        target = target or {}
        # Brand: prefer matched target's value, then explicit hint, else nothing
        brand = (target.get("brand") or brand_hint or "").lower()

        # Movement (already detected English label from detect_movement())
        mvh = (
            movement.lower()
            if movement and movement.lower() != "unknown"
            else None
        )

        # Chrono detection from title (these scanners have no `watch_type` field)
        title_l = title.lower()
        is_chrono = any(
            t in title_l for t in ("chrono", "chronograph", "cronograph", "cronograf")
        )

        # Item id for logging — sku is most stable; fall back to URL tail
        item_id = sku or (url.rsplit("/", 1)[-1] if "/" in url else url[:50]) or "?"

        # Build a record dict shaped like what shadow_compare expects
        record = {
            "listing_id":    item_id,
            "brand":         brand,
            "model":         "",                # secondhand core doesn't extract model
            "reference":     "",                # nor reference
            "movement_hint": mvh,
            "watch_type":    "chronograph" if is_chrono else "",
            "title":         title,
            "raw_text":      f"{title} {cond} {raw_text}".strip()[:1000],
            # Inject legacy as auction_estimate so shadow_compare logs the diff
            "auction_estimate": (
                {"expected_hammer": float(legacy_close)}
                if legacy_close and legacy_close > 0 else {}
            ),
        }

        new = shadow_compare(record)
        new_hammer = new.get("expected_hammer")

        if new_hammer is None:
            return legacy_close  # nothing better to offer

        new_conf = new.get("confidence", "low")

        # Promote when legacy returned no value
        if not legacy_close or legacy_close <= 0:
            logger.info(
                "[COMPARABLES-2H] item=%s PROMOTED expected_hammer=%.0f€ conf=%s "
                "L%s n=%d bucket=%s",
                item_id, new_hammer, new_conf,
                new.get("level_used"), new.get("n", 0), new.get("source_bucket"),
            )
            return new_hammer

        # Override when our confidence is high
        if new_conf == "high":
            if _should_block_downward_override(legacy_close, new_hammer, conf=new_conf, n=new.get('n', 0)):
                logger.info(
                    "[COMPARABLES-2H] item=%s SKIP_OVERRIDE_DOWNWARD ratio=%.2f "
                    "old=%.0f€ new=%.0f€ (legacy preserved)",
                    item_id, new_hammer / float(legacy_close),
                    float(legacy_close), new_hammer,
                )
                return legacy_close
            logger.info(
                "[COMPARABLES-2H] item=%s OVERRIDE old=%.0f€ → new=%.0f€ conf=high "
                "L%s n=%d bucket=%s",
                item_id, float(legacy_close), new_hammer,
                new.get("level_used"), new.get("n", 0), new.get("source_bucket"),
            )
            return new_hammer

        # Otherwise defer to the curated target value
        return legacy_close

    except Exception as e:
        logger.warning("[COMPARABLES-2H] failed: %s", e, exc_info=False)
        return legacy_close  # safe fallback — never break the secondhand pipeline


def apply_comparables_engine_vinted(
    legacy_close,
    listing,
    target=None,
    brand_hint=None,
):
    """
    PHASE 2 wrapper for the Vinted scanner.

    Same decision matrix as the eBay / CC / secondhand variants:
      - new returns NONE                       → keep legacy
      - legacy is None / 0                     → use new (promote)
      - both have value AND conf=high          → use new (override)
      - both have value AND conf < high        → keep legacy
    Plus the asymmetric downward guardrail (block override when new/old < 0.5)
    to protect target-curated values from being dragged down by coarse buckets.

    Vinted's Listing schema differs from CC/secondhand:
      - .item_id (not .sku)
      - .status (not .cond)
      - No vision_data (Vinted scanner doesn't run Claude Vision)
      - No detected_movement (we infer from title text via the engine itself)

    Logs via [SHADOW] (diff legacy vs new) plus [COMPARABLES-VT] action lines.
    Never raises; on any error returns legacy_close unchanged.
    """
    try:
        title    = getattr(listing, "title", "") or ""
        raw_text = getattr(listing, "raw_text", "") or ""
        status   = getattr(listing, "status", "") or ""
        url      = getattr(listing, "url", "") or ""
        item_id  = getattr(listing, "item_id", "") or ""

        target = target or {}
        # Brand: prefer matched target's, then explicit hint
        brand = (target.get("brand") or brand_hint or "").lower()

        # Chrono detection from title (Vinted has no watch_type field)
        title_l = title.lower()
        is_chrono = any(
            t in title_l for t in ("chrono", "chronograph", "cronograph", "cronograf")
        )

        # ID for logging — item_id is most stable; fall back to URL tail
        log_id = item_id or (url.rsplit("/", 1)[-1] if "/" in url else url[:50]) or "?"

        # Build a record dict shaped like what shadow_compare expects
        record = {
            "listing_id":    log_id,
            "brand":         brand,
            "model":         "",                # Vinted doesn't extract model
            "reference":     "",                # nor reference
            "movement_hint": None,              # let engine infer from text
            "watch_type":    "chronograph" if is_chrono else "",
            "title":         title,
            "raw_text":      f"{title} {status} {raw_text}".strip()[:1000],
            "auction_estimate": (
                {"expected_hammer": float(legacy_close)}
                if legacy_close and legacy_close > 0 else {}
            ),
        }

        new = shadow_compare(record)
        new_hammer = new.get("expected_hammer")

        if new_hammer is None:
            return legacy_close

        new_conf = new.get("confidence", "low")

        # Promote when legacy returned no value
        if not legacy_close or legacy_close <= 0:
            logger.info(
                "[COMPARABLES-VT] item=%s PROMOTED expected_hammer=%.0f€ conf=%s "
                "L%s n=%d bucket=%s",
                log_id, new_hammer, new_conf,
                new.get("level_used"), new.get("n", 0), new.get("source_bucket"),
            )
            return new_hammer

        # Override when our confidence is high — but only if not catastrophically lower
        if new_conf == "high":
            if _should_block_downward_override(legacy_close, new_hammer, conf=new_conf, n=new.get('n', 0)):
                logger.info(
                    "[COMPARABLES-VT] item=%s SKIP_OVERRIDE_DOWNWARD ratio=%.2f "
                    "old=%.0f€ new=%.0f€ (legacy preserved)",
                    log_id, new_hammer / float(legacy_close),
                    float(legacy_close), new_hammer,
                )
                return legacy_close
            logger.info(
                "[COMPARABLES-VT] item=%s OVERRIDE old=%.0f€ → new=%.0f€ conf=high "
                "L%s n=%d bucket=%s",
                log_id, float(legacy_close), new_hammer,
                new.get("level_used"), new.get("n", 0), new.get("source_bucket"),
            )
            return new_hammer

        # Otherwise defer to the curated target value
        return legacy_close

    except Exception as e:
        logger.warning("[COMPARABLES-VT] failed: %s", e, exc_info=False)
        return legacy_close


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
        if _should_block_downward_override(old_hammer, new_hammer, conf=new_conf, n=new.get('n', 0)):
            item_id = record.get("listing_id") or record.get("id") or "?"
            logger.info(
                "[COMPARABLES] item=%s SKIP_OVERRIDE_DOWNWARD ratio=%.2f "
                "old=%.0f€ new=%.0f€ (legacy preserved)",
                item_id, new_hammer / old_hammer, old_hammer, new_hammer,
            )
            return record
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
