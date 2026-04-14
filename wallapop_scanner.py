#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIMELAB — Wallapop ES scanner

Scans Wallapop's search API for watch opportunities.

Env vars:
  TELEGRAM_BOT_TOKEN  (required)
  TELEGRAM_CHAT_ID    (required)
  WP_MAX_ITEMS        (default 200)   max listings to evaluate
  WP_THROTTLE_S       (default 0.5)   delay between requests
  WP_TIMEOUT          (default 20)    HTTP timeout
  WP_MIN_MATCH        (default 60)    min match score
  WP_MIN_NET_EUR      (default 20)    min net profit euros
  WP_MIN_NET_ROI      (default 0.10)  min ROI
  WP_CLOSE_HAIRCUT    (default 0.88)  haircut on p50 estimate
  WP_DEBUG            (default 0)     verbose output
  WP_ALLOW_FAKE_RISK  (default "low,medium")

Fees:
  CATWIKI_COMMISSION_RATE  (default 0.125)
  CATWIKI_COMMISSION_VAT   (default 0.21)
  CHRONO24_FEE_RATE        (default 0.065)
  CHRONO24_PRICE_MULT      (default 1.20)
  CHRONO24_MIN_BUY_EUR     (default 200.0)
"""

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

from timelab_core.matching import detect_brand_ambiguity, derive_match_confidence
from timelab_core.scoring import (
    bucket_from_score,
    brand_score,
    compute_confidence,
    compute_opportunity_score,
    derive_close_estimate_confidence,
    estimate_close_price,
    explain_bucket,
    liquidity_score,
)
from timelab_core.model_engine import (
    gate_decision,
    load_model_master,
    load_target_stats,
    resolve_listing_identity,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

def env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except Exception:
        return default

def env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, str(default))).strip().replace(",", "."))
    except Exception:
        return default

def env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None else str(v).strip()

TELEGRAM_BOT_TOKEN = env_str("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = env_str("TELEGRAM_CHAT_ID",   "")

WP_MAX_ITEMS     = env_int("WP_MAX_ITEMS",  200)
WP_THROTTLE_S    = env_float("WP_THROTTLE_S", 0.5)
WP_TIMEOUT       = env_int("WP_TIMEOUT",    20)
WP_MIN_MATCH     = env_int("WP_MIN_MATCH",  60)
WP_MIN_NET_EUR   = env_float("WP_MIN_NET_EUR",  20.0)
WP_MIN_NET_ROI   = env_float("WP_MIN_NET_ROI",  0.10)
WP_CLOSE_HAIRCUT = env_float("WP_CLOSE_HAIRCUT", 0.88)
WP_DEBUG         = env_int("WP_DEBUG", 0) == 1
ALLOW_FAKE_RISK  = {s.strip().lower() for s in env_str("WP_ALLOW_FAKE_RISK", "low,medium").split(",") if s.strip()}

CATWIKI_COMMISSION_RATE = env_float("CATWIKI_COMMISSION_RATE", 0.125)
CATWIKI_COMMISSION_VAT  = env_float("CATWIKI_COMMISSION_VAT",  0.21)

# BUG-3 FIX: Chrono24 constants
CHRONO24_FEE_RATE    = float(os.getenv("CHRONO24_FEE_RATE",    "0.065"))
CHRONO24_PRICE_MULT  = float(os.getenv("CHRONO24_PRICE_MULT",  "1.20"))
CHRONO24_MIN_BUY_EUR = float(os.getenv("CHRONO24_MIN_BUY_EUR", "200.0"))

# ─────────────────────────────────────────────
# WALLAPOP API
# ─────────────────────────────────────────────

WALLAPOP_SEARCH_URL    = "https://api.wallapop.com/api/v3/general/search"
WALLAPOP_WATCH_CATEGORY = "14000"

WALLAPOP_QUERIES = [
    "reloj omega",
    "reloj longines",
    "reloj tissot",
    "reloj seiko vintage",
    "reloj hamilton",
    "reloj zenith",
    "reloj oris",
    "reloj junghans",
    "reloj certina",
    "reloj baume mercier",
    "reloj tag heuer",
    "reloj breitling vintage",
    "reloj rado",
    "reloj fortis",
    "reloj cyma vintage",
]

WALLAPOP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "es-ES,es;q=0.9",
    "Referer": "https://es.wallapop.com/",
    "Origin": "https://es.wallapop.com",
}

# ─────────────────────────────────────────────
# FILTER TERMS
# ─────────────────────────────────────────────

BANNED_BRANDS = {
    "lotus", "festina", "calvin klein", "diesel", "armani", "emporio armani",
    "michael kors", "guess", "tommy hilfiger", "fossil", "dkny", "police",
    "samsung", "huawei", "xiaomi", "garmin", "fitbit", "amazfit",
    "skagen", "swatch", "ice watch", "sector", "viceroy", "casio",
}

REPUTABLE_BRANDS = {
    "omega", "longines", "tag heuer", "tissot", "seiko", "hamilton",
    "oris", "zenith", "baume", "baume & mercier", "frederique constant",
    "raymond weil", "sinn", "junghans", "certina", "tudor", "breitling",
    "rolex", "jaeger", "jaeger lecoultre", "iwc", "cartier", "panerai",
    "rado", "fortis", "cyma", "mido", "eterna", "bulova", "citizen",
    "alpina", "glycine", "doxa", "heuer",
}

BAD_COND_TERMS = [
    "para piezas", "por piezas", "sin funcionar", "no funciona", "averiado",
    "defectuoso", "incompleto", "reparar", "para reparar", "for parts",
    "spares", "repair", "broken", "no arranca", "parado",
]

LADIES_TERMS = [
    "senora", "mujer", "lady", "donna", "femme", "damenuhr", "chica",
]

SMARTWATCH_TERMS = [
    "smartwatch", "connected", "apple watch", "watch series",
    "galaxy watch", "wear os",
]

# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class WallapopListing:
    item_id:     str
    title:       str
    description: str
    price_eur:   float
    url:         str
    location:    str
    seller:      str
    images:      int  = 0
    reserved:    bool = False
    cond:        str  = ""

# ─────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────

def chunk_text(text: str, max_len: int = 3800) -> List[str]:
    if len(text) <= max_len:
        return [text]
    out, cur, cur_len = [], [], 0
    for line in text.splitlines(True):
        if cur_len + len(line) > max_len and cur:
            out.append("".join(cur))
            cur, cur_len = [], 0
        cur.append(line)
        cur_len += len(line)
    if cur:
        out.append("".join(cur))
    return out

def telegram_send(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for chunk in chunk_text(text):
        try:
            requests.post(
                url,
                json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "disable_web_page_preview": True},
                timeout=20,
            )
        except Exception:
            pass

# ─────────────────────────────────────────────
# WALLAPOP API CALLS
# ─────────────────────────────────────────────

def search_wallapop(
    query: str,
    max_items: int = 50,
    session: Optional[requests.Session] = None,
) -> List[WallapopListing]:
    """Search Wallapop's JSON API for a given query."""
    sess = session or requests.Session()
    results: List[WallapopListing] = []
    start = 0
    page_size = 40

    while len(results) < max_items:
        params = {
            "keywords":    query,
            "category_ids": WALLAPOP_WATCH_CATEGORY,
            "order_by":    "newest",
            "start":       str(start),
            "items_count": str(min(page_size, max_items - len(results))),
            "country_code": "ES",
            "language":    "es_ES",
        }
        try:
            r = sess.get(
                WALLAPOP_SEARCH_URL,
                headers=WALLAPOP_HEADERS,
                params=params,
                timeout=WP_TIMEOUT,
            )
            if r.status_code != 200:
                break
            data = r.json()
        except Exception:
            break

        search_objects = (
            data.get("data", {})
                .get("section", {})
                .get("payload", {})
                .get("items", [])
        )
        if not search_objects:
            search_objects = data.get("items", []) or []
        if not search_objects:
            break

        for item in search_objects:
            listing = _parse_wallapop_item(item)
            if listing:
                results.append(listing)

        if len(search_objects) < page_size:
            break

        start += page_size
        time.sleep(WP_THROTTLE_S)

    return results[:max_items]


def _parse_wallapop_item(item: Dict[str, Any]) -> Optional[WallapopListing]:
    """Parse a single Wallapop API item into a WallapopListing."""
    try:
        item_id = str(item.get("id") or item.get("item_id") or "")
        if not item_id:
            return None

        title = str(item.get("title") or "").strip()
        if not title:
            return None

        description = str(item.get("description") or "").strip()

        price_block = item.get("price") or item.get("salePrice") or {}
        if isinstance(price_block, dict):
            price_eur = float(price_block.get("amount") or price_block.get("value") or 0)
            currency  = str(price_block.get("currency") or "EUR").upper()
            if currency != "EUR":
                return None
        elif isinstance(price_block, (int, float)):
            price_eur = float(price_block)
        else:
            return None

        if price_eur <= 0 or price_eur > 25000:
            return None

        web_slug = item.get("web_slug") or item.get("slug") or item_id
        url      = f"https://es.wallapop.com/item/{web_slug}"

        location_data = item.get("location") or {}
        if isinstance(location_data, dict):
            city = str(location_data.get("city") or location_data.get("postal_code") or "")
        else:
            city = str(location_data or "")

        seller_data = item.get("user") or item.get("seller") or {}
        if isinstance(seller_data, dict):
            seller = str(seller_data.get("micro_name") or seller_data.get("id") or "")
        else:
            seller = ""

        images    = item.get("images") or item.get("image") or []
        img_count = len(images) if isinstance(images, list) else (1 if images else 0)

        reserved = bool(item.get("reserved") or item.get("flags", {}).get("reserved"))
        sold     = bool(item.get("sold")     or item.get("flags", {}).get("sold"))
        if sold:
            return None

        cond_raw = str(item.get("condition") or "").lower()
        cond_map = {
            "new": "nuevo", "as_good_as_new": "como nuevo",
            "good": "bueno", "fair": "aceptable", "has_given_it_all": "muy usado",
        }
        cond = cond_map.get(cond_raw, cond_raw or "desconocido")

        return WallapopListing(
            item_id=item_id,
            title=title,
            description=description,
            price_eur=price_eur,
            url=url,
            location=city,
            seller=seller,
            images=img_count,
            reserved=reserved,
            cond=cond,
        )
    except Exception:
        return None

# ─────────────────────────────────────────────
# TEXT HELPERS
# ─────────────────────────────────────────────

def canon(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().strip())

def has_any(text: str, terms: List[str]) -> bool:
    t = canon(text)
    return any(canon(x) in t for x in terms)

def extract_brand(title: str) -> Optional[str]:
    t = canon(title)
    for b in sorted(REPUTABLE_BRANDS, key=lambda x: -len(x)):
        if canon(b) in t:
            if b in ("tag", "heuer"):
                return "tag heuer"
            if "baume" in b:
                return "baume & mercier"
            return canon(b)
    for b in sorted(BANNED_BRANDS, key=lambda x: -len(x)):
        if canon(b) in t:
            return canon(b)
    return None

def is_banned(brand: Optional[str]) -> bool:
    return bool(brand and canon(brand) in {canon(x) for x in BANNED_BRANDS})

def is_reputable(brand: Optional[str]) -> bool:
    return bool(brand and canon(brand) in {canon(x) for x in REPUTABLE_BRANDS})

# ─────────────────────────────────────────────
# TARGET LOADING
# ─────────────────────────────────────────────

def load_targets(path: str = "target_list.json") -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "targets" in raw:
        targets = raw["targets"]
    elif isinstance(raw, list):
        targets = raw
    else:
        raise ValueError("target_list.json invalido")
    valid = [
        t for t in targets
        if isinstance(t, dict)
        and t.get("id") and t.get("brand")
        and isinstance(t.get("catawiki_estimate"), dict)
        and "p50" in t["catawiki_estimate"]
    ]
    if not valid:
        raise ValueError("No valid targets found")
    return valid

# ─────────────────────────────────────────────
# MATCHING / SCORING
# ─────────────────────────────────────────────

def model_keyword_hits(title: str, target: Dict[str, Any]) -> int:
    from timelab_core.matching import keyword_in_title
    t   = canon(title)
    kws = [canon(x) for x in (target.get("model_keywords") or []) if isinstance(x, str)]
    return sum(1 for kw in kws if kw and len(kw) > 2 and keyword_in_title(kw, t))

def violates_must_exclude(title: str, target: Dict[str, Any]) -> bool:
    t = canon(title)
    return any(
        x and canon(x) in t
        for x in (target.get("must_exclude") or [])
        if isinstance(x, str)
    )

def compute_match_score(title: str, target: Dict[str, Any]) -> int:
    t     = canon(title)
    score = 0
    brand = canon(target.get("brand", ""))
    if brand and brand in t:
        score += 35
    must_in = [canon(x) for x in (target.get("must_include") or []) if isinstance(x, str)]
    if must_in:
        present = sum(1 for x in must_in if x and x in t)
        score  += int(35 * present / max(1, len(must_in)))
    else:
        score += 10
    hits = model_keyword_hits(title, target)
    if hits > 0:
        score += min(25, hits * 8)
    if has_any(title, BAD_COND_TERMS):
        score -= 25
    return max(0, min(100, score))

def best_target(
    title: str,
    targets: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], int, int]:
    best_t, best_s, best_hits = None, -1, 0
    for trg in targets:
        if violates_must_exclude(title, trg):
            continue
        hits      = model_keyword_hits(title, trg)
        target_id = str(trg.get("id", "")).upper()
        is_generic = target_id.endswith("_GENERIC")
        kws       = trg.get("model_keywords") or []
        if not is_generic and isinstance(kws, list) and len(kws) > 0 and hits < 1:
            continue
        s = compute_match_score(title, trg)
        if s > best_s:
            best_s, best_t, best_hits = s, trg, hits
    return best_t, best_s, best_hits

def condition_score(cond: str, title: str) -> int:
    c = canon(cond) + " " + canon(title)
    if "perfecto" in c or "como nuevo" in c or "nuevo" in c:
        return 20
    if "excelente" in c or "muy bueno" in c:
        return 12
    if "bueno" in c:
        return 5
    if has_any(c, BAD_COND_TERMS):
        return -35
    if "aceptable" in c or "usado" in c:
        return -10
    return 0

# ─────────────────────────────────────────────
# ECONOMICS
# ─────────────────────────────────────────────

def estimate_net(buy: float, close: float) -> Tuple[float, float]:
    """Net profit after Catawiki commission + VAT. Shipping arbitrage +35 euros."""
    commission     = close * CATWIKI_COMMISSION_RATE
    commission_vat = commission * CATWIKI_COMMISSION_VAT
    SHIP_ARB       = 35.0
    net = close + SHIP_ARB - commission - commission_vat - buy
    roi = net / max(1e-9, buy)
    return round(net, 2), round(roi, 4)

def estimate_chrono24(buy: float, catawiki_p50: float) -> Tuple[float, float, float]:
    """BUG-3 FIX: Alternative estimate if sold on Chrono24 (6.5% fee, BIN ~20% above Catawiki p50)."""
    c24_close = round(catawiki_p50 * CHRONO24_PRICE_MULT, 2)
    c24_net   = round(c24_close * (1 - CHRONO24_FEE_RATE) - buy, 2)
    c24_roi   = round(c24_net / max(1e-9, buy), 4)
    return c24_close, c24_net, c24_roi

# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────

def run() -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    try:
        targets = load_targets("target_list.json")
    except Exception as e:
        telegram_send(f"TIMELAB Wallapop scanner: error cargando targets\n{type(e).__name__}: {e}")
        return

    model_master = load_model_master()
    target_stats = load_target_stats()

    session = requests.Session()

    diag: Dict[str, Any] = {
        "queries":      len(WALLAPOP_QUERIES),
        "collected":    0,
        "scanned":      0,
        "passed_match": 0,
        "passed_net":   0,
        "dropped": {"ladies": 0, "smartwatch": 0, "banned": 0, "no_brand": 0, "bad_cond": 0, "reserved": 0},
    }

    seen_ids: set = set()
    all_listings: List[WallapopListing] = []

    for query in WALLAPOP_QUERIES:
        if len(all_listings) >= WP_MAX_ITEMS:
            break
        try:
            batch = search_wallapop(query, max_items=40, session=session)
        except Exception as e:
            if WP_DEBUG:
                print(f"ERROR query '{query}': {e}")
            continue

        for listing in batch:
            if listing.item_id in seen_ids:
                continue
            seen_ids.add(listing.item_id)
            all_listings.append(listing)

        time.sleep(WP_THROTTLE_S)

    diag["collected"] = len(all_listings)

    candidates: List[Dict[str, Any]] = []

    for listing in all_listings:
        diag["scanned"] += 1
        full_text = f"{listing.title} {listing.description}"

        if listing.reserved:
            diag["dropped"]["reserved"] += 1
            continue
        if has_any(full_text, LADIES_TERMS):
            diag["dropped"]["ladies"] += 1
            continue
        if has_any(full_text, SMARTWATCH_TERMS):
            diag["dropped"]["smartwatch"] += 1
            continue
        if has_any(full_text, BAD_COND_TERMS):
            diag["dropped"]["bad_cond"] += 1
            continue

        brand = extract_brand(full_text)
        if brand is None:
            diag["dropped"]["no_brand"] += 1
            continue
        if is_banned(brand):
            diag["dropped"]["banned"] += 1
            continue
        if not is_reputable(brand):
            diag["dropped"]["no_brand"] += 1
            continue

        target, match, hits = best_target(full_text, targets)
        identity = resolve_listing_identity(full_text, None, model_master)

        if identity.get("target_id"):
            forced = next(
                (t for t in targets if str(t.get("id", "")).upper() == str(identity.get("target_id", "")).upper()),
                None,
            )
            if forced:
                target = forced
                match  = max(match, int(identity.get("final_match_score", 0)))
                hits   = max(hits, int(identity.get("keyword_score", 0) // 6))

        if target is None or match < WP_MIN_MATCH:
            continue

        brand_ctx = detect_brand_ambiguity(full_text, str(target.get("brand", "")))
        if brand_ctx.get("movement_brand_contamination"):
            continue

        risk = str(target.get("risk", "medium")).lower()
        if risk not in ALLOW_FAKE_RISK:
            continue

        diag["passed_match"] += 1

        est      = target.get("catawiki_estimate", {})
        p50      = float(est.get("p50", 0))
        p75      = float(est.get("p75", p50))
        triggers = target.get("p75_triggers_any") or []
        cscore   = condition_score(listing.cond, full_text)
        close_est, listing_flags = estimate_close_price(
            p50=p50, p75=p75,
            detail_text=full_text,
            condition_score=cscore,
            haircut=WP_CLOSE_HAIRCUT,
            triggers=triggers,
        )

        max_buy = target.get("max_buy_eur")
        if isinstance(max_buy, (int, float)) and listing.price_eur > float(max_buy):
            continue

        net, roi = estimate_net(listing.price_eur, close_est)

        if not (net >= WP_MIN_NET_EUR and roi >= WP_MIN_NET_ROI):
            continue

        diag["passed_net"] += 1

        # BUG-3 FIX: Chrono24 alternative estimate
        c24_close, c24_net, c24_roi = 0.0, 0.0, 0.0
        suggest_c24 = False
        if listing.price_eur >= CHRONO24_MIN_BUY_EUR and p50 > 0:
            c24_close, c24_net, c24_roi = estimate_chrono24(listing.price_eur, p50)
            suggest_c24 = c24_net > net

        target_id   = str(target.get("id", "")).upper()
        is_generic  = target_id.endswith("_GENERIC")
        stats       = target_stats.get(target_id, {})
        sample_size = int(stats.get("sample_size", 0))
        valuation_conf = min(95, 40 + sample_size * 8)

        opp_score = compute_opportunity_score(
            net=net, roi=roi, match_score=match,
            condition_score=cscore,
            brand_points=brand_score(risk),
            liquidity_points=liquidity_score(str(target.get("liquidity", "medium"))),
            ambiguity_penalty=int(brand_ctx.get("penalty", 0)),
        )
        bucket = bucket_from_score(opp_score, is_generic=is_generic, discovery=False)

        match_band = identity.get("match_confidence_band", "low")
        gate = gate_decision(
            match_band, sample_size, valuation_conf,
            net, roi, risk in ALLOW_FAKE_RISK,
            bool(identity.get("model_ambiguity", True)),
        )
        if gate == "SKIP":
            continue
        if gate == "REVIEW":
            bucket = "REVIEW"
        if gate == "BUY" and bucket not in {"BUY", "SUPER_BUY"}:
            bucket = "BUY"

        close_conf  = derive_close_estimate_confidence(
            listing_flags,
            used_p75=(listing_flags.get("is_full_set") or listing_flags.get("is_nos")),
            condition_score=cscore,
        )
        match_conf  = derive_match_confidence(match, hits, int(brand_ctx.get("penalty", 0)))
        conf        = compute_confidence(75, match_conf, close_conf)
        reason_text = explain_bucket(opp_score, bucket, net, roi, match, int(brand_ctx.get("penalty", 0)))

        max_buy_suggested = round(p50 * WP_CLOSE_HAIRCUT * 0.55, 0)

        candidates.append({
            "listing":    listing,
            "target":     target,
            "match":      match,
            "kw_hits":    hits,
            "close_est":  close_est,
            "net":        net,
            "roi":        roi,
            "bucket":     bucket,
            "score":      opp_score,
            "flags":      listing_flags,
            "reason_flags": brand_ctx.get("reason_flags", []),
            "confidence": conf,
            "max_buy_suggested": max_buy_suggested,
            "explain": {
                "target":                target.get("id"),
                "keyword_hits":          hits,
                "penalties":             int(brand_ctx.get("penalty", 0)),
                "comparables":           {"p50": p50, "p75": p75},
                "bucket_reason":         reason_text,
                "match_confidence_band": match_band,
                "sample_size":           sample_size,
                "valuation_confidence":  valuation_conf,
            },
            "chrono24": {
                "suggested": suggest_c24,
                "close_est": c24_close,
                "net":       c24_net,
                "roi":       c24_roi,
            },
        })

    candidates.sort(key=lambda x: (x["score"], x["net"], x["match"]), reverse=True)
    top = candidates[:10]

    header = f"TIMELAB Wallapop Scan - TOP {len(top)}\n\n"

    if not top:
        telegram_send(
            header
            + "Sin oportunidades que cumplan filtros.\n"
            + f"Recolectados: {diag['collected']} | Escaneados: {diag['scanned']}"
        )
    else:
        lines = [header]
        for i, it in enumerate(top, 1):
            li     = it["listing"]
            bucket = it.get("bucket", "BUY")
            emoji  = "verde" if "BUY" in bucket else "amarillo"
            flags  = it.get("flags", {})
            c24    = it.get("chrono24", {})
            expl   = it.get("explain", {})

            flag_parts = []
            if flags.get("has_box"):     flag_parts.append("caja")
            if flags.get("has_papers"):  flag_parts.append("papeles")
            if flags.get("has_service"): flag_parts.append("revisado")
            if flags.get("is_nos"):      flag_parts.append("NOS")
            if flags.get("is_full_set"): flag_parts.append("full set")
            flags_str = " ".join(flag_parts) or "sin extras"

            lines.append(f"{i}) [{bucket}] [Wallapop] {li.title}")
            lines.append(f"   Compra: {li.price_eur:.0f}EUR  |  Cierre Catawiki est.: {it['close_est']:.0f}EUR")
            lines.append(f"   Neto: {it['net']:.0f}EUR  |  ROI: {it['roi']*100:.1f}%  |  Score: {it['score']}")
            if c24.get("suggested") and c24.get("net", 0) > 0:
                lines.append(f"   Chrono24 MEJOR -> cierre {c24['close_est']:.0f}EUR | neto {c24['net']:.0f}EUR | ROI {c24['roi']*100:.1f}%")
            elif c24.get("close_est", 0) > 0:
                lines.append(f"   Chrono24 alt. -> cierre {c24['close_est']:.0f}EUR | neto {c24['net']:.0f}EUR | ROI {c24['roi']*100:.1f}%")
            lines.append(f"   Max compra sugerido: {it['max_buy_suggested']:.0f}EUR  |  Match: {it['match']}  |  KW: {it['kw_hits']}")
            lines.append(f"   Estado: {li.cond}  |  {flags_str}  |  Fotos: {li.images}  |  {li.location}")
            lines.append(f"   Target: {it['target'].get('id', 'N/A')}  |  Risk: {','.join(it['reason_flags']) or 'ok'}")
            lines.append(f"   {expl.get('bucket_reason', '')[:120]}")
            lines.append(f"   {li.url}\n")

        telegram_send("\n".join(lines).strip())

    if WP_DEBUG:
        dbg = [
            "TIMELAB Wallapop Debug",
            f"Queries: {diag['queries']} | Collected: {diag['collected']} | Scanned: {diag['scanned']}",
            f"Passed: match={diag['passed_match']} | net={diag['passed_net']}",
            f"Dropped: ladies={diag['dropped']['ladies']} | smart={diag['dropped']['smartwatch']} | "
            f"banned={diag['dropped']['banned']} | no_brand={diag['dropped']['no_brand']} | "
            f"bad_cond={diag['dropped']['bad_cond']} | reserved={diag['dropped']['reserved']}",
            f"Thresholds: match>={WP_MIN_MATCH} | net>={WP_MIN_NET_EUR}EUR | roi>={WP_MIN_NET_ROI} | haircut={WP_CLOSE_HAIRCUT}",
        ]
        telegram_send("\n".join(dbg))


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        telegram_send(f"TIMELAB Wallapop scanner crashed\n{type(e).__name__}: {e}")
        raise
