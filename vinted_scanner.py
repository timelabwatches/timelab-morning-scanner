#!/usr/bin/env python3
"""
TIMELAB — Vinted scanner v1.

Modeled on cashconverters_scanner.py architecture but adapted for Vinted's
JSON API at /api/v2/catalog/items.

Key design decisions:
  - Session-level homepage warmup (Vinted requires Cloudflare cookies before
    the API will return JSON). See Wallapop probe lessons learned: this works
    on Vinted but not on Wallapop because Wallapop additionally uses DataDome.
  - 15 brand queries covering 95% of historical close volume (Q1+Q2+Catawiki).
  - Vinted-specific noise filters that CC doesn't need:
      * brand_title == "Self" / "Other" / "Otro" / empty → reject
      * title contains supplement/toy/plastic-toy noise → reject
      * path field must include "relojes" → reject
  - Pricing uses target_list.json's catawiki_estimate (same as CC's legacy).
    Phase 2 (apply_comparables_engine_*) is NOT integrated in v1 by design —
    we want to see what raw scanning produces before adding the override
    layer. Adding Phase 2 later is the same one-line edit done in CC.
  - Separate Telegram channel via TELEGRAM_CHAT_ID_VINTED.
  - Per-shop cooldown state at state/state_vinted.json.

Run:  python vinted_scanner.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

from comparables.shadow import apply_comparables_engine_vinted


# ─────────────────────────────────────────────
# CONFIG / ENV HELPERS
# ─────────────────────────────────────────────

def env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except Exception:
        return default

def env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, str(default))).strip())
    except Exception:
        return default

def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default) or default


TELEGRAM_BOT_TOKEN  = env_str("TELEGRAM_BOT_TOKEN_VINTED", "") or env_str("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID    = env_str("TELEGRAM_CHAT_ID_VINTED", "") or env_str("TELEGRAM_CHAT_ID", "")

VT_MAX_ITEMS_PER_QUERY = env_int("VT_MAX_ITEMS_PER_QUERY", 60)
VT_PAGES_PER_QUERY     = env_int("VT_PAGES_PER_QUERY",     3)
VT_THROTTLE_S          = env_float("VT_THROTTLE_S",        1.2)
VT_TIMEOUT             = env_int("VT_TIMEOUT",             20)
VT_MIN_MATCH_SCORE     = env_int("VT_MIN_MATCH_SCORE",     55)
VT_MIN_NET_EUR         = env_float("VT_MIN_NET_EUR",       30.0)
VT_MIN_NET_ROI         = env_float("VT_MIN_NET_ROI",       0.15)
# Sanity cap: legitimate Vinted arbitrage rarely exceeds ~150-200% ROI.
# ROI above this is almost always a false positive: replica, stolen item,
# title misleading (parts-only sold), or wrong target match. Reject these.
VT_ROI_SANITY_MAX      = env_float("VT_ROI_SANITY_MAX",    3.0)
VT_CLOSE_HAIRCUT       = env_float("VT_CLOSE_HAIRCUT",     0.85)
VT_COOLDOWN_HOURS      = env_int("VT_COOLDOWN_HOURS",      48)
VT_DEBUG               = env_int("VT_DEBUG",               1)

# Catawiki economics
CATWIKI_COMMISSION_RATE = env_float("CATWIKI_COMMISSION_RATE", 0.125)
CATWIKI_COMMISSION_VAT  = env_float("CATWIKI_COMMISSION_VAT",  0.21)
SHIP_ARB_EUR            = env_float("SHIP_ARB_EUR",            35.0)


# ─────────────────────────────────────────────
# VINTED API
# ─────────────────────────────────────────────

VINTED_HOME    = "https://www.vinted.es/"
VINTED_API     = "https://www.vinted.es/api/v2/catalog/items"
WATCHES_CATEGORY = "304"  # Vinted ES "Relojes" category id

# Top 15 brands by historical close volume (Q1+Q2 contabilidad + Catawiki)
VINTED_QUERIES = [
    "tissot", "seiko", "longines", "hamilton", "zenith",
    "omega", "citizen", "baume mercier", "cauny", "certina",
    "junghans", "maurice lacroix", "oris", "cyma", "tag heuer",
]

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/120.0.0.0 Safari/537.36")

HOME_HEADERS = {
    "User-Agent":      UA,
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "DNT":             "1",
    "Connection":      "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Ch-Ua":          '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    "Sec-Ch-Ua-Mobile":   "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest":     "document",
    "Sec-Fetch-Mode":     "navigate",
    "Sec-Fetch-Site":     "none",
    "Sec-Fetch-User":     "?1",
}

API_HEADERS = {
    "User-Agent":      UA,
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate",  # NOT br — requests doesn't decompress brotli
    "Referer":         "https://www.vinted.es/catalog?catalog[]=" + WATCHES_CATEGORY,
    "Origin":          "https://www.vinted.es",
    "DNT":             "1",
    "Connection":      "keep-alive",
    "X-Requested-With": "XMLHttpRequest",
    "Sec-Ch-Ua":          '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    "Sec-Ch-Ua-Mobile":   "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest":     "empty",
    "Sec-Fetch-Mode":     "cors",
    "Sec-Fetch-Site":     "same-origin",
}


# ─────────────────────────────────────────────
# FILTER TERMS (Vinted-specific noise)
# ─────────────────────────────────────────────

# brand_title values that are user-entered junk (not real brands)
JUNK_BRAND_TITLES = {
    "self", "other", "otro", "otra", "otros", "varios",
    "no marca", "sin marca", "marca generica", "marca genérica",
    "", "n/a", "none", "no aplicable",
}

# Reject any listing whose title contains these — almost always not a watch
NOISE_TITLE_TOKENS = [
    "omega 3", "omega-3", "omega3",       # vitaminas/aceite
    "vitamina", "vitamin",
    "perfume", "colonia", "fragancia",
    "infantil", "niño", "niña", "niños",  # juguete tipico
    "muñeca", "muñeco", "barbie",
    "casio digital g-shock",              # pasa pero no nuestro target
    "smartwatch", "smart watch", "pulsera inteligente",
    "fitness", "actividad",
    "pulsera de hilo", "pulsera trenzada", # accesorios no relojes
    "correa solo", "solo correa",          # vendiendo solo la correa
    "para reloj", "de reloj",              # accesorios para reloj
    "pin de", "pin reloj",                 # pins
    "tarjeta postal", "postal antigua",    # postales con relojes
    "puzzle", "rompecabezas",
    "póster", "poster",
    "camiseta", "tshirt", "t-shirt",       # ropa
    "llavero",
]

# Reject brands that are user-entered but not our targets
BANNED_BRANDS = {
    "lotus", "festina", "casio", "calvin klein", "diesel", "armani",
    "emporio armani", "michael kors", "guess", "tommy hilfiger",
    "fossil", "dkny", "police", "skagen", "swatch", "ice watch",
    "sector", "viceroy", "samsung", "huawei", "xiaomi", "garmin",
    "fitbit", "amazfit", "marea", "munich", "morellato",
    "daniel wellington", "cluse", "bulbul", "casual",
    # Lo que aparezca como "Self"/"Otro" lo cazamos por JUNK_BRAND_TITLES
}

# Brands we actively want
REPUTABLE_BRANDS = {
    "tissot", "seiko", "longines", "hamilton", "zenith", "omega",
    "citizen", "baume", "baume & mercier", "cauny", "certina",
    "junghans", "maurice lacroix", "oris", "cyma", "tag heuer",
    "tag", "heuer", "tudor", "rolex", "breitling", "iwc",
    "frederique constant", "raymond weil", "sinn", "yema", "lip",
    "favre-leuba", "doxa", "eterna", "bulova", "movado",
    "universal geneve", "wittnauer", "vulcain", "alpina",
    "zodiac", "edox", "duward", "dogma", "fortis", "rado",
    "mido", "candino", "girard-perregaux", "jaeger lecoultre",
    "panerai", "cartier",
}

BAD_CONDITION_TERMS = [
    "no funciona", "para piezas", "para repuestos", "para restaurar",
    "estropeado", "roto", "rota", "averia", "avería",
    "se vende sin", "incompleto", "falta",
]

# Tokens that, if they are the FIRST word of the title, indicate the listing
# is selling watch PARTS, not the watch itself. Caught a "Caja reloj Tissot"
# false positive in the v1 run where the seller listed just the empty box.
# Vinted has many ES/IT/FR listings so we cover all three.
PARTS_ONLY_FIRST_TOKENS = {
    # Spanish
    "caja", "estuche", "correa", "cristal", "esfera", "máquina", "maquina",
    "movimiento", "manecillas", "agujas", "armis", "pulsera", "dial",
    "bisel", "corona",
    # Italian (cinturino, cassa, etc.)
    "cinturino", "cassa", "cassette", "vetro", "quadrante", "lancette",
    "movimento", "corona",
    # French
    "bracelet", "boîte", "boite", "boîtier", "boitier", "verre", "cadran",
    "aiguilles", "couronne",
}

# As above but checked anywhere in the FIRST 3 words of the title — covers
# patterns like "Oris cinturino in caucciù" where the brand name is first
# but the parts noun comes second. Same multilingual coverage.
PARTS_ONLY_EARLY_TOKENS = {
    # Spanish
    "correa", "esfera", "movimiento", "manecillas",
    # Italian
    "cinturino", "cassa", "vetro", "quadrante", "lancette", "movimento",
    # French
    "bracelet", "boîte", "boite", "boîtier", "boitier", "verre", "cadran",
    "aiguilles",
}

# Premium watch model tokens. When ANY of these appears in a listing title,
# the watch is identified as a high-tier model. We tighten the ROI sanity:
# real premium watches don't sell at 70%+ discount on Vinted — that's a fake.
PREMIUM_MODEL_TOKENS = {
    # Omega
    "seamaster 007", "speedmaster", "moonwatch", "speedy", "constellation",
    # Rolex
    "submariner", "daytona", "datejust", "gmt-master", "gmt master",
    "explorer", "yacht-master", "sea-dweller", "milgauss", "presidente",
    # Patek
    "nautilus", "aquanaut", "calatrava", "perpetual",
    # Audemars Piguet
    "royal oak",
    # TAG Heuer
    "carrera", "monaco", "aquaracer",
    # Tudor
    "black bay", "pelagos",
    # Breitling
    "navitimer", "chronomat", "superocean",
    # Other premium
    "el primero", "big bang", "luminor", "vintage rolex",
}
PREMIUM_ROI_MAX = 1.5  # 150% — real premium watches don't get 70% off


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class VintedListing:
    item_id:      str
    title:        str
    price_eur:    float
    currency:     str
    url:          str
    brand_title:  str
    size_title:   str
    status:       str           # "Nuevo con etiquetas", "Muy bueno", etc.
    user_id:      str
    photos_count: int
    favourite_count: int
    view_count:   int
    path:         str           # e.g. "Mujer / Accesorios / Relojes"
    raw_text:     str = ""      # for matching


# ─────────────────────────────────────────────
# UTIL
# ─────────────────────────────────────────────

def canon(s: str) -> str:
    if not s: return ""
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def has_any(text: str, terms: List[str]) -> bool:
    t = canon(text)
    return any(term in t for term in terms)


# ─────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────

def telegram_send(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TG] missing token/chat_id — skipping send", flush=True)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    # Telegram caps at 4096 chars; chunk if needed
    for chunk in (text[i:i+3800] for i in range(0, len(text), 3800)):
        try:
            requests.post(
                url,
                json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk,
                      "disable_web_page_preview": True},
                timeout=10,
            )
        except Exception as e:
            print(f"[TG] send failed: {e}", flush=True)


# ─────────────────────────────────────────────
# VINTED SESSION + SEARCH
# ─────────────────────────────────────────────

_VT_WARMUP_DONE = False

def vt_warmup(sess: requests.Session) -> bool:
    """Hit homepage once to acquire CloudFlare + session cookies."""
    try:
        r = sess.get(VINTED_HOME, headers=HOME_HEADERS, timeout=VT_TIMEOUT)
        cookie_names = sorted({c.name for c in sess.cookies})
        print(f"[VT] warmup status={r.status_code} cookies={len(sess.cookies)} "
              f"names={cookie_names[:8]}", flush=True)
        return r.status_code == 200
    except Exception as e:
        print(f"[VT] warmup failed: {type(e).__name__}: {e}", flush=True)
        return False


def search_vinted(
    query: str,
    sess: requests.Session,
    max_items: int = VT_MAX_ITEMS_PER_QUERY,
) -> List[VintedListing]:
    """Search Vinted catalog API for one query, paginating up to VT_PAGES_PER_QUERY."""
    global _VT_WARMUP_DONE
    if not _VT_WARMUP_DONE:
        _VT_WARMUP_DONE = True  # set even on failure to avoid retry storms
        vt_warmup(sess)
        time.sleep(1.5)

    results: List[VintedListing] = []
    for page in range(1, VT_PAGES_PER_QUERY + 1):
        if len(results) >= max_items:
            break
        params = {
            "search_text":  query,
            "catalog_ids":  WATCHES_CATEGORY,
            "order":        "newest_first",
            "page":         str(page),
            "per_page":     "20",
        }
        try:
            r = sess.get(VINTED_API, headers=API_HEADERS,
                         params=params, timeout=VT_TIMEOUT)
            if r.status_code != 200:
                print(f"[VT] q='{query}' page={page} status={r.status_code} BREAK", flush=True)
                break
            data = r.json()
        except Exception as e:
            print(f"[VT] q='{query}' page={page} EXCEPTION {type(e).__name__}: {e}", flush=True)
            break

        items = data.get("items") or []
        if not items:
            break

        for item in items:
            listing = _parse_vinted_item(item)
            if listing:
                results.append(listing)

        if len(items) < 20:
            break  # last page

        time.sleep(VT_THROTTLE_S)

    return results[:max_items]


def _parse_vinted_item(item: Dict[str, Any]) -> Optional[VintedListing]:
    """Convert API item dict → VintedListing. Returns None on parse failure."""
    try:
        item_id = str(item.get("id") or "")
        if not item_id:
            return None
        title = str(item.get("title") or "").strip()
        if not title:
            return None

        # Price comes as string sometimes, dict {amount, currency_code} other times
        price_field = item.get("price")
        if isinstance(price_field, dict):
            price_eur = float(price_field.get("amount") or 0)
            currency = str(price_field.get("currency_code") or "EUR")
        else:
            price_eur = float(price_field or 0)
            currency = "EUR"

        if price_eur <= 0:
            return None
        if currency.upper() != "EUR":
            return None  # only ES domain, defensive

        url = str(item.get("url") or "")
        brand_title = str(item.get("brand_title") or "").strip()
        size_title  = str(item.get("size_title")  or "").strip()
        status      = str(item.get("status")      or "").strip()
        user_id     = str((item.get("user") or {}).get("id") or "") if isinstance(item.get("user"), dict) else ""
        photos      = item.get("photos") or []
        photos_count = len(photos) if isinstance(photos, list) else 0
        fav_count   = int(item.get("favourite_count") or 0)
        view_count  = int(item.get("view_count") or 0)
        path        = str(item.get("path") or "")

        raw_text = f"{title} {brand_title} {size_title} {status}".strip()

        return VintedListing(
            item_id=item_id, title=title, price_eur=price_eur, currency=currency,
            url=url, brand_title=brand_title, size_title=size_title,
            status=status, user_id=user_id, photos_count=photos_count,
            favourite_count=fav_count, view_count=view_count,
            path=path, raw_text=raw_text,
        )
    except Exception as e:
        print(f"[VT] parse error: {e}", flush=True)
        return None


# ─────────────────────────────────────────────
# FILTER PIPELINE
# ─────────────────────────────────────────────

def reject_reason(li: VintedListing) -> Optional[str]:
    """
    Quick pre-filter before target matching. Returns reject reason or None.
    The order is intentional — cheap checks first.
    """
    # Junk brand_title (user-entered "Self"/"Otro"/etc)
    if canon(li.brand_title) in JUNK_BRAND_TITLES:
        return "junk_brand_title"

    # Banned commodity / fashion brands
    bt = canon(li.brand_title)
    if bt in BANNED_BRANDS:
        return f"banned_brand:{bt}"

    # NOTE: previously also filtered by `path`, but the Vinted API field shape
    # we assumed (string with "/") doesn't match what the catalog endpoint
    # returns, so 84% of valid candidates were being killed. Since we already
    # filter by `catalog_ids=304` in the search params, every returned item
    # is in the Watches category by definition. Path filter removed.

    # Parts-only listings: title's FIRST word indicates this is a strap, box,
    # crown, etc. — not the full watch. Title-anywhere check is too aggressive
    # ("Tissot Seastar con caja original" should pass). Only reject when the
    # parts noun OPENS the title.
    title_words = canon(li.title).split()
    first_token = title_words[0] if title_words else ""
    if first_token in PARTS_ONLY_FIRST_TOKENS:
        return f"parts_only:{first_token}"

    # Also catch "Brand cinturino" pattern (parts token in early positions).
    # Multilingual: cinturino (IT), bracelet (FR), correa (ES), etc. Common
    # when the seller front-loads the brand name to be searchable but is
    # actually selling just the strap/case/etc.
    for tok in title_words[1:3]:
        if tok in PARTS_ONLY_EARLY_TOKENS:
            return f"parts_only_early:{tok}"

    # Title noise (vitamins, perfumes, kid toys, etc)
    if has_any(li.title, NOISE_TITLE_TOKENS):
        return "noise_title"

    # Bad condition mentioned in title (broken, for parts, etc)
    if has_any(li.title, BAD_CONDITION_TERMS):
        return "bad_condition"

    # Sanity: very low price + popular brand often means "para piezas" or fake
    if li.price_eur < 15:
        return "price_too_low"

    # Sanity: very high price (we don't operate above 1500€)
    if li.price_eur > 1500:
        return "price_too_high"

    return None


def extract_brand_from_listing(li: VintedListing) -> Optional[str]:
    """Try to identify the actual brand from title + brand_title."""
    bt = canon(li.brand_title)

    # If brand_title is reputable, trust it
    if bt in REPUTABLE_BRANDS:
        return bt
    # Handle "tag heuer" partial match
    if "tag" in bt and "heuer" in bt:
        return "tag heuer"
    if "baume" in bt:
        return "baume & mercier"

    # Otherwise try title with word-boundary matching, longest first
    title_canon = canon(li.title)
    for b in sorted(REPUTABLE_BRANDS, key=lambda x: -len(x)):
        if re.search(r'(?<![a-z])' + re.escape(b) + r'(?![a-z])', title_canon):
            if b == "tag" or b == "heuer":
                return "tag heuer"
            if b == "baume":
                return "baume & mercier"
            return b
    return None


# ─────────────────────────────────────────────
# TARGET MATCHING
# ─────────────────────────────────────────────

def load_targets(path: str = "target_list.json") -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("targets") or data.get("items") or []
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"[VT] load_targets failed: {e}", flush=True)
        return []


def keyword_hits(text: str, keywords: List[str]) -> int:
    if not keywords: return 0
    t = canon(text)
    return sum(1 for kw in keywords if canon(kw) and canon(kw) in t)


def compute_match_score(li: VintedListing, brand: str, target: Dict[str, Any]) -> Tuple[int, int]:
    """
    Returns (score, real_hits).

    v3 — Stricter matching that requires at least ONE model-level keyword to
    hit, not just the brand. Previous logic gave 40 base for brand match plus
    15×hits, so a generic "Maurice Lacroix vintage" with 0 model knowledge
    scored 70 because the brand tokens themselves hit must_include.

    Strategy: separate must_include keywords into brand-redundant tokens (the
    target.brand split into words — e.g. {'maurice', 'lacroix'}) vs real model
    keywords (everything else — e.g. ['aikon']). Require ≥1 real model hit.
    """
    target_brand = canon(target.get("brand", ""))
    if target_brand != brand:
        return 0, 0

    must_include = target.get("must_include") or target.get("keywords_any") or []
    if isinstance(must_include, str):
        must_include = [must_include]
    must_exclude = target.get("must_exclude") or []
    if isinstance(must_exclude, str):
        must_exclude = [must_exclude]

    text = li.raw_text
    text_c = canon(text)  # canon'd once for the bonus checks below

    # Hard exclude
    if must_exclude and has_any(text, must_exclude):
        return 0, 0

    # Separate brand-redundant keywords (already implied by brand match)
    # from real model keywords (Aikon, Spirit Zulu, Seastar, etc.)
    brand_tokens = set(canon(target_brand).split())
    real_keywords = [kw for kw in must_include
                     if canon(kw) and canon(kw) not in brand_tokens]
    brand_keywords = [kw for kw in must_include
                      if canon(kw) and canon(kw) in brand_tokens]

    real_hits  = keyword_hits(text, real_keywords)
    brand_hits = keyword_hits(text, brand_keywords)

    # Two flavors of target:
    #   (a) SPECIFIC: has real model keywords (e.g. ["aikon"]). Require ≥1 hit
    #       to avoid mass-matching all M.Lacroix listings to the Aikon target.
    #   (b) GENERIC: must_include is ONLY the brand (or empty). Catches
    #       vintage/generic listings that don't have model-specific keywords.
    #       Without a higher base score, these never reach threshold 55.
    if real_keywords:
        # SPECIFIC target — require model-level hit
        if real_hits == 0:
            return 0, 0
        score = 30 + min(40, 20 * real_hits) + min(10, 5 * brand_hits)
    else:
        # GENERIC target — brand match alone is the signal
        score = 45 + min(15, 8 * brand_hits)

    # Bonus 1 — soft signal from model_keywords (not a hard gate, just helps
    # disambiguate between competing generics). E.g. "Tissot vintage 1965"
    # against TISSOT_CHRONOGRAPH_GENERIC (mk=["chrono"]) vs TISSOT_VTG_GENERIC
    # (mk=["vintage"]) — the second wins via this bonus.
    target_model_kws = target.get("model_keywords") or []
    if isinstance(target_model_kws, str):
        target_model_kws = [target_model_kws]
    mk_hits = sum(
        1 for kw in target_model_kws
        if canon(kw) and canon(kw) not in brand_tokens and canon(kw) in text_c
    )
    score += min(6, 2 * mk_hits)

    # Bonus 2 — family field tokens. "Vintage Generic" vs "Chronograph (generic)"
    # vs "Aikon" — the family tells us what the target is FOR. Token hits in the
    # listing text reinforce the match.
    family_str = canon(target.get("family", ""))
    family_tokens = [
        t for t in family_str.split()
        if t and len(t) > 2 and t not in brand_tokens and t != "generic"
    ]
    family_hits = sum(1 for t in family_tokens if t in text_c)
    score += min(4, 2 * family_hits)

    # Bonus 3 — tie-breaker by must_exclude length. Targets with more exclude
    # rules are more carefully constrained and should win ties over targets
    # that match almost anything from the brand. Cap so it never dominates.
    score += min(2, len(must_exclude) // 7)

    # Penalty — when target claims to be "chronograph" (in its id or family)
    # but the listing has no chrono signal, the target is a poor fit. This
    # reliably breaks ties between VTG_CHRONOGRAPH_GENERIC and VTG_GENERIC
    # for vintage non-chrono listings.
    target_id_c = canon(target.get("id", ""))
    target_family_c = canon(target.get("family", ""))
    target_claims_chrono = (
        "chrono" in target_id_c or "chrono" in target_family_c
    )
    listing_has_chrono = (
        "chrono" in text_c or "cronograph" in text_c
        or "cronograf" in text_c or "chronograf" in text_c
    )
    if target_claims_chrono and not listing_has_chrono:
        score -= 5

    cond = canon(li.status)
    if "nuevo" in cond:  score += 10
    elif "muy bueno" in cond: score += 6
    elif "bueno" in cond: score += 3

    return min(100, score), real_hits


def best_target(li: VintedListing, brand: str, targets: List[Dict[str, Any]]
               ) -> Tuple[Optional[Dict[str, Any]], int, int]:
    """Pick the highest-scoring target for this listing."""
    best: Optional[Dict[str, Any]] = None
    best_score = 0
    best_hits = 0
    for t in targets:
        score, hits = compute_match_score(li, brand, t)
        if score > best_score:
            best_score = score
            best_hits = hits
            best = t
    return best, best_score, best_hits


# ─────────────────────────────────────────────
# ECONOMICS
# ─────────────────────────────────────────────

def estimate_close(target: Dict[str, Any], cond: str, title: str) -> float:
    """Catawiki close estimate. Mirrors CC's logic without vision adjustments."""
    est = target.get("catawiki_estimate") or {}
    p50 = float(est.get("p50") or 0.0)
    if p50 <= 0:
        return 0.0
    # Condition adjustment
    c = canon(cond)
    t = canon(title)
    adj = 1.0
    if "nuevo" in c:           adj = 1.05
    elif "muy bueno" in c:     adj = 1.02
    elif "bueno" in c:         adj = 1.00
    elif "aceptable" in c:     adj = 0.92
    if has_any(t, BAD_CONDITION_TERMS): adj = min(adj, 0.80)
    return round(p50 * VT_CLOSE_HAIRCUT * adj, 2)


def estimate_net(buy_eur: float, close_eur: float) -> Tuple[float, float]:
    """Net profit: close - commission - VAT + ship_arb - buy. ROI = net/buy."""
    commission = close_eur * CATWIKI_COMMISSION_RATE
    commission_vat = commission * CATWIKI_COMMISSION_VAT
    net = close_eur + SHIP_ARB_EUR - commission - commission_vat - buy_eur
    roi = net / max(1e-9, buy_eur)
    return round(net, 2), round(roi, 4)


# ─────────────────────────────────────────────
# COOLDOWN STATE
# ─────────────────────────────────────────────

STATE_PATH = "state/state_vinted.json"

def load_state() -> Dict[str, float]:
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state: Dict[str, float]) -> None:
    try:
        os.makedirs(os.path.dirname(STATE_PATH) or ".", exist_ok=True)
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"[VT] save_state failed: {e}", flush=True)

def in_cooldown(item_id: str, state: Dict[str, float], hours: int) -> bool:
    last = state.get(item_id)
    if not last: return False
    age_h = (time.time() - last) / 3600
    return age_h < hours


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run() -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[VT] missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID(_VINTED) — abort")
        return

    targets = load_targets("target_list.json")
    if not targets:
        print("[VT] no targets loaded — abort")
        return
    print(f"[VT] loaded {len(targets)} targets")

    state = load_state()
    sess = requests.Session()

    diag = {
        "queries": 0, "collected": 0, "scanned": 0,
        "passed_match": 0, "passed_econ": 0,
        "rejected": {
            "junk_brand_title": 0, "parts_only": 0, "parts_only_early": 0,
            "noise_title": 0, "bad_condition": 0,
            "price_too_low": 0, "price_too_high": 0,
            "banned_brand": 0, "no_brand": 0, "no_target_match": 0,
            "below_match_threshold": 0, "below_econ_threshold": 0,
            "roi_too_good": 0, "premium_roi_implausible": 0, "cooldown": 0,
        },
    }
    candidates: List[Dict[str, Any]] = []

    for query in VINTED_QUERIES:
        diag["queries"] += 1
        print(f"\n[VT] query: '{query}'", flush=True)
        items = search_vinted(query, sess)
        diag["collected"] += len(items)
        print(f"[VT] q='{query}' collected={len(items)}", flush=True)

        for li in items:
            diag["scanned"] += 1

            # Pre-filter
            rej = reject_reason(li)
            if rej:
                key = rej.split(":")[0]
                if key in diag["rejected"]:
                    diag["rejected"][key] += 1
                else:
                    diag["rejected"][key] = diag["rejected"].get(key, 0) + 1
                continue

            # Brand extraction
            brand = extract_brand_from_listing(li)
            if not brand:
                diag["rejected"]["no_brand"] += 1
                continue

            # Target match
            target, score, hits = best_target(li, brand, targets)
            if not target:
                diag["rejected"]["no_target_match"] += 1
                continue

            if score < VT_MIN_MATCH_SCORE:
                diag["rejected"]["below_match_threshold"] += 1
                continue

            diag["passed_match"] += 1

            # Economics
            close_est = estimate_close(target, li.status, li.title)
            # PHASE 2 — Override/promote with comparables engine when conf=high
            # or legacy was None. Same logic as in CC and secondhand scanners.
            close_est = apply_comparables_engine_vinted(
                legacy_close=close_est,
                listing=li,
                target=target,
                brand_hint=brand,
            )
            if close_est is None or close_est <= 0:
                diag["rejected"]["below_econ_threshold"] += 1
                continue
            net, roi = estimate_net(li.price_eur, close_est)
            if net < VT_MIN_NET_EUR or roi < VT_MIN_NET_ROI:
                diag["rejected"]["below_econ_threshold"] += 1
                continue

            # Sanity guardrail: ROI implausibly high → almost always a false
            # positive (replica, stolen, parts-only listing the parts filter
            # missed, target mismatch). Real Vinted arbitrage rarely exceeds
            # 150-200% ROI. Reject anything above VT_ROI_SANITY_MAX (default 3.0).
            if roi > VT_ROI_SANITY_MAX:
                diag["rejected"]["roi_too_good"] += 1
                continue

            # Tighter sanity for PREMIUM models: real Submariner / Daytona /
            # Speedmaster / Royal Oak / etc. listings on Vinted at 70%+
            # discount are virtually always replicas. If the title contains
            # any premium token, ROI must stay under 150%.
            title_lower = canon(li.title)
            has_premium = any(tok in title_lower for tok in PREMIUM_MODEL_TOKENS)
            if has_premium and roi > PREMIUM_ROI_MAX:
                diag["rejected"]["premium_roi_implausible"] += 1
                continue

            # Max buy price safety
            max_buy = target.get("max_buy_eur")
            if isinstance(max_buy, (int, float)) and li.price_eur > float(max_buy):
                diag["rejected"]["below_econ_threshold"] += 1
                continue

            # Cooldown
            if in_cooldown(li.item_id, state, VT_COOLDOWN_HOURS):
                diag["rejected"]["cooldown"] += 1
                continue

            diag["passed_econ"] += 1
            candidates.append({
                "listing": li, "target": target, "score": score, "hits": hits,
                "brand": brand, "close_est": close_est, "net": net, "roi": roi,
            })

    # Sort by net profit, take top
    # Dedupe first: same listing can come back from multiple queries
    # (Vinted's relevance is fuzzy). Keep the version with highest net.
    by_id: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        iid = c["listing"].item_id
        if iid not in by_id or c["net"] > by_id[iid]["net"]:
            by_id[iid] = c
    candidates = list(by_id.values())

    candidates.sort(key=lambda c: c["net"], reverse=True)
    top = candidates[:5]

    # Build alert
    if top:
        hora = datetime.now().strftime("%H:%M")
        lines = [
            f"⌚ TIMELAB · Vinted · {hora}",
            "━" * 26,
            "",
        ]
        for i, c in enumerate(top, 1):
            li = c["listing"]; t = c["target"]
            lines.append(f"{i}) [{c['brand'].title()}] {li.title[:80]}")
            lines.append(f"   💶 Compra: {li.price_eur:.0f}€  →  Cierre est.: {c['close_est']:.0f}€")
            lines.append(f"   ✅ Neto: {c['net']:.0f}€  |  ROI: {c['roi']*100:.0f}%  |  Score: {c['score']}")
            lines.append(f"   📋 Estado: {li.status}  |  Fotos: {li.photos_count}  |  Fav: {li.favourite_count}")
            lines.append(f"   🎯 Target: {t.get('id', '?')}")
            lines.append(f"   🔗 {li.url}")
            lines.append("")
            # Mark in cooldown state
            state[li.item_id] = time.time()

        telegram_send("\n".join(lines).strip())
    else:
        # Daily silent run also gets a heartbeat
        hora = datetime.now().strftime("%H:%M")
        msg = (f"⌚ TIMELAB · Vinted · {hora}\n"
               + "━" * 26
               + f"\nSin oportunidades hoy.\n"
               + f"Recolectados: {diag['collected']} | Escaneados: {diag['scanned']}")
        telegram_send(msg)

    # Persist cooldown
    save_state(state)

    # Debug summary
    if VT_DEBUG:
        rej_lines = " | ".join(f"{k}={v}" for k, v in diag["rejected"].items() if v > 0)
        dbg = [
            "TIMELAB Vinted Debug",
            f"Queries: {diag['queries']} | Collected: {diag['collected']} | Scanned: {diag['scanned']}",
            f"Passed: match={diag['passed_match']} | econ={diag['passed_econ']}",
            f"Rejected: {rej_lines or '(none)'}",
            f"Thresholds: match≥{VT_MIN_MATCH_SCORE} | net≥{VT_MIN_NET_EUR}€ | roi≥{VT_MIN_NET_ROI} | haircut={VT_CLOSE_HAIRCUT}",
            f"Top {len(top)} sent to Telegram (channel: {'VINTED' if env_str('TELEGRAM_CHAT_ID_VINTED', '') else 'main'})",
        ]
        telegram_send("\n".join(dbg))
        print("\n".join(dbg), flush=True)


if __name__ == "__main__":
    run()
