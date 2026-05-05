#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIMELAB — CashConverters ES scanner (requests + BeautifulSoup)

Core:
- Scans wristwatches on cashconverters.es (latest items)
- Matches against target_list.json
- Estimates Catawiki close via target_list.p50 * CLOSE_HAIRCUT * condition_adj
- Filters: reputable brand + match threshold + net/ROI threshold + allowed fake risk
- Sends a SEPARATE Telegram message with:
  "🕗 TIMELAB Morning Scan — TOP X (CashConverters ES)"

Key hardening:
- Robust price extraction (JSON-LD -> meta -> DOM -> € regex w/ context scoring)
- Prevent over-broad matching:
  If a target has model_keywords, REQUIRE at least 1 keyword hit for eligibility,
  except for *_GENERIC targets.
- Global exclusions: ladies / smartwatch-ish terms (to avoid Connected, etc.)

Env vars:
  TELEGRAM_BOT_TOKEN (required)
  TELEGRAM_CHAT_ID (required)

  CC_MAX_ITEMS (default 100)
  CC_PAGE_SIZE (default 60)
  CC_MAX_PAGES (default 50)
  CC_TIMEOUT (default 20)
  CC_THROTTLE_S (default 0.8)

  CC_DEBUG (default 0)
  CC_VERIFY_MODE (default 0)   # if 1: show near-misses in debug
  CC_STRICT_KEYWORDS (default 1)  # if 1: enforce model_keywords must hit at least 1 (except *_GENERIC)

  CC_MIN_MATCH_SCORE (default 65)
  CC_MIN_NET_EUR (default 20)
  CC_MIN_NET_ROI (default 0.08)
  CLOSE_HAIRCUT (default 0.90)

  CC_GOOD_BRANDS_TARGET (default 60)
  CC_ALLOW_FAKE_RISK (default "low,medium")

Fees:
  CATWIKI_COMMISSION_RATE (default 0.125)
  CATWIKI_COMMISSION_VAT (default 0.21)
"""

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from timelab_core.matching import detect_brand_ambiguity, detect_discovery_family, derive_match_confidence, keyword_in_title as core_keyword_in_title
from timelab_core.scoring import bucket_from_score, brand_score, compute_confidence, compute_opportunity_score, derive_close_estimate_confidence, estimate_close_price, explain_bucket, liquidity_score
from timelab_core.vision import load_vision_annotations, summarize_visual_hints
from timelab_core.model_engine import gate_decision, load_model_master, load_target_stats, resolve_listing_identity

from comparables.shadow import apply_comparables_engine_cc


# -----------------------------
# CONFIG / DEFAULTS
# -----------------------------
BASE_LISTING_URL = "https://www.cashconverters.es/es/es/comprar/relojes/?cgid=1471"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

BANNED_BRANDS = {
    "lotus", "festina", "calvin klein", "ck", "diesel", "armani", "emporio armani",
    "michael kors", "guess", "tommy", "tommy hilfiger", "fossil", "dkny", "police",
    "welder", "welder k", "samsung", "huawei", "xiaomi", "garmin", "fitbit", "amazfit",
    "skagen", "swatch", "ice watch", "ice", "sector", "viceroy", "casio",
}

REPUTABLE_BRANDS = {
    "omega", "longines", "tag heuer", "tag", "heuer", "tissot", "seiko", "hamilton",
    "oris", "zenith", "baume", "baume & mercier", "baume mercier", "frederique constant",
    "raymond weil", "sinn", "junghans", "certina", "tudor", "breitling", "rolex",
    "jaeger", "jaeger lecoultre", "iwc", "cartier", "panerai",
}

BAD_COND_TERMS = [
    "para piezas", "por piezas", "solo piezas", "sin funcionar", "no funciona", "averiado",
    "defectuoso", "incompleto", "reparar", "para reparar", "da riparare", "non funziona",
    "for parts", "spares", "parts", "repair", "broken",
]

GOOD_COND_TERMS = [
    "revisado", "revisada", "servicio", "serviced", "funciona", "working", "perfecto",
    "excelente", "muy buen estado", "buen estado", "como nuevo",
]

SMARTWATCH_TERMS = [
    "smartwatch", "connected", "apple watch", "watch series", "galaxy watch", "wear os",
]
LADIES_TERMS = [
    "señora", "senora", "mujer", "lady", "donna", "femme", "damenuhr", "girlfriend",
]

CATWIKI_COMMISSION_RATE = float(os.getenv("CATWIKI_COMMISSION_RATE", "0.125"))
CATWIKI_COMMISSION_VAT  = float(os.getenv("CATWIKI_COMMISSION_VAT",  "0.21"))
# CC ships for FREE to the seller. Catawiki charges ~50€ shipping to the buyer;
# real cost is ~10-15€ → ~35€ net arbitrage per operation (same as Wallapop/eBay).
SHIP_ARB_EUR = float(os.getenv("SHIP_ARB_EUR", "35.0"))
# Chrono24: 6.5% seller fee (no shipping arbitrage — handled separately per listing).
CHRONO24_FEE_RATE    = float(os.getenv("CHRONO24_FEE_RATE",    "0.065"))
CHRONO24_PRICE_MULT  = float(os.getenv("CHRONO24_PRICE_MULT",  "1.20"))   # C24 BIN prices ~20% above Catawiki p50
CHRONO24_MIN_BUY_EUR = float(os.getenv("CHRONO24_MIN_BUY_EUR", "200.0"))  # Only suggest C24 above this buy price


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
    return default if v is None else str(v)


TELEGRAM_BOT_TOKEN = env_str("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = env_str("TELEGRAM_CHAT_ID", "").strip()

CC_MAX_ITEMS = env_int("CC_MAX_ITEMS", 100)
CC_PAGE_SIZE = env_int("CC_PAGE_SIZE", 60)
CC_MAX_PAGES = env_int("CC_MAX_PAGES", 50)
CC_TIMEOUT = env_int("CC_TIMEOUT", 20)
CC_THROTTLE_S = env_float("CC_THROTTLE_S", 0.8)
CC_DEBUG = env_int("CC_DEBUG", 0) == 1
CC_VERIFY_MODE = env_int("CC_VERIFY_MODE", 0) == 1

CC_MIN_MATCH_SCORE = env_int("CC_MIN_MATCH_SCORE", 65)
CC_MIN_NET_EUR = env_float("CC_MIN_NET_EUR", 20.0)
CC_MIN_NET_ROI = env_float("CC_MIN_NET_ROI", 0.08)
CLOSE_HAIRCUT = env_float("CLOSE_HAIRCUT", 0.90)

CC_GOOD_BRANDS_TARGET = env_int("CC_GOOD_BRANDS_TARGET", 60)

ALLOW_FAKE_RISK = {s.strip().lower() for s in env_str("CC_ALLOW_FAKE_RISK", "low,medium").split(",") if s.strip()}
CC_STRICT_KEYWORDS = env_int("CC_STRICT_KEYWORDS", 1) == 1
CC_DISCOVERY_MODE = env_int("CC_DISCOVERY_MODE", 1) == 1
CC_VISION_HINTS_PATH = env_str("CC_VISION_HINTS_PATH", "vision_annotations.json")

# ── VISION VIA CLAUDE API ────────────────────────────────────────────────────
ANTHROPIC_API_KEY   = env_str("ANTHROPIC_API_KEY", "")
CC_VISION_ENABLED   = env_int("CC_VISION_ENABLED", 1) == 1 and bool(ANTHROPIC_API_KEY)
CC_VISION_MODEL     = env_str("CC_VISION_MODEL", "claude-sonnet-4-20250514")
CC_VISION_TIMEOUT   = env_int("CC_VISION_TIMEOUT", 30)
CC_VISION_MAX_CALLS = env_int("CC_VISION_MAX_CALLS", 50)  # safety cap per run


@dataclass
class Listing:
    title: str
    price_eur: float
    url: str
    store: str
    cond: str
    availability: str
    price_confidence: int = 0
    description: str = ""   # full page text — used for movement/ref detection
    image_url: str = ""     # main product image URL for vision analysis

# ─────────────────────────────────────────────────────────────────────────────
# VISION ANALYSIS via Claude API
# Called only on shortlisted candidates (~40 per run). Cost: ~€0.003/listing.
# Returns structured dict or None on failure.
# ─────────────────────────────────────────────────────────────────────────────
import base64 as _b64
import urllib.request as _urlreq

_VISION_PROMPT = """You are a professional watch authenticator and identifier.
Analyze this watch listing image and return ONLY valid JSON with these exact keys:
{
  "brand": "brand name or null",
  "model": "model name or null",
  "reference": "reference number or null",
  "movement": "automatic" | "manual" | "quartz" | "kinetic" | "solar" | "unknown",
  "case_size_mm": number or null,
  "condition": "mint" | "excellent" | "good" | "fair" | "poor" | "unknown",
  "confidence": 0-100,
  "notes": "any important details in one sentence"
}
Return ONLY the JSON object, no other text."""

_vision_call_count = 0

def analyze_listing_image(image_url: str, listing_title: str) -> Optional[Dict[str, Any]]:
    """Call Claude vision API to identify the watch in a CC listing image."""
    global _vision_call_count
    if not CC_VISION_ENABLED or not image_url:
        return None
    if _vision_call_count >= CC_VISION_MAX_CALLS:
        return None
    try:
        # Download image
        req = _urlreq.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
        with _urlreq.urlopen(req, timeout=10) as resp:
            img_bytes = resp.read()
        if len(img_bytes) < 1000:  # skip tiny/broken images
            return None
        # Detect media type
        media_type = "image/jpeg"
        if image_url.lower().endswith(".png"):  media_type = "image/png"
        elif image_url.lower().endswith(".webp"): media_type = "image/webp"
        img_b64 = _b64.b64encode(img_bytes).decode()

        # Call Claude API
        import json as _json
        import urllib.request as _ur
        payload = {
            "model": CC_VISION_MODEL,
            "max_tokens": 300,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": img_b64
                    }},
                    {"type": "text", "text": f"Listing title: {listing_title}\n\n{_VISION_PROMPT}"}
                ]
            }]
        }
        api_req = _ur.Request(
            "https://api.anthropic.com/v1/messages",
            data=_json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            },
            method="POST"
        )
        with _ur.urlopen(api_req, timeout=CC_VISION_TIMEOUT) as resp:
            result = _json.loads(resp.read())

        _vision_call_count += 1
        raw = result.get("content", [{}])[0].get("text", "").strip()
        # Strip any accidental markdown fences
        raw = raw.strip("` \n").removeprefix("json").strip()
        vision_data = _json.loads(raw)
        vision_data["_raw_used"] = True
        return vision_data

    except Exception as e:
        return {"error": str(e)[:80], "_raw_used": False}


def apply_vision_to_match(listing, target, match, hits, targets):
    """
    Use vision output to:
    1. Reject automatic targets when watch is quartz/solar/kinetic
    2. Find a better target if vision identifies a specific model
    3. Adjust match confidence based on vision identification
    Returns (target, match, hits, vision_data)
    """
    if not CC_VISION_ENABLED or not listing.image_url:
        return target, match, hits, None

    vision = analyze_listing_image(listing.image_url, listing.title)
    if not vision or "error" in vision:
        return target, match, hits, vision

    movement = (vision.get("movement") or "unknown").lower()
    v_brand   = (vision.get("brand") or "").lower().strip()
    v_model   = (vision.get("model") or "").lower().strip()
    v_ref     = (vision.get("reference") or "").lower().strip()
    v_conf    = int(vision.get("confidence") or 0)

    # ── Rule 1: Quartz/kinetic/solar → reject automatic-only targets ──────────
    MECHANICAL = {"automatic", "manual"}
    ELECTRONIC  = {"quartz", "kinetic", "solar"}
    target_notes  = (target.get("catawiki_estimate") or {}).get("notes", "").lower()
    target_family = (target.get("family") or "").lower()
    target_id_low = str(target.get("id","")).lower()
    # Check if target is for a mechanical watch using multiple signals
    AUTO_SIGNALS = ["automatic","auto","automático","powermatic","movement auto",
                    "calibre auto","mechanical","self-winding","manufacture"]
    is_auto_target = (
        any(w in target_family for w in AUTO_SIGNALS) or
        any(w in target_id_low  for w in ["_auto","powermatic","mechanical"]) or
        any(w in target_notes   for w in AUTO_SIGNALS)
    )

    if movement in ELECTRONIC and is_auto_target and v_conf >= 60:
        # Try to find a generic/quartz target instead
        quartz_fallback = next(
            (t for t in targets
             if t.get("brand","").lower() == v_brand
             and str(t.get("id","")).endswith("_GENERIC")),
            None
        )
        if quartz_fallback:
            target = quartz_fallback
            match  = max(40, match - 20)
            hits   = 0
        else:
            # No good fallback — downgrade match confidence significantly
            match = max(20, match - 30)

    # ── Rule 2: Vision identifies specific model → try to find better target ──
    elif v_conf >= 75 and v_model:
        search_text = f"{v_brand} {v_model} {v_ref}".strip()
        better_target, better_match, better_hits = best_target(search_text, targets)
        if better_target and better_target.get("id") != target.get("id") and better_match > match:
            target = better_target
            match  = better_match
            hits   = better_hits

    # ── Rule 3: Vision brand ≠ listing brand → flag as mismatch ──────────────
    elif v_conf >= 80 and v_brand and target:
        t_brand = canon(target.get("brand",""))
        if v_brand and t_brand and v_brand not in t_brand and t_brand not in v_brand:
            match = max(10, match - 40)  # heavy penalty for brand mismatch

    return target, match, hits, vision




# -----------------------------
# TELEGRAM
# -----------------------------
def chunk_text(text: str, max_len: int) -> List[str]:
    if len(text) <= max_len:
        return [text]
    out = []
    cur = []
    cur_len = 0
    for line in text.splitlines(True):
        if cur_len + len(line) > max_len and cur:
            out.append("".join(cur))
            cur = []
            cur_len = 0
        cur.append(line)
        cur_len += len(line)
    if cur:
        out.append("".join(cur))
    return out


def telegram_send(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for chunk in chunk_text(text, 3800):
        try:
            requests.post(
                url,
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": chunk,
                    "disable_web_page_preview": True,
                },
                timeout=20,
            )
        except Exception:
            pass


# -----------------------------
# HTTP / PARSING
# -----------------------------
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": env_str("CC_USER_AGENT", DEFAULT_USER_AGENT),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.7",
            "Connection": "keep-alive",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
    )
    return s


def polite_sleep(seconds: float) -> None:
    if seconds and seconds > 0:
        time.sleep(seconds)


def fetch(url: str, session: requests.Session) -> Optional[requests.Response]:
    for attempt in range(2):
        try:
            return session.get(url, timeout=CC_TIMEOUT)
        except Exception:
            if attempt == 0:
                polite_sleep(CC_THROTTLE_S)
    return None


def canon(s: str) -> str:
    s = s or ""
    s = s.lower().strip()
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def parse_price(text: str) -> Optional[float]:
    if not text:
        return None
    t = str(text).strip()
    t = t.replace("\u00a0", " ")
    t = t.replace("€", "").replace("eur", "").replace("EUR", "").strip()
    t = re.sub(r"[^0-9,.\-]", "", t)

    # Keep only one leading sign if present.
    t = re.sub(r"(?!^)-", "", t)

    if "," in t and "." in t:
        if t.rfind(",") > t.rfind("."):
            t = t.replace(".", "").replace(",", ".")
        else:
            t = t.replace(",", "")
    else:
        if "," in t and "." not in t:
            t = t.replace(",", ".")

    m = re.search(r"(-?\d+(?:\.\d+)?)", t)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def extract_product_urls(html: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    urls: List[str] = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href:
            continue
        if "/segunda-mano/" in href and href.endswith(".html"):
            full = href
            if full.startswith("/"):
                full = "https://www.cashconverters.es" + full
            if full.startswith("https://www.cashconverters.es/es/es/segunda-mano/"):
                urls.append(full)

    if not urls:
        rx = re.findall(r'(https://www\.cashconverters\.es/es/es/segunda-mano/[^"\']+?\.html)', html)
        urls.extend(rx)

    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _extract_price_from_jsonld(soup: BeautifulSoup) -> Optional[float]:
    scripts = soup.find_all("script", attrs={"type": "application/ld+json"})
    for sc in scripts:
        raw = sc.string or sc.get_text(strip=True)
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue

        nodes = data if isinstance(data, list) else [data]
        for node in nodes:
            if not isinstance(node, dict):
                continue

            graph = node.get("@graph")
            nodes2 = graph if isinstance(graph, list) else [node]

            for n in nodes2:
                if not isinstance(n, dict):
                    continue
                typ = str(n.get("@type", "")).lower()
                if "product" not in typ and typ != "product":
                    continue

                offers = n.get("offers")
                if isinstance(offers, dict):
                    price = offers.get("price")
                    if price is None and isinstance(offers.get("priceSpecification"), dict):
                        price = offers["priceSpecification"].get("price")
                    p = parse_price(price)
                    if p and p > 0:
                        return p
                elif isinstance(offers, list):
                    for off in offers:
                        if not isinstance(off, dict):
                            continue
                        price = off.get("price")
                        if price is None and isinstance(off.get("priceSpecification"), dict):
                            price = off["priceSpecification"].get("price")
                        p = parse_price(price)
                        if p and p > 0:
                            return p
    return None


def _extract_price_from_meta(soup: BeautifulSoup) -> Optional[float]:
    meta_price = soup.find("meta", attrs={"property": "product:price:amount"})
    if meta_price and meta_price.get("content"):
        p = parse_price(meta_price.get("content", ""))
        if p and p > 0:
            return p

    meta_price2 = soup.find("meta", attrs={"itemprop": "price"})
    if meta_price2 and meta_price2.get("content"):
        p = parse_price(meta_price2.get("content", ""))
        if p and p > 0:
            return p
    return None


def _extract_price_from_dom(soup: BeautifulSoup) -> Optional[float]:
    selectors = [
        "[itemprop='price']",
        ".pdp-price", ".product-price", ".price", ".value",
        ".prices .value", ".product-detail-price", ".js-product-price",
        "[data-price]", "[data-qa*='price']", "[class*='price']",
    ]
    best_price = None
    best_score = -10**18

    for sel in selectors:
        for el in soup.select(sel):
            class_name = canon(" ".join(el.get("class", [])))
            data_qa = canon(el.get("data-qa", ""))
            ctx = canon(el.get_text(" ", strip=True))

            for attr in ("content", "data-price", "data-value", "value"):
                if el.has_attr(attr):
                    p = parse_price(el.get(attr))
                    if p and p > 0:
                        score = _price_confidence_score(p, class_name, data_qa, ctx)
                        if score > best_score:
                            best_score = score
                            best_price = p
            txt = el.get_text(" ", strip=True)
            p = parse_price(txt)
            if p and p > 0:
                score = _price_confidence_score(p, class_name, data_qa, ctx)
                if score > best_score:
                    best_score = score
                    best_price = p
    return best_price


def _price_confidence_score(price: float, class_name: str, data_qa: str, ctx: str) -> int:
    score = 0

    anchor = f"{class_name} {data_qa} {ctx}"
    if "price" in anchor or "precio" in anchor or "pvp" in anchor:
        score += 35
    if "product" in anchor or "pdp" in anchor:
        score += 15
    if "old" in anchor or "before" in anchor or "tachado" in anchor:
        score -= 25
    if "envío" in anchor or "envio" in anchor or "shipping" in anchor:
        score -= 20

    if 15 <= price <= 20000:
        score += 20
    if price < 15 or price > 30000:
        score -= 100

    return score * 1000 + int(price)


def _extract_price_from_scripts(html: str) -> Optional[float]:
    if not html:
        return None

    indicator_terms = ("price", "precio", "sale", "offer", "current", "amount")
    reject_terms = ("shipping", "envio", "envío", "installment", "cuota")

    candidates = []
    regex = re.compile(r"(?:\"|')(?:price|salePrice|currentPrice|unitPrice|amount)(?:\"|')\s*:\s*(?:\"|')?([0-9][0-9\.,]{0,14})(?:\"|')?", re.IGNORECASE)

    for m in regex.finditer(html):
        raw = m.group(1)
        p = parse_price(raw)
        if not p or p <= 0 or p < 15 or p > 20000:
            continue

        start = max(0, m.start() - 120)
        end = min(len(html), m.end() + 120)
        ctx = canon(html[start:end])

        score = 0
        if any(term in ctx for term in indicator_terms):
            score += 40
        if any(term in ctx for term in reject_terms):
            score -= 80
        if "price" in ctx or "precio" in ctx:
            score += 20
        if "sale" in ctx or "offer" in ctx:
            score += 10

        # mild preference for realistic retail ranges
        if 40 <= p <= 5000:
            score += 10

        candidates.append((score, p))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    best_score, best_price = candidates[0]
    if best_score < 0:
        return None
    return best_price


def _extract_price_from_regex(html: str) -> Optional[float]:
    if not html:
        return None

    pattern_after = re.compile(r"(.{0,80})(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)\s*€(.{0,80})", re.IGNORECASE | re.DOTALL)
    pattern_before = re.compile(r"(.{0,80})€\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)(.{0,80})", re.IGNORECASE | re.DOTALL)
    matches = list(pattern_after.finditer(html)) + list(pattern_before.finditer(html))
    if not matches:
        return None

    best_score = -10**18
    best_price = None

    for m in matches:
        left = canon(m.group(1))
        mid = m.group(2)
        right = canon(m.group(3))
        p = parse_price(mid)
        if p is None:
            continue

        if p < 15 or p > 20000:
            continue

        ctx = left + " " + right
        score = 0

        if "precio" in ctx or "price" in ctx or "pvp" in ctx:
            score += 50
        if "segunda-mano" in ctx or "segunda mano" in ctx:
            score += 10
        if "comprar" in ctx or "añadir" in ctx or "carrito" in ctx:
            score += 10

        if "envío" in ctx or "envio" in ctx:
            score -= 15
        if "cookie" in ctx or "privacidad" in ctx or "newsletter" in ctx:
            score -= 40

        if "," in str(m.group(2)) or "." in str(m.group(2)):
            score += 5

        score2 = score * 1000 + int(p)

        if score2 > best_score:
            best_score = score2
            best_price = p

    return best_price


def extract_price_best_effort(soup: BeautifulSoup, html: str) -> Optional[float]:
    p = _extract_price_from_jsonld(soup)
    if p and p > 0:
        return p
    p = _extract_price_from_meta(soup)
    if p and p > 0:
        return p
    p = _extract_price_from_dom(soup)
    if p and p > 0:
        return p
    p = _extract_price_from_scripts(html)
    if p and p > 0:
        return p
    p = _extract_price_from_regex(html)
    if p and p > 0:
        return p
    return None


def estimate_price_confidence(soup: BeautifulSoup, html: str) -> int:
    if _extract_price_from_jsonld(soup):
        return 92
    if _extract_price_from_meta(soup):
        return 88
    if _extract_price_from_dom(soup):
        return 76
    if _extract_price_from_scripts(html):
        return 68
    if _extract_price_from_regex(html):
        return 55
    return 20


def parse_detail_page(html: str, url: str) -> Listing:
    soup = BeautifulSoup(html, "lxml")

    title = ""
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(" ", strip=True)
    if not title:
        title = (soup.title.get_text(" ", strip=True) if soup.title else "") or ""
    title = canon(title)

    price = extract_price_best_effort(soup, html)
    price_confidence = estimate_price_confidence(soup, html)
    if price is None:
        price = 0.0

    store = ""
    m = re.search(r"\b(CC\d{3}_E\d+_\d)\b", html)
    if m:
        store = m.group(1)
    if not store:
        store = url.rsplit("/", 1)[-1].replace(".html", "")

    text_all = canon(soup.get_text(" ", strip=True))

    cond = ""
    if "estado" in text_all:
        m2 = re.search(r"estado\s+([a-záéíóúñ ]{3,25})", text_all)
        if m2:
            cond = canon(m2.group(1))[:30]
    if not cond:
        if "perfecto" in text_all or "impecable" in text_all:
            cond = "perfecto"
        elif "excelente" in text_all or "muy buen estado" in text_all:
            cond = "muy bueno"
        elif "bueno" in text_all:
            cond = "bueno"
        elif "usado" in text_all or "de uso" in text_all:
            cond = "usado"
        else:
            cond = "desconocido"

    availability = []
    if "envío" in text_all or "envio" in text_all:
        availability.append("envío")
    if "tienda" in text_all or "recogida" in text_all:
        availability.append("tienda")
    availability_str = "desconocido" if not availability else " + ".join(availability)

    # Extract description = search for "tipo de movimiento" in full text_all
    # CC's ficha técnica is an expandable section that may appear anywhere in page text.
    # We extract a focused window around the movement keyword if found,
    # otherwise use the last 800 chars where ficha técnica typically sits.
    description = ""
    mov_idx = text_all.find("tipo de movimiento")
    if mov_idx >= 0:
        description = text_all[max(0, mov_idx - 50): mov_idx + 200]
    else:
        # Fallback: last portion of page text contains ficha técnica
        description = text_all[-800:] if len(text_all) > 800 else text_all

    # Extract main product image URL
    image_url = ""
    for sel in ["img.product-image", "img[class*='product']", "img[class*='detail']",
                "div.product-images img", "figure img", "img[class*='main']"]:
        img = soup.select_one(sel)
        if img and img.get("src","").startswith("http"):
            image_url = img["src"]
            break
    if not image_url:
        # Try og:image meta tag
        og = soup.find("meta", property="og:image") or soup.find("meta", attrs={"name":"og:image"})
        if og and og.get("content","").startswith("http"):
            image_url = og["content"]
    if not image_url:
        # Last resort: first large-ish img with https
        for img in soup.find_all("img", src=True):
            src = img.get("src","")
            if src.startswith("https") and any(ext in src.lower() for ext in [".jpg",".jpeg",".webp",".png"]):
                image_url = src
                break

    return Listing(
        title=title,
        price_eur=float(price),
        url=url,
        store=store,
        cond=canon(cond),
        availability=availability_str,
        price_confidence=price_confidence,
        description=description,
        image_url=image_url,
    )


# -----------------------------
# TARGETS / MATCHING / SCORING
# -----------------------------
def load_targets(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "targets" in raw and isinstance(raw["targets"], list):
        targets = raw["targets"]
    elif isinstance(raw, list):
        targets = raw
    else:
        raise ValueError("target_list.json inválido: debe ser {targets:[...]} o lista")

    valid: List[Dict[str, Any]] = []
    for t in targets:
        if not isinstance(t, dict):
            continue
        if not t.get("id") or not t.get("brand") or not t.get("catawiki_estimate"):
            continue
        est = t.get("catawiki_estimate")
        if not isinstance(est, dict) or "p50" not in est:
            continue
        valid.append(t)

    if not valid:
        raise ValueError("No valid targets loaded from target_list.json")
    return valid


def extract_brand(title: str) -> Optional[str]:
    import re as _re
    t = canon(title)
    for b in sorted(REPUTABLE_BRANDS, key=lambda x: -len(x)):
        cb = canon(b)
        # Word-boundary check: prevents "oris" matching inside "motorista", etc.
        if _re.search(r'(?<![a-z])' + _re.escape(cb) + r'(?![a-z])', t):
            if b in ("tag", "heuer"):
                return "tag heuer"
            if b == "baume" or "baume" in b:
                return "baume & mercier"
            return canon(b)
    for b in sorted(BANNED_BRANDS, key=lambda x: -len(x)):
        if canon(b) in t:
            return canon(b)
    return None


def is_banned_brand(brand: Optional[str]) -> bool:
    if not brand:
        return False
    return canon(brand) in {canon(x) for x in BANNED_BRANDS}


def is_reputable_brand(brand: Optional[str]) -> bool:
    if not brand:
        return False
    b = canon(brand)
    rep = {canon(x) for x in REPUTABLE_BRANDS}
    return b in rep or (b == "tag heuer" and ("tag heuer" in rep))


def has_any(text: str, terms: List[str]) -> bool:
    t = canon(text)
    return any(canon(x) in t for x in terms)


def violates_must_exclude(title: str, target: Dict[str, Any]) -> bool:
    t = canon(title)
    ex = [canon(x) for x in (target.get("must_exclude") or []) if isinstance(x, str)]
    return any(x and x in t for x in ex)


def model_keyword_hits(title: str, target: Dict[str, Any]) -> int:
    t = canon(title)
    kws = [canon(x) for x in (target.get("model_keywords") or []) if isinstance(x, str)]
    hits = 0
    for kw in kws:
        if not kw:
            continue
        if len(kw) <= 2 and not kw.isdigit():
            continue
        if keyword_in_title(kw, t):
            hits += 1
    return hits


def keyword_in_title(keyword: str, title: str) -> bool:
    return core_keyword_in_title(keyword, title)


def has_chrono_evidence(title: str, target: Dict[str, Any]) -> bool:
    t = canon(title)
    target_id = str(target.get("id", "")).upper()

    if target_id == "SEIKO_CHRONOGRAPH_GENERIC" or target_id == "SEIKO_VTG_CHRONOGRAPH_GENERIC":
        seiko_terms = [
            "chrono", "chronograph", "crono", "tachymeter", "tachimetro", "subdial",
            "7n32", "7t", "7a", "8t", "8r", "6s", "6138", "6139", "7016", "7018",
        ]
        return any(term in t for term in seiko_terms)

    if target_id == "TISSOT_CHRONOGRAPH_GENERIC" or target_id == "TISSOT_VTG_CHRONOGRAPH_GENERIC":
        tissot_terms = [
            "chrono", "chronograph", "crono", "tachymeter", "tachimetre", "subdial",
            "prc 200", "prs 200", "prs516", "prs 516", "v8", "couturier",
            "c01", "valjoux", "7750", "eta 7750", "lemania",
        ]
        return any(term in t for term in tissot_terms)

    return True


def target_requires_chrono_evidence(target: Dict[str, Any]) -> bool:
    target_id = str(target.get("id", "")).upper()
    return target_id in {
        "SEIKO_CHRONOGRAPH_GENERIC",
        "SEIKO_VTG_CHRONOGRAPH_GENERIC",
        "TISSOT_CHRONOGRAPH_GENERIC",
        "TISSOT_VTG_CHRONOGRAPH_GENERIC",
    }


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 1: Movement + detail detection from page description text
# No API calls — purely regex/keyword scan of the scraped HTML text.
# ─────────────────────────────────────────────────────────────────────────────
def detect_movement_from_text(title: str, description: str) -> str:
    """
    Returns: 'automatic' | 'manual' | 'quartz' | 'solar' | 'kinetic' | 'unknown'
    Uses page description text (CC ficha técnica included in text_all).
    """
    text = canon(title + " " + description)

    # Quartz/electronic signals — check first (most common false positive source)
    if any(w in text for w in [
        "cuarzo", "quartz", "quarzo", "battery", "bateria", "batería",
        "solar", "kinetic", "eco-drive", "eco drive", "light powered",
        "radio controlled", "radio-controlled", "atomic",
        # CC ficha técnica specific
        "tipo de movimiento: cuarzo", "tipo de movimiento : cuarzo",
    ]):
        if "solar" in text: return "solar"
        if "kinetic" in text: return "kinetic"
        return "quartz"

    # Manual wind
    if any(w in text for w in [
        "cuerda manual", "manual wind", "hand wind", "carga manual",
        "remontage manuel", "carica manuale",
        # CC ficha técnica
        "tipo de movimiento: cuerda", "tipo de movimiento : cuerda",
    ]):
        return "manual"

    # Automatic signals
    if any(w in text for w in [
        "automático", "automatico", "automatic", "automat",
        "self-winding", "self winding", "rotor", "powermatic",
        "calibre automático", "movimiento automático",
        # CC ficha técnica
        "tipo de movimiento: automático", "tipo de movimiento : automático",
        "tipo de movimiento: automatico",
        # Common calibre markers
        "eta 28", "eta 25", "valjoux", "miyota", "nh35", "nh36",
        "7s26", "7s36", "4r35", "4r36", "6r15",
        "srp", "srpc", "srpe",   # Seiko Prospex automatic refs
        "powermatic 80", "pm80",
    ]):
        return "automatic"

    # ── Reference-number based detection ─────────────────────────────────────
    # Seiko quartz calibres: 7T42, 7T32, 7T82, 7T92, 5Y22, 5Y23, V739, V8xx
    import re as _re_ref
    if _re_ref.search(r'\b7t[348][0-9]\b|\b5y2[0-9]\b|\bv739\b|\bv8[0-9]\b', text):
        return "quartz"
    # Tissot quartz: T120.417 (Seastar Chrono), V8 model, PR50
    if _re_ref.search(r't120[._]41[0-9]|pr50|tissot v8\b', text):
        return "quartz"
    # Seiko 5 / SNK family → automatic
    if _re_ref.search(r'\bsnk[a-z0-9]{2,6}\b|\bsnab\b', text):
        return "automatic"
    # Seiko vintage automatic calibres: 76xx (dress), 61xx, 66xx, 56xx, 52xx
    if _re_ref.search(r'\b7[6][0-9]{2}[- ][0-9]|\b6[1256][0-9]{2}[- ][0-9]', text):
        return "automatic"
    # Hamilton vintage automatics: H10, H694 family often labelled without "automatic"
    # Longines vintage: L4.xxx, L6.xxx (no movement in title often)
    # Orient: RA-xx → automatic
    if _re_ref.search(r'\bra-[a-z]{2}[0-9]', text):
        return "automatic"

    return "unknown"


def detect_extras_from_text(description: str) -> Dict[str, bool]:
    """Detect box/papers/service from description text (more reliable than title)."""
    t = canon(description)
    return {
        "has_box":     any(w in t for w in ["caja original","caja","estuche","box","boîte","scatola"]),
        "has_papers":  any(w in t for w in ["papeles","papers","garantia","garantía","warranty","documentos","certificado"]),
        "has_service": any(w in t for w in ["revisado","serviced","revision","revisión","overhaul","repasado"]),
        "is_nos":      any(w in t for w in ["sin usar","nuevo","nos","new old stock","nunca usado"]),
        "is_full_set": any(w in t for w in ["full set","set completo","completo con","caja y papeles"]),
    }


AUTO_TARGET_SIGNALS = [
    "automatic","auto","automático","powermatic","mechanical","self-winding",
    "calibre auto","movement auto",
]

def target_requires_mechanical(target: Dict[str, Any]) -> bool:
    """Returns True if this target is specifically for automatic/manual watches."""
    family  = (target.get("family") or "").lower()
    tid     = str(target.get("id","")).lower()
    notes   = (target.get("catawiki_estimate") or {}).get("notes","").lower()
    return any(
        w in family or w in tid or w in notes
        for w in AUTO_TARGET_SIGNALS
    )


def apply_description_filter(listing, target, match, hits, targets):
    """
    Level 1: use page description text to:
    1. Detect movement type
    2. Reject quartz watches matched to automatic targets
    3. Find a more appropriate target if needed
    Returns: (target, match, hits, movement_type)
    """
    movement = detect_movement_from_text(listing.title, listing.description)

    if movement in ("quartz", "solar", "kinetic") and target_requires_mechanical(target):
        # Try to find a generic target for this brand instead
        brand_lower = (target.get("brand") or "").lower()
        generic = next(
            (t for t in targets
             if t.get("brand","").lower() == brand_lower
             and str(t.get("id","")).upper().endswith("_GENERIC")),
            None
        )
        if generic:
            return generic, max(35, match - 15), 0, movement
        else:
            # No generic target → heavy match penalty, will likely not pass threshold
            return target, max(10, match - 35), hits, movement

    return target, match, hits, movement



def compute_match_score(title: str, target: Dict[str, Any]) -> int:
    t = canon(title)
    score = 0

    import re as _re2
    brand = canon(target.get("brand", ""))
    if brand and _re2.search(r'(?<![a-z])' + _re2.escape(brand) + r'(?![a-z])', t):
        score += 35

    must_in = [canon(x) for x in (target.get("must_include") or []) if isinstance(x, str)]
    if must_in:
        present = sum(1 for x in must_in if x and x in t)
        if present == len(must_in):
            score += 35
        else:
            score += int(35 * (present / max(1, len(must_in))))
    else:
        score += 10

    hits = model_keyword_hits(title, target)
    if hits > 0:
        score += min(25, hits * 8)

    if has_any(t, GOOD_COND_TERMS):
        score += 5
    if has_any(t, BAD_COND_TERMS):
        score -= 25

    return max(0, min(100, score))


def best_target(title: str, targets: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], int, int]:
    """
    Returns (target, match_score, kw_hits).
    With CC_STRICT_KEYWORDS enabled: if target has model_keywords -> require kw_hits >= 1,
    except for *_GENERIC targets.
    """
    best_t = None
    best_s = -1
    best_hits = 0

    for trg in targets:
        if violates_must_exclude(title, trg):
            continue

        hits = model_keyword_hits(title, trg)

        kws = trg.get("model_keywords") or []
        target_id = str(trg.get("id", "")).upper()
        is_generic_target = target_id.endswith("_GENERIC")

        if isinstance(kws, list) and len(kws) > 0 and not is_generic_target:
            if hits < 1:
                continue  # Non-generic targets ALWAYS require ≥1 keyword hit

        s = compute_match_score(title, trg)
        # Prefer higher score; on equal score, prefer specific over generic
        best_is_generic = str((best_t or {}).get("id","")).upper().endswith("_GENERIC")
        is_better = (s > best_s) or (s == best_s and best_is_generic and not is_generic_target)
        if is_better:
            best_s = s
            best_t = trg
            best_hits = hits

    return best_t, best_s, best_hits


def condition_adjustment(cond: str, title: str) -> float:
    c = canon(cond) + " " + canon(title)
    if "perfecto" in c or "impecable" in c or "como nuevo" in c:
        return 1.05
    if "excelente" in c or "muy bueno" in c or "muy buen estado" in c:
        return 1.03
    if "bueno" in c:
        return 1.00
    if "usado" in c or "de uso" in c or "desgaste" in c:
        return 0.92
    if has_any(c, BAD_COND_TERMS):
        return 0.80
    return 1.00


def condition_score_from_text(cond: str, title: str) -> int:
    adj = condition_adjustment(cond, title)
    if adj >= 1.05:
        return 20
    if adj >= 1.03:
        return 12
    if adj >= 1.0:
        return 5
    if adj <= 0.80:
        return -35
    if adj <= 0.92:
        return -20
    return 0


def estimate_close_eur(target: Dict[str, Any], cond: str, title: str) -> Tuple[float, Dict[str, bool]]:
    est = target.get("catawiki_estimate", {})
    p50 = float(est.get("p50", 0.0))
    p75 = float(est.get("p75", p50))
    triggers = target.get("p75_triggers_any") or []
    cscore = condition_score_from_text(cond, title)
    close, flags = estimate_close_price(
        p50=p50,
        p75=p75,
        detail_text=title + " " + cond,
        condition_score=cscore,
        haircut=float(CLOSE_HAIRCUT),
        triggers=triggers,
    )
    return close, flags


def estimate_net(buy_eur: float, close_eur: float) -> Tuple[float, float]:
    """
    Net profit for a CashConverters purchase sold on Catawiki.
    CC ships for FREE to the seller (no shipping cost on buy side).
    Catawiki shipping arbitrage (+35€) still applies on the sell side.
    Formula: close * (1 - fee) + SHIP_ARB - buy
    """
    commission = close_eur * CATWIKI_COMMISSION_RATE
    commission_vat = commission * CATWIKI_COMMISSION_VAT
    fees = commission + commission_vat
    net = close_eur - fees + SHIP_ARB_EUR - buy_eur
    roi = net / max(1e-9, buy_eur)
    return round(net, 2), round(roi, 4)


def estimate_chrono24(buy_eur: float, catawiki_p50: float) -> Tuple[float, float, float]:
    """
    Alternative net profit if sold on Chrono24 instead of Catawiki.
    C24 fee: 6.5%. No shipping arbitrage (separate from Catawiki structure).
    C24 BIN price estimated as catawiki_p50 * CHRONO24_PRICE_MULT (~1.20).
    Returns: (c24_close_est, c24_net, c24_roi)
    """
    c24_close = round(catawiki_p50 * CHRONO24_PRICE_MULT, 2)
    c24_net   = round(c24_close * (1 - CHRONO24_FEE_RATE) - buy_eur, 2)
    c24_roi   = round(c24_net / max(1e-9, buy_eur), 4)
    return c24_close, c24_net, c24_roi


def risk_allowed(target: Dict[str, Any]) -> bool:
    r = canon(str(target.get("risk", "medium")))
    return r in ALLOW_FAKE_RISK


# -----------------------------
# MAIN
# -----------------------------
def run() -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    session = make_session()
    vision_hints = load_vision_annotations(CC_VISION_HINTS_PATH)
    model_master = load_model_master()
    target_stats = load_target_stats()

    try:
        targets = load_targets("target_list.json")
    except Exception as e:
        telegram_send(
            "❌ TIMELAB CashConverters scanner: target_list.json inválido (no se pudieron cargar targets válidos)\n\n"
            f"error: {type(e).__name__}: {e}"
        )
        return



    diag: Dict[str, Any] = {
        "scanned": 0,
        "page_bad": 0,
        "brands": {"reputable": 0, "banned": 0, "not_reputable": 0, "no_brand": 0},
        "passed": {"match_ok": 0, "net_ok": 0},
        "dropped": {"ladies": 0, "smartwatch": 0, "no_kw_hit": 0},
        "thresholds": {
            "match>=": CC_MIN_MATCH_SCORE,
            "net>=": CC_MIN_NET_EUR,
            "roi>=": CC_MIN_NET_ROI,
            "haircut": CLOSE_HAIRCUT,
            "strict_kw": 1 if CC_STRICT_KEYWORDS else 0,
        },
        "stop": {"max_items": CC_MAX_ITEMS, "good_brands_target": CC_GOOD_BRANDS_TARGET},
        "targets": {"items": len(targets), "valid": len(targets)},
        "verify_mode": 1 if CC_VERIFY_MODE else 0,
    }

    seen_urls = set()
    listing_urls: List[str] = []
    reputable_seen = 0

    for page_idx in range(CC_MAX_PAGES):
        if len(listing_urls) >= CC_MAX_ITEMS:
            break

        start = page_idx * CC_PAGE_SIZE
        page_url = f"{BASE_LISTING_URL}&sz={CC_PAGE_SIZE}&start={start}"
        r = fetch(page_url, session)
        if not r or r.status_code != 200 or not r.text or len(r.text) < 2000:
            diag["page_bad"] += 1
            break

        urls = extract_product_urls(r.text)
        polite_sleep(CC_THROTTLE_S)

        if not urls:
            break

        for u in urls:
            if u in seen_urls:
                continue
            seen_urls.add(u)
            listing_urls.append(u)
            if len(listing_urls) >= CC_MAX_ITEMS:
                break

    candidates: List[Dict[str, Any]] = []
    debug_rejected: List[Dict[str, Any]] = []

    for u in listing_urls:
        if diag["scanned"] >= CC_MAX_ITEMS:
            break

        r = fetch(u, session)
        if not r or r.status_code != 200 or not r.text:
            diag["page_bad"] += 1
            polite_sleep(CC_THROTTLE_S)
            continue

        listing = parse_detail_page(r.text, u)
        polite_sleep(CC_THROTTLE_S)
        diag["scanned"] += 1

        if has_any(listing.title, LADIES_TERMS):
            diag["dropped"]["ladies"] += 1
            continue
        if has_any(listing.title, SMARTWATCH_TERMS):
            diag["dropped"]["smartwatch"] += 1
            continue

        b = extract_brand(listing.title)
        if b is None:
            diag["brands"]["no_brand"] += 1
            continue
        if is_banned_brand(b):
            diag["brands"]["banned"] += 1
            continue
        if not is_reputable_brand(b):
            diag["brands"]["not_reputable"] += 1
            continue

        diag["brands"]["reputable"] += 1
        reputable_seen += 1

        if has_any(listing.title, BAD_COND_TERMS):
            continue

        # Match on title + description
        match_text = listing.title
        if listing.description:
            match_text = listing.title + " " + listing.description[:300]
        target, match, hits = best_target(match_text, targets)
        identity = resolve_listing_identity(listing.title, summarize_visual_hints(vision_hints.get(listing.url, {})), model_master)

        # ── LEVEL 1: description-based movement filter ───────────────────────
        detected_movement = "unknown"
        vision_data = None  # initialize early; updated later if vision is invoked
        if target is not None:
            target, match, hits, detected_movement = apply_description_filter(
                listing, target, match, hits, targets
            )

        # NOTE: identity target override DISABLED — model_master.json only has 5 Tissot models.
        # Enabling it caused all Seiko watches to be forced into SEIKO_PROSPEX_TURTLE_AUTO
        # with its p50=520, giving completely fictitious profit estimates.
        # Re-enable when model_master is properly populated with all targets.
        # if identity.get("target_id"):
        #     forced = next(...)

        if target is None:
            diag["dropped"]["no_kw_hit"] += 1
            if CC_DISCOVERY_MODE:
                family = detect_discovery_family(listing.title)
                if family:
                    debug_rejected.append({"title": listing.title, "price": listing.price_eur, "url": listing.url, "reason": f"discovery:{family}"})
            continue

        brand_ctx = detect_brand_ambiguity(listing.title, str(target.get("brand", "")))
        if brand_ctx.get("movement_brand_contamination"):
            if CC_VERIFY_MODE:
                debug_rejected.append({
                    "title": listing.title,
                    "price": listing.price_eur,
                    "url": listing.url,
                    "target": target.get("id"),
                    "reason": "movement_brand_contamination",
                })
            continue

        if match < CC_MIN_MATCH_SCORE:
            if CC_VERIFY_MODE:
                debug_rejected.append(
                    {
                        "title": listing.title,
                        "price": listing.price_eur,
                        "url": listing.url,
                        "match": match,
                        "kw_hits": hits,
                        "vision":   vision_data,
                        "movement": detected_movement,
                        "target": target.get("id"),
                        "cond": listing.cond,
                    }
                )
            continue

        diag["passed"]["match_ok"] += 1

        if not risk_allowed(target):
            continue

        if target_requires_chrono_evidence(target) and not has_chrono_evidence(listing.title, target):
            if CC_VERIFY_MODE:
                debug_rejected.append(
                    {
                        "title": listing.title,
                        "price": listing.price_eur,
                        "url": listing.url,
                        "match": match,
                        "kw_hits": hits,
                        "vision":   vision_data,
                        "movement": detected_movement,
                        "target": target.get("id"),
                        "cond": listing.cond,
                        "reason": "missing_chrono_evidence",
                    }
                )
            continue


        close_est, listing_flags = estimate_close_eur(target, listing.cond, listing.title)
        close_est = apply_comparables_engine_cc(
            legacy_close=close_est,
            listing=listing,
            target=target,
            vision_data=vision_data,
            detected_movement=detected_movement,
        )

        max_buy = target.get("max_buy_eur")
        if isinstance(max_buy, (int, float)) and listing.price_eur > float(max_buy):
            if CC_VERIFY_MODE:
                debug_rejected.append(
                    {
                        "title": listing.title,
                        "price": listing.price_eur,
                        "url": listing.url,
                        "match": match,
                        "kw_hits": hits,
                        "vision":   vision_data,
                        "movement": detected_movement,
                        "target": target.get("id"),
                        "cond": listing.cond,
                        "reason": "over_max_buy",
                    }
                )
            continue

        shipping = 0.0  # CC ships for free to seller
        net, roi = estimate_net(listing.price_eur, close_est)

        # BUG-3 FIX: Chrono24 alternative estimate (only for watches above threshold)
        est_block = target.get("catawiki_estimate", {})
        p50_raw   = float(est_block.get("p50", 0) or 0)
        c24_close, c24_net, c24_roi = (0.0, 0.0, 0.0)
        suggest_c24 = False
        if listing.price_eur >= CHRONO24_MIN_BUY_EUR and p50_raw > 0:
            c24_close, c24_net, c24_roi = estimate_chrono24(listing.price_eur, p50_raw)
            suggest_c24 = c24_net > net  # True when C24 beats Catawiki

        if not (net >= CC_MIN_NET_EUR and roi >= CC_MIN_NET_ROI):
            if CC_VERIFY_MODE:
                debug_rejected.append(
                    {
                        "title": listing.title,
                        "price": listing.price_eur,
                        "url": listing.url,
                        "match": match,
                        "kw_hits": hits,
                        "vision":   vision_data,
                        "movement": detected_movement,
                        "target": target.get("id"),
                        "cond": listing.cond,
                        "net": net,
                        "roi": roi,
                        "close": close_est,
                        "reason": "net_or_roi_below",
                    }
                )
            continue

        diag["passed"]["net_ok"] += 1

        target_id = str(target.get("id", "")).upper()
        is_generic = target_id.endswith("_GENERIC")
        cscore = condition_score_from_text(listing.cond, listing.title)

        stats = target_stats.get(target_id, {})
        sample_size = int(stats.get("sample_size", 0))
        hammer_low = float(stats.get("p25", close_est * 0.9) or (close_est * 0.9))
        hammer_base = float(stats.get("p50", close_est) or close_est)
        hammer_high = float(stats.get("p75", close_est * 1.1) or (close_est * 1.1))
        trend_90d = float(stats.get("last_90d_trend", 0.0) or 0.0)
        opp_score = compute_opportunity_score(
            net=net,
            roi=roi,
            match_score=match,
            condition_score=cscore,
            brand_points=brand_score(str(target.get("risk", "medium"))),
            liquidity_points=liquidity_score(str(target.get("liquidity", "medium"))),
            ambiguity_penalty=int(brand_ctx.get("penalty", 0)),
        )
        bucket = bucket_from_score(opp_score, is_generic=is_generic, discovery=False)
        close_conf = derive_close_estimate_confidence(listing_flags, used_p75=(listing_flags.get("is_full_set") or listing_flags.get("is_nos")), condition_score=cscore)
        match_conf = derive_match_confidence(max(match, int(identity.get("final_match_score", 0))), hits, int(brand_ctx.get("penalty", 0)))
        valuation_conf = int((close_conf + min(95, sample_size * 8)) / 2)
        conf = compute_confidence(listing.price_confidence, match_conf, close_conf)

        # CC: prices non-negotiable, brand & match already validated → use "medium" as default
        # instead of "low" (which causes blanket SKIP when model_master doesn't know the model)
        # ── VISION ANALYSIS ──────────────────────────────────────────────────
        # Only called when listing has an image and has passed net/ROI filter.
        # Corrects quartz/auto mismatch before gate decision.
        vision_data = None
        if CC_VISION_ENABLED and listing.image_url:
            target, match, hits, vision_data = apply_vision_to_match(
                listing, target, match, hits, targets
            )
            # If vision downgraded match below threshold, drop
            if match < CC_MIN_MATCH_SCORE:
                diag["dropped"]["no_kw_hit"] += 1
                continue
            # Recalculate close/net with potentially new target
            close_est, listing_flags = estimate_close_eur(target, listing.cond, match_text)
            close_est = apply_comparables_engine_cc(
                legacy_close=close_est,
                listing=listing,
                target=target,
                vision_data=vision_data,
                detected_movement=detected_movement,
            )
            net, roi = estimate_net(listing.price_eur, close_est)
            if net < CC_MIN_NET_EUR or roi < CC_MIN_NET_ROI:
                continue

        # CC: precio fijo, marca validada, match ya superado.
        # Pasamos "medium"/False directamente — identity devuelve "low" (truthy) para
        # modelos no en model_master, y "low" or "medium" = "low" en Python (bug previo).
        gate = gate_decision("medium", sample_size, valuation_conf, net, roi, risk_allowed(target), False)
        if gate == "SKIP":
            continue
        if gate == "REVIEW":
            bucket = "REVIEW"
        if gate == "BUY" and bucket not in {"BUY", "SUPER_BUY"}:
            bucket = "BUY"

        visual = summarize_visual_hints(vision_hints.get(listing.url, {}))
        reason_text = explain_bucket(opp_score, bucket, net, roi, match, int(brand_ctx.get("penalty", 0)))

        candidates.append(
            {
                "listing": listing,
                "target": target,
                "match": match,
                "kw_hits": hits,
                "close_est": close_est,
                "net": net,
                "roi": roi,
                "bucket": bucket,
                "score": opp_score,
                "flags": listing_flags,
                "reason_flags": brand_ctx.get("reason_flags", []),
                "confidence": conf,
                "visual": visual,
                "explain": {
                    "target": target.get("id"),
                    "keyword_hits": hits,
                    "penalties": int(brand_ctx.get("penalty", 0)),
                    "comparables": {"p50": target.get("catawiki_estimate", {}).get("p50"), "p75": target.get("catawiki_estimate", {}).get("p75")},
                    "bucket_reason": reason_text,
                    "match_confidence_band": identity.get("match_confidence_band", "low"),
                    "reference_score": identity.get("reference_score", 0),
                    "spec_score": identity.get("spec_score", 0),
                },
                "valuation": {
                    "hammer_low": round(hammer_low, 2),
                    "hammer_base": round(hammer_base, 2),
                    "hammer_high": round(hammer_high, 2),
                    "sample_size": sample_size,
                    "valuation_confidence": valuation_conf,
                    "last_90d_trend": trend_90d,
                },
                "movement":  detected_movement,
                "chrono24": {
                    "suggested": suggest_c24,
                    "close_est": c24_close,
                    "net":       c24_net,
                    "roi":       c24_roi,
                },
            }
        )

    candidates.sort(key=lambda x: (x.get("score", 0), x["net"], x["match"]), reverse=True)
    top = candidates[:10]

    from datetime import datetime as _dt
    hora = _dt.now().strftime("%H:%M")
    sep_heavy = "━" * 26

    if not top:
        telegram_send(
            f"⌚ TIMELAB · Cash Converters · {hora}\n{sep_heavy}\n"
            f"😴 Sin oportunidades hoy\n"
            f"Escaneados: {diag['scanned']} · Con marca: {diag['brands']['reputable']}"
        )
    else:
        lines = [f"⌚ TIMELAB · Cash Converters · {hora}", sep_heavy, ""]
        for i, it in enumerate(top, 1):
            bucket      = it.get("bucket", "BUY")
            emoji       = "🟢" if "BUY" in bucket else "🟡"
            listing     = it["listing"]
            target_obj  = it["target"]
            flags       = it.get("flags", {})
            expl        = it.get("explain", {})
            val         = it.get("valuation", {})
            c24         = it.get("chrono24", {})
            conf        = it.get("confidence", {})

            # Flags as compact readable string instead of raw dict
            flag_parts = []
            if flags.get("has_box"):    flag_parts.append("📦caja")
            if flags.get("has_papers"): flag_parts.append("📄papeles")
            if flags.get("has_service"):flag_parts.append("🔧revisado")
            if flags.get("is_nos"):     flag_parts.append("✨NOS")
            if flags.get("is_full_set"):flag_parts.append("🎁full set")
            flags_str = " ".join(flag_parts) or "sin extras"

            # Confidence as compact string
            conf_str = (
                f"precio:{conf.get('price_confidence','?')} "
                f"match:{conf.get('match_confidence','?')} "
                f"cierre:{conf.get('close_estimate_confidence','?')}"
            )

            # Valuation range
            h_low  = val.get("hammer_low",  0)
            h_base = val.get("hammer_base", 0)
            h_high = val.get("hammer_high", 0)
            n      = val.get("sample_size", 0)
            trend  = val.get("last_90d_trend", 0)
            trend_str = f"+{trend:.0f}€" if trend > 0 else (f"{trend:.0f}€" if trend < 0 else "estable")

            # ── Human-readable watch name from title ──────────────────────
            # Capitalise title, strip the generic "reloj pulsera caballero/unisex" prefix
            import re as _re
            clean_title = listing.title
            for prefix in ["reloj pulsera caballero ","reloj pulsera unisex ",
                           "reloj pulsera senora ","reloj bolsillo ","reloj "]:
                if clean_title.lower().startswith(prefix):
                    clean_title = clean_title[len(prefix):]
                    break
            clean_title = clean_title.strip().title()

            # ── ROI colour ────────────────────────────────────────────────────
            roi_pct = it["roi"] * 100
            if roi_pct >= 120:   roi_emoji = "🚀"
            elif roi_pct >= 80:  roi_emoji = "💚"
            elif roi_pct >= 50:  roi_emoji = "🟡"
            else:                roi_emoji = "🔵"

            # ── Movement label ────────────────────────────────────────────────
            mov = it.get("movement", "unknown")
            MOV_LABEL = {
                "automatic": "⚙️ Automático",
                "manual":    "🔑 Cuerda manual",
                "quartz":    "🔋 Cuarzo",
                "solar":     "☀️ Solar",
                "kinetic":   "⚡ Kinetic",
                "unknown":   "❓ Mov. desconocido",
            }
            mov_label = MOV_LABEL.get(mov, "❓ " + mov)

            # ── Extras compact ────────────────────────────────────────────────
            extras = []
            if flags.get("has_box"):     extras.append("📦 caja")
            if flags.get("has_papers"):  extras.append("📄 papeles")
            if flags.get("has_service"): extras.append("🔧 revisado")
            if flags.get("is_nos"):      extras.append("✨ NOS")
            if flags.get("is_full_set"): extras.append("🎁 full set")
            extras_str = "  ".join(extras) if extras else "sin extras"

            # ── Cond readable ─────────────────────────────────────────────────
            COND_MAP = {
                "perfecto":"🌟 Perfecto","impecable":"🌟 Impecable",
                "muy bueno":"👍 Muy bueno","excelente":"👍 Excelente",
                "bueno":"✔️ Bueno","usado":"⚠️ Usado",
                "real":"✔️ Bueno","de uso":"⚠️ Usado",
            }
            cond_str = next((v for k,v in COND_MAP.items() if k in (listing.cond or "")), f"📋 {listing.cond}")

            # ── Trend arrow ───────────────────────────────────────────────────
            trend_arrow = "↑" if trend > 5 else ("↓" if trend < -5 else "→")

            # ── Build the card ────────────────────────────────────────────────
            sep = "──────────────────────────"
            lines.append(sep)
            lines.append(f"{i}) {emoji} {bucket}  ·  Cash Converters")
            lines.append(f"🏷 *{clean_title}*")
            lines.append("")
            lines.append(f"💶 Compra: *{listing.price_eur:.0f}€*")
            lines.append(f"🎯 Catawiki: ~*{it['close_est']:.0f}€*  ·  +{it['net']:.0f}€ neto  ·  {roi_emoji} {roi_pct:.0f}% ROI")
            if c24.get("close_est", 0) > 0:
                c24_label = "📈 Chrono24 (mejor):" if c24.get("suggested") else "📊 Chrono24 (alt.):"
                lines.append(f"{c24_label} ~{c24['close_est']:.0f}€  ·  +{c24['net']:.0f}€  ·  {c24['roi']*100:.0f}% ROI")
            lines.append("")
            lines.append(f"{mov_label}  ·  {cond_str}")
            if extras_str != "sin extras":
                lines.append(f"{extras_str}")
            if n > 0:
                lines.append(f"📊 Rango Catawiki: {h_low:.0f}–{h_base:.0f}–{h_high:.0f}€  ({n} ventas, {trend_arrow})")
            lines.append("")
            lines.append(f"🔗 {listing.url}")
            lines.append("")

        telegram_send("\n".join(lines).strip())

    if CC_DEBUG:
        dbg = []
        dbg.append(f"🧪 TIMELAB CC Debug — scanned:{diag['scanned']} | page_bad:{diag['page_bad']}")
        dbg.append(
            "brands: "
            f"reputable:{diag['brands']['reputable']} | "
            f"banned:{diag['brands']['banned']} | "
            f"not_reputable:{diag['brands']['not_reputable']} | "
            f"no_brand:{diag['brands']['no_brand']}"
        )
        dbg.append(f"passed: match_ok:{diag['passed']['match_ok']} | net_ok:{diag['passed']['net_ok']}")
        dbg.append(
            f"dropped: ladies:{diag['dropped']['ladies']} | "
            f"smartwatch:{diag['dropped']['smartwatch']} | "
            f"no_kw_hit:{diag['dropped']['no_kw_hit']}"
        )
        dbg.append(
            "thresholds: "
            f"match>={CC_MIN_MATCH_SCORE} | "
            f"net>={CC_MIN_NET_EUR} AND roi>={CC_MIN_NET_ROI} | "
            f"haircut:{CLOSE_HAIRCUT} | strict_kw:{1 if CC_STRICT_KEYWORDS else 0}"
        )
        dbg.append(f"stop: max_items:{CC_MAX_ITEMS} | good_brands_target:{CC_GOOD_BRANDS_TARGET}")
        dbg.append(f"targets: items:{len(targets)} | valid:{len(targets)}")
        dbg.append(f"verify_mode:{1 if CC_VERIFY_MODE else 0}")

        if CC_VERIFY_MODE and debug_rejected:
            dbg.append("\nTop candidates (even if rejected):")
            debug_rejected.sort(key=lambda x: (x.get("match", 0), x.get("price", 0)), reverse=True)
            for it in debug_rejected[:5]:
                extra = []
                extra.append(f"kw:{it.get('kw_hits')}")
                if "net" in it and "roi" in it:
                    extra.append(f"net:{it['net']:.2f}")
                    extra.append(f"roi:{it['roi']*100:.1f}%")
                if "reason" in it:
                    extra.append(f"reason:{it['reason']}")
                extra_s = " | " + " ".join(extra) if extra else ""
                dbg.append(
                    f"- match:{it.get('match')} target:{it.get('target')} "
                    f"price:{it.get('price'):.2f} {it.get('title')}{extra_s}"
                )
                dbg.append(f"  url:{it.get('url')}")

        telegram_send("\n".join(dbg).strip())


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        telegram_send("❌ TIMELAB CashConverters scanner crashed\n" f"{type(e).__name__}: {e}")
        raise
