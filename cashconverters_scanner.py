#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIMELAB — CashConverters ES scanner (requests + BeautifulSoup)

Fixes:
- Robust price extraction (JSON-LD -> meta -> DOM -> regex w/ context scoring)
"""

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup


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

CATWIKI_COMMISSION_RATE = float(os.getenv("CATWIKI_COMMISSION_RATE", "0.125"))
CATWIKI_COMMISSION_VAT = float(os.getenv("CATWIKI_COMMISSION_VAT", "0.21"))


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


@dataclass
class Listing:
    title: str
    price_eur: float
    url: str
    store: str
    cond: str
    availability: str


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
    chunks = chunk_text(text, 3800)
    for chunk in chunks:
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
            r = session.get(url, timeout=CC_TIMEOUT)
            return r
        except Exception:
            if attempt == 0:
                polite_sleep(CC_THROTTLE_S)
            continue
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
    # remove currency and thousands separators
    t = t.replace("\u00a0", " ")
    t = t.replace("€", "").replace("eur", "").replace("EUR", "")
    t = t.strip()

    # common format: "1.234,56" or "1,234.56" or "123,45"
    # Normalize: remove thousands separators then convert decimal comma to dot
    # First, keep only digits, comma, dot
    t = re.sub(r"[^0-9,.\-]", "", t)

    # If both comma and dot exist, decide decimal separator by last occurrence
    if "," in t and "." in t:
        if t.rfind(",") > t.rfind("."):
            # comma decimal, dots thousands
            t = t.replace(".", "")
            t = t.replace(",", ".")
        else:
            # dot decimal, commas thousands
            t = t.replace(",", "")
    else:
        # only comma -> decimal
        if "," in t and "." not in t:
            t = t.replace(",", ".")
        # only dot -> ok

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

        # JSON-LD can be dict or list
        nodes = data if isinstance(data, list) else [data]
        for node in nodes:
            if not isinstance(node, dict):
                continue

            # sometimes nested in @graph
            if "@graph" in node and isinstance(node["@graph"], list):
                nodes2 = node["@graph"]
            else:
                nodes2 = [node]

            for n in nodes2:
                if not isinstance(n, dict):
                    continue
                typ = str(n.get("@type", "")).lower()
                if "product" not in typ and typ != "product":
                    continue

                offers = n.get("offers")
                if isinstance(offers, dict):
                    price = offers.get("price") or offers.get("priceSpecification", {}).get("price")
                    if price is not None:
                        p = parse_price(price)
                        if p and p > 0:
                            return p
                elif isinstance(offers, list):
                    for off in offers:
                        if not isinstance(off, dict):
                            continue
                        price = off.get("price") or off.get("priceSpecification", {}).get("price")
                        p = parse_price(price)
                        if p and p > 0:
                            return p
    return None


def _extract_price_from_meta(soup: BeautifulSoup) -> Optional[float]:
    # meta property product:price:amount or itemprop=price
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
    # Try a bunch of likely selectors (site changes often)
    selectors = [
        "[itemprop='price']",
        ".pdp-price", ".product-price", ".price", ".value",
        ".prices .value", ".product-detail-price", ".js-product-price",
        "[data-price]", "[data-qa*='price']", "[class*='price']",
    ]
    for sel in selectors:
        for el in soup.select(sel):
            # data attributes first
            for attr in ("content", "data-price", "data-value", "value"):
                if el.has_attr(attr):
                    p = parse_price(el.get(attr))
                    if p and p > 0:
                        return p
            txt = el.get_text(" ", strip=True)
            p = parse_price(txt)
            if p and p > 0:
                return p
    return None


def _extract_price_from_regex(html: str) -> Optional[float]:
    """
    Last resort: find amounts like "109,90 €" and score by context.
    Avoid picking random numbers (IDs, sizes, times).
    """
    if not html:
        return None

    # Capture with euro sign or the word "€"
    # Keep some context around each match to score.
    pattern = re.compile(r"(.{0,80})(\d{1,4}(?:[.,]\d{2})?)\s*€(.{0,80})", re.IGNORECASE | re.DOTALL)
    matches = list(pattern.finditer(html))
    if not matches:
        return None

    best_score = -10**9
    best_price = None

    for m in matches:
        left = canon(m.group(1))
        mid = m.group(2)
        right = canon(m.group(3))
        p = parse_price(mid)
        if p is None:
            continue

        # Plausibility bounds for watches in CC
        if p < 15 or p > 20000:
            continue

        ctx = left + " " + right
        score = 0

        # Positive signals
        if "price" in ctx or "precio" in ctx or "pvp" in ctx:
            score += 50
        if "product" in ctx or "producto" in ctx or "detail" in ctx:
            score += 20
        if "add to cart" in ctx or "comprar" in ctx or "añadir" in ctx:
            score += 15
        if "second-hand" in ctx or "segunda-mano" in ctx or "segunda mano" in ctx:
            score += 10

        # Negative signals (avoid footer, shipping, unrelated)
        if "envío" in ctx or "envio" in ctx:
            score -= 15
        if "cookie" in ctx or "privacidad" in ctx:
            score -= 40
        if "rating" in ctx or "review" in ctx:
            score -= 20
        if "newsletter" in ctx:
            score -= 20

        # Favor prices that look like "xxx,xx" (retail)
        if "," in str(m.group(2)) or "." in str(m.group(2)):
            score += 5

        # Tie-breaker: choose larger plausible price if scores similar
        score2 = score * 1000 + int(p)

        if score2 > best_score:
            best_score = score2
            best_price = p

    return best_price


def extract_price_best_effort(soup: BeautifulSoup, html: str) -> Optional[float]:
    # 1) JSON-LD
    p = _extract_price_from_jsonld(soup)
    if p and p > 0:
        return p

    # 2) meta
    p = _extract_price_from_meta(soup)
    if p and p > 0:
        return p

    # 3) DOM selectors
    p = _extract_price_from_dom(soup)
    if p and p > 0:
        return p

    # 4) regex fallback
    p = _extract_price_from_regex(html)
    if p and p > 0:
        return p

    return None


def parse_detail_page(html: str, url: str) -> Listing:
    soup = BeautifulSoup(html, "lxml")

    # Title
    title = ""
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(" ", strip=True)
    if not title:
        title = (soup.title.get_text(" ", strip=True) if soup.title else "") or ""
    title = canon(title)

    # Price (FIXED)
    price = extract_price_best_effort(soup, html)
    if price is None:
        price = 0.0

    # Store code
    store = ""
    m = re.search(r"\b(CC\d{3}_E\d+_\d)\b", html)
    if m:
        store = m.group(1)
    if not store:
        store = url.rsplit("/", 1)[-1].replace(".html", "")

    # Condition
    cond = ""
    text_all = canon(soup.get_text(" ", strip=True))
    if "estado" in text_all:
        m2 = re.search(r"estado\s+([a-záéíóúñ ]{3,20})", text_all)
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

    # Availability
    availability = []
    if "envío" in text_all or "envio" in text_all:
        availability.append("envío")
    if "tienda" in text_all or "recogida" in text_all:
        availability.append("tienda")
    availability_str = "desconocido" if not availability else " + ".join(availability)

    return Listing(
        title=title,
        price_eur=float(price),
        url=url,
        store=store,
        cond=canon(cond),
        availability=availability_str,
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
    t = canon(title)
    for b in sorted(REPUTABLE_BRANDS, key=lambda x: -len(x)):
        if canon(b) in t:
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
    b = canon(brand)
    return b in {canon(x) for x in BANNED_BRANDS}


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


def compute_match_score(title: str, target: Dict[str, Any]) -> int:
    t = canon(title)
    score = 0

    brand = canon(target.get("brand", ""))
    if brand and brand in t:
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

    kws = [canon(x) for x in (target.get("model_keywords") or []) if isinstance(x, str)]
    if kws:
        present = sum(1 for x in kws if x and x in t)
        score += min(20, present * 7)

    if has_any(t, GOOD_COND_TERMS):
        score += 5
    if has_any(t, BAD_COND_TERMS):
        score -= 25

    return max(0, min(100, score))


def best_target(title: str, targets: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], int]:
    best_t = None
    best_s = -1
    for trg in targets:
        if violates_must_exclude(title, trg):
            continue
        s = compute_match_score(title, trg)
        if s > best_s:
            best_s = s
            best_t = trg
    return best_t, best_s


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


def estimate_close_eur(target: Dict[str, Any], cond: str, title: str) -> float:
    est = target.get("catawiki_estimate", {})
    p50 = float(est.get("p50", 0.0))
    base = p50 * float(CLOSE_HAIRCUT)
    return round(base * condition_adjustment(cond, title), 2)


def estimate_net(buy_eur: float, close_eur: float, shipping_eur: float = 0.0) -> Tuple[float, float]:
    commission = close_eur * CATWIKI_COMMISSION_RATE
    commission_vat = commission * CATWIKI_COMMISSION_VAT
    fees = commission + commission_vat
    net = close_eur - fees - buy_eur - shipping_eur
    denom = max(1e-9, (buy_eur + shipping_eur))
    roi = net / denom
    return round(net, 2), round(roi, 4)


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
        "thresholds": {
            "match>=": CC_MIN_MATCH_SCORE,
            "net>=": CC_MIN_NET_EUR,
            "roi>=": CC_MIN_NET_ROI,
            "haircut": CLOSE_HAIRCUT,
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

        if reputable_seen >= CC_GOOD_BRANDS_TARGET:
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

        target, match = best_target(listing.title, targets)
        if not target or match < CC_MIN_MATCH_SCORE:
            if target and CC_VERIFY_MODE:
                debug_rejected.append(
                    {"title": listing.title, "price": listing.price_eur, "url": listing.url,
                     "match": match, "target": target.get("id"), "cond": listing.cond}
                )
            continue
        diag["passed"]["match_ok"] += 1

        if not risk_allowed(target):
            continue

        close_est = estimate_close_eur(target, listing.cond, listing.title)

        max_buy = target.get("max_buy_eur")
        if isinstance(max_buy, (int, float)) and listing.price_eur > float(max_buy):
            if CC_VERIFY_MODE:
                debug_rejected.append(
                    {"title": listing.title, "price": listing.price_eur, "url": listing.url,
                     "match": match, "target": target.get("id"), "cond": listing.cond,
                     "reason": "over_max_buy"}
                )
            continue

        shipping = 0.0
        net, roi = estimate_net(listing.price_eur, close_est, shipping)

        if not (net >= CC_MIN_NET_EUR or roi >= CC_MIN_NET_ROI):
            if CC_VERIFY_MODE:
                debug_rejected.append(
                    {"title": listing.title, "price": listing.price_eur, "url": listing.url,
                     "match": match, "target": target.get("id"), "cond": listing.cond,
                     "net": net, "roi": roi, "close": close_est,
                     "reason": "net_or_roi_below"}
                )
            continue

        diag["passed"]["net_ok"] += 1

        candidates.append(
            {"title": listing.title, "buy": listing.price_eur, "close": close_est,
             "net": net, "roi": roi, "match": match, "cond": listing.cond,
             "disp": listing.availability, "store": listing.store, "url": listing.url,
             "target": target.get("id")}
        )

    candidates.sort(key=lambda x: (x["net"], x["match"]), reverse=True)
    top = candidates[:10]

    header = f"🕗 TIMELAB Morning Scan — TOP {len(top)} (CashConverters ES)\n\n"

    if not top:
        telegram_send(header + "No se encontraron oportunidades que cumplan filtros (marca reputada + match + net/ROI).")
    else:
        lines = [header]
        for i, it in enumerate(top, 1):
            lines.append(f"{i}) [cc] {it['title']}")
            lines.append(f"   💶 Compra: {it['buy']:.2f}€ | 🎯 Cierre est.: {it['close']:.2f}€")
            lines.append(
                f"   ✅ Neto est.: {it['net']:.2f}€ | ROI: {it['roi']*100:.1f}% | "
                f"Match: {it['match']} | Cond: {it['cond']} | Disp: {it['disp']}"
            )
            lines.append(f"   🧩 Target: {it['target']}")
            lines.append(f"   📍 {it['store']}")
            lines.append(f"   🔗 {it['url']}\n")
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
            "thresholds: "
            f"match>={CC_MIN_MATCH_SCORE} | "
            f"net>={CC_MIN_NET_EUR} OR roi>={CC_MIN_NET_ROI} | "
            f"haircut:{CLOSE_HAIRCUT}"
        )
        dbg.append(f"stop: max_items:{CC_MAX_ITEMS} | good_brands_target:{CC_GOOD_BRANDS_TARGET}")
        dbg.append(f"targets: items:{len(targets)} | valid:{len(targets)}")
        dbg.append(f"verify_mode:{1 if CC_VERIFY_MODE else 0}")

        if CC_VERIFY_MODE and debug_rejected:
            dbg.append("\nTop candidates (even if rejected):")
            debug_rejected.sort(key=lambda x: x.get("match", 0), reverse=True)
            for it in debug_rejected[:5]:
                extra = []
                if "net" in it and "roi" in it:
                    extra.append(f"net:{it['net']:.2f}")
                    extra.append(f"roi:{it['roi']*100:.1f}%")
                if "reason" in it:
                    extra.append(f"reason:{it['reason']}")
                extra_s = (" | " + " ".join(extra)) if extra else ""
                dbg.append(
                    f"- match:{it.get('match')} target:{it.get('target')} price:{it.get('price'):.2f} "
                    f"{it.get('title')}{extra_s}"
                )
                dbg.append(f"  url:{it.get('url')}")

        telegram_send("\n".join(dbg).strip())


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        telegram_send("❌ TIMELAB CashConverters scanner crashed\n" f"{type(e).__name__}: {e}")
        raise