#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import math
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlencode

import requests
from bs4 import BeautifulSoup


# =========================
# ENV / CONFIG
# =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

CC_MAX_ITEMS = int(os.getenv("CC_MAX_ITEMS", "100"))
CC_TIMEOUT = float(os.getenv("CC_TIMEOUT", "20"))
CC_THROTTLE_S = float(os.getenv("CC_THROTTLE_S", "0.8"))
CC_DEBUG = os.getenv("CC_DEBUG", "0").strip() in ("1", "true", "TRUE", "yes", "YES")

# thresholds
CC_MIN_MATCH_SCORE = int(os.getenv("CC_MIN_MATCH_SCORE", "55"))
CC_MIN_NET_EUR = float(os.getenv("CC_MIN_NET_EUR", "20"))
CC_MIN_NET_ROI = float(os.getenv("CC_MIN_NET_ROI", "0.08"))
CLOSE_HAIRCUT = float(os.getenv("CLOSE_HAIRCUT", "0.90"))

# Verification mode: relax net/ROI gating (still compute + show)
CC_VERIFY_MODE = os.getenv("CC_VERIFY_MODE", "0").strip() in ("1", "true", "TRUE", "yes", "YES")

# How many reputable-brand items to try to collect before stopping early (speed)
CC_GOOD_BRANDS_TARGET = int(os.getenv("CC_GOOD_BRANDS_TARGET", "60"))

# Catawiki seller commission model (approx; you can align later to your TIMELAB exact model)
CATWIKI_SELLER_COMMISSION = float(os.getenv("CATWIKI_SELLER_COMMISSION", "0.125"))  # 12.5%
CATWIKI_SELLER_COMMISSION_VAT = float(os.getenv("CATWIKI_SELLER_COMMISSION_VAT", "0.21"))  # IVA 21%
CATWIKI_EFFECTIVE_FEE = CATWIKI_SELLER_COMMISSION * (1.0 + CATWIKI_SELLER_COMMISSION_VAT)

USER_AGENT = os.getenv(
    "CC_USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36"
)

BASE = "https://www.cashconverters.es"


# =========================
# Brand policy
# =========================
# Whitelist: marcas “relojería” que Catawiki suele aceptar sin fricción (puedes ampliar)
REPUTABLE_BRANDS = {
    "omega", "rolex", "tudor", "breitling", "tag heuer", "heuer",
    "longines", "tissot", "hamilton", "certina", "seiko", "citizen",
    "oris", "zenith", "jaeger lecoultre", "jlc", "iwc", "cartier",
    "panerai", "bell & ross", "bell ross", "baume & mercier", "baume mercier",
    "rado", "eterna", "movado", "junghans", "sinn", "nomos",
    "mido", "girard perregaux", "gp", "vacheron constantin",
    "audemars piguet", "patek philippe", "u-boat", "u boat",
    "fortis", "frederique constant", "fc", "raymond weil",
    "maurice lacroix", "edox", "alpina", "bulova", "doxa",
    "glashutte", "glashütte", "piaget", "chopard"
}

# Banlist: marcas típicas moda/low-tier / smartwatch (para no contaminar)
BANNED_BRANDS = {
    "lotus", "festina", "calvin klein", "armani", "emporio armani",
    "guess", "diesel", "police", "daniel wellington", "dw",
    "michael kors", "mk", "fossil", "skagen", "swatch",
    "casio",  # si quieres permitir Casio vintage, sácalo de aquí
    "samsung", "apple", "huawei", "xiaomi", "amazfit", "garmin",
    "invicta", "viceroy", "toywatch", "tommy hilfiger", "lacoste"
}

# Tokens “malos” (multi-idioma) tipo “para piezas / no funciona”
BAD_TOKENS = [
    r"\bpara\s+piezas\b", r"\bno\s+funciona\b", r"\baveriado\b", r"\bdefectuoso\b",
    r"\bfor\s+parts\b", r"\bnot\s+working\b", r"\bbroken\b",
    r"\bparts\s+only\b", r"\bspares\b",
    r"\bincompleto\b", r"\bsin\s+correa\b", r"\bsin\s+tapa\b",
]

GOOD_TOKENS = [
    r"\brevisado\b", r"\bservicio\b", r"\brecien\s+revisado\b",
    r"\bfunciona\b", r"\bperfecto\b", r"\bcomo\s+nuevo\b", r"\bnos\b",
    r"\bfull\s+set\b", r"\bcaja\s+y\s+papeles\b", r"\bcon\s+papeles\b"
]


# =========================
# Data structures
# =========================
@dataclass
class CCItem:
    title: str
    url: str
    price_eur: float
    condition: str = ""
    availability: str = ""
    store: str = ""
    brand: str = ""


# =========================
# Helpers
# =========================
def canon(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_brand(title: str) -> str:
    t = canon(title)
    # check explicit multiword brands first
    for b in sorted(REPUTABLE_BRANDS | BANNED_BRANDS, key=lambda x: -len(x)):
        if canon(b) in t:
            return canon(b)
    # heuristic: first token might be a brand
    first = t.split(" ")[0] if t else ""
    return first


def is_bad_listing(text: str) -> bool:
    txt = canon(text)
    for pat in BAD_TOKENS:
        if re.search(pat, txt):
            return True
    return False


def condition_boost(text: str) -> float:
    """Small adjustment factor based on condition signals."""
    txt = canon(text)
    # default neutral
    boost = 1.0
    if re.search(r"\bperfecto\b|\bcomo nuevo\b|\bnos\b|\bfull set\b", txt):
        boost *= 1.05
    if re.search(r"\busado\b|\bmarcas\b|\brasgu", txt):
        boost *= 0.95
    if re.search(r"\bbueno\b", txt):
        boost *= 1.00
    return boost


def safe_float_eur(s: str) -> Optional[float]:
    if not s:
        return None
    s = s.replace("\xa0", " ")
    m = re.search(r"(\d{1,3}(?:[.\s]\d{3})*|\d+)(?:,(\d{2}))?\s*€", s)
    if not m:
        return None
    a = m.group(1).replace(".", "").replace(" ", "")
    b = m.group(2) or "00"
    try:
        return float(f"{a}.{b}")
    except Exception:
        return None


def telegram_send(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        # fail silently in local runs
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True
    }
    requests.post(url, json=payload, timeout=20)


# =========================
# Targets loading + matching
# =========================
def load_targets(path: str) -> List[Dict[str, Any]]:
    """
    Accepts either:
      - a dict: {"targets":[...]}
      - a raw list: [...]
    Each target must have at least:
      - id
      - brand
      - catawiki_estimate: {"p50":..., "p75":...}
      - model_keywords (list) (can be empty)
      - risk (optional)
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "targets" in raw and isinstance(raw["targets"], list):
        targets = raw["targets"]
    elif isinstance(raw, list):
        targets = raw
    else:
        raise ValueError("Unsupported target_list.json shape")

    valid = []
    for t in targets:
        if not isinstance(t, dict):
            continue
        if "id" not in t or "brand" not in t:
            continue
        if "catawiki_estimate" not in t or not isinstance(t["catawiki_estimate"], dict):
            continue
        ce = t["catawiki_estimate"]
        if "p50" not in ce:
            continue
        if "model_keywords" in t and not isinstance(t["model_keywords"], list):
            continue
        if "model_keywords" not in t:
            t["model_keywords"] = []
        valid.append(t)

    if not valid:
        raise ValueError("No valid targets loaded from target_list.json")

    return valid


def compute_match_score(title: str, target: Dict[str, Any]) -> int:
    """
    Simple robust score:
      - brand presence (big weight)
      - model keywords presence (sum)
      - must_include / must_exclude if present
    """
    t = canon(title)
    brand = canon(target.get("brand", ""))
    score = 0

    if brand and brand in t:
        score += 60
    else:
        # sometimes target brand is "TAG HEUER" but title "tagheuer" etc
        if brand and brand.replace(" ", "") in t.replace(" ", ""):
            score += 55

    # must_include / exclude
    must_include = target.get("must_include") or []
    must_exclude = target.get("must_exclude") or []
    if isinstance(must_include, str):
        must_include = [must_include]
    if isinstance(must_exclude, str):
        must_exclude = [must_exclude]

    for w in must_include:
        if canon(w) and canon(w) in t:
            score += 10
        else:
            # missing a required token -> heavy penalty
            score -= 15

    for w in must_exclude:
        if canon(w) and canon(w) in t:
            score -= 40

    # model keywords
    kws = target.get("model_keywords", []) or []
    hit = 0
    for kw in kws:
        kwc = canon(kw)
        if not kwc:
            continue
        if kwc in t:
            hit += 1
    score += min(30, hit * 10)

    # clamp
    score = max(0, min(100, score))
    return int(score)


def best_target(title: str, targets: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], int]:
    best = None
    best_score = -1
    for trg in targets:
        s = compute_match_score(title, trg)
        if s > best_score:
            best_score = s
            best = trg
    return best, best_score


def estimate_close_eur(title: str, condition_text: str, target: Dict[str, Any]) -> float:
    ce = target["catawiki_estimate"]
    base = float(ce.get("p50", 0))
    # conservative haircut + small condition adjustment
    est = base * CLOSE_HAIRCUT
    est *= condition_boost(f"{title} {condition_text}")
    return float(est)


def estimate_net_profit(buy_eur: float, close_est_eur: float) -> Tuple[float, float]:
    """
    Approx:
      seller fee: CATWIKI_EFFECTIVE_FEE of hammer
      net = close_est - fee - buy
      roi = net / buy
    """
    fee = close_est_eur * CATWIKI_EFFECTIVE_FEE
    net = close_est_eur - fee - buy_eur
    roi = (net / buy_eur) if buy_eur > 0 else 0.0
    return float(net), float(roi)


# =========================
# CashConverters scraping
# =========================
def http_get(session: requests.Session, url: str) -> Tuple[int, str]:
    r = session.get(url, timeout=CC_TIMEOUT)
    return r.status_code, r.text


def parse_listing_cards(html: str) -> List[CCItem]:
    """
    Tries to extract product cards from listing/search page.
    Works best when page contains product tiles linking to /segunda-mano/<ID>.html
    """
    soup = BeautifulSoup(html, "lxml")
    items: List[CCItem] = []

    # common pattern: links to /segunda-mano/CCxxx_Exxxxx_0.html
    for a in soup.select("a[href*='/segunda-mano/']"):
        href = a.get("href", "")
        if not href:
            continue
        if "/segunda-mano/" not in href:
            continue
        if not href.endswith(".html"):
            continue

        url = urljoin(BASE, href)
        # try to find title nearby
        title = canon(a.get_text(" ", strip=True)) or ""
        # if title is too short, maybe inside child nodes
        if len(title) < 6:
            title = canon(a.get("title", "")) or title

        # price: search in parent container
        price = None
        parent = a
        for _ in range(4):
            if parent is None:
                break
            txt = parent.get_text(" ", strip=True)
            price = safe_float_eur(txt)
            if price is not None:
                break
            parent = parent.parent

        if price is None:
            continue

        # minimal
        items.append(CCItem(title=title or "(sin título)", url=url, price_eur=price))

    # Dedup by URL keeping first
    dedup: Dict[str, CCItem] = {}
    for it in items:
        if it.url not in dedup:
            dedup[it.url] = it
    return list(dedup.values())


def fetch_detail(session: requests.Session, item: CCItem) -> CCItem:
    """
    Enrich with condition/availability/store by reading product page title/meta/text.
    """
    status, html = http_get(session, item.url)
    if status != 200 or not html:
        return item

    soup = BeautifulSoup(html, "lxml")

    # Title: h1 is usually present
    h1 = soup.select_one("h1")
    if h1:
        t = canon(h1.get_text(" ", strip=True))
        if t and len(t) > 6:
            item.title = t

    # Extract some signals
    page_text = soup.get_text(" ", strip=True)
    page_text_c = canon(page_text)

    # Condition: look for common words used by CC
    # (bueno, perfecto, usado)
    cond = ""
    for c in ["perfecto", "como nuevo", "bueno", "usado", "aceptable"]:
        if re.search(rf"\b{re.escape(c)}\b", page_text_c):
            cond = c
            break
    item.condition = cond

    # Availability: try to detect envío / tienda
    avail = ""
    if "envío" in page_text_c or "envio" in page_text_c:
        avail = "envío"
    if "recogida en tienda" in page_text_c:
        avail = (avail + " + tienda").strip(" +")
    item.availability = avail

    # Store / code: CC037_E... appears in title tag often
    # Try meta og:url or title tag
    title_tag = soup.title.get_text(" ", strip=True) if soup.title else ""
    m = re.search(r"\b(CC\d{3}_[A-Z]\d+_\d+)\b", title_tag)
    if m:
        item.store = m.group(1)

    # Brand from title after enrichment
    item.brand = extract_brand(item.title)

    return item


def build_listing_urls() -> List[str]:
    """
    CashConverters site changes a lot; we use robust entrypoints:
    - search by query 'reloj pulsera' + category relojes
    We'll try multiple URL patterns; first that returns enough items will be used.
    """
    urls = []

    # 1) Search endpoint (generic)
    q = "reloj pulsera"
    urls.append(f"{BASE}/es/es/segunda-mano/?{urlencode({'q': q})}")

    # 2) Category endpoint (seen in your earlier browsing)
    urls.append(f"{BASE}/es/es/comprar/relojes/")

    return urls


def scan_cashconverters() -> Tuple[List[CCItem], Dict[str, Any]]:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "es-ES,es;q=0.9,en;q=0.7"})

    debug: Dict[str, Any] = {
        "scanned": 0,
        "page_bad": 0,
        "brands": {"reputable": 0, "banned": 0, "no_brand": 0, "not_reputable": 0},
    }

    seen: set = set()
    collected: List[CCItem] = []

    entrypoints = build_listing_urls()

    chosen_entry = None
    for u in entrypoints:
        try:
            st, html = http_get(session, u)
            cards = parse_listing_cards(html) if (st == 200 and html) else []
            if len(cards) >= 10:
                chosen_entry = u
                break
        except Exception:
            continue

    if not chosen_entry:
        return [], {**debug, "error": "No entrypoint produced enough cards"}

    # simple pagination strategy:
    # try adding ?start=0,24,48 ... (many CC pages use start/offset; if ignored, we still dedup)
    offsets = list(range(0, 2000, 24))

    for off in offsets:
        if len(collected) >= CC_MAX_ITEMS:
            break
        if debug["brands"]["reputable"] >= CC_GOOD_BRANDS_TARGET and len(collected) >= min(CC_MAX_ITEMS, CC_GOOD_BRANDS_TARGET):
            break

        url = chosen_entry
        if "?" in url:
            url = f"{url}&start={off}"
        else:
            url = f"{url}?start={off}"

        try:
            st, html = http_get(session, url)
            if st != 200 or not html:
                debug["page_bad"] += 1
                continue

            cards = parse_listing_cards(html)
            if not cards:
                # probably end
                break

            for it in cards:
                if len(collected) >= CC_MAX_ITEMS:
                    break
                if it.url in seen:
                    continue

                seen.add(it.url)
                debug["scanned"] += 1

                # throttle
                time.sleep(CC_THROTTLE_S)

                # enrich detail (improves title/match)
                it = fetch_detail(session, it)

                # quality filter: reject obvious broken/parts
                if is_bad_listing(f"{it.title} {it.condition}"):
                    continue

                br = canon(it.brand) if it.brand else extract_brand(it.title)
                it.brand = br

                if not br:
                    debug["brands"]["no_brand"] += 1
                    continue

                if br in BANNED_BRANDS:
                    debug["brands"]["banned"] += 1
                    continue

                if br not in REPUTABLE_BRANDS:
                    debug["brands"]["not_reputable"] += 1
                    continue

                debug["brands"]["reputable"] += 1
                collected.append(it)

        except Exception:
            debug["page_bad"] += 1
            continue

    return collected, debug


# =========================
# Telegram formatting
# =========================
def fmt_money(x: float) -> str:
    return f"{x:,.2f}€".replace(",", "X").replace(".", ",").replace("X", ".")


def build_message(top: List[Dict[str, Any]], debug: Dict[str, Any]) -> str:
    header = "🕗 TIMELAB Morning Scan — TOP {} (CashConverters ES)".format(len(top))
    if CC_VERIFY_MODE:
        header += " — VERIFY"

    lines = [header, ""]

    if not top:
        lines.append("No se encontraron oportunidades que cumplan filtros (marca reputada + match + net/ROI).")
        if CC_VERIFY_MODE:
            lines.append("Modo VERIFY activo: debería listar mejores matches aunque no cumplan net/ROI. Si sigue TOP 0, entonces no hay match suficiente con los targets.")
        lines.append("")
    else:
        for i, row in enumerate(top, 1):
            it: CCItem = row["item"]
            target = row["target_id"]
            match = row["match"]
            close_est = row["close_est"]
            net = row["net"]
            roi = row["roi"]

            cond = it.condition or "-"
            disp = it.availability or "-"
            store = it.store or "-"

            lines.append(f"{i}) [cc] {it.title}")
            lines.append(
                f"   💶 Compra: {fmt_money(it.price_eur)} | 🎯 Cierre est.: {fmt_money(close_est)}"
            )
            lines.append(
                f"   ✅ Neto est.: {fmt_money(net)} | ROI: {roi*100:.1f}% | Match: {match} | Cond: {cond} | Disp: {disp}"
            )
            lines.append(f"   🧩 Target: {target}")
            lines.append(f"   📍 {store}")
            lines.append(f"   🔗 {it.url}")
            lines.append("")

    if CC_DEBUG:
        b = debug.get("brands", {})
        lines.append("🧪 TIMELAB CC Debug — "
                     f"scanned:{debug.get('scanned',0)} | page_bad:{debug.get('page_bad',0)}")
        lines.append("brands: "
                     f"reputable:{b.get('reputable',0)} | banned:{b.get('banned',0)} | "
                     f"not_reputable:{b.get('not_reputable',0)} | no_brand:{b.get('no_brand',0)}")
        lines.append(f"passed: match_ok:{debug.get('match_ok',0)} | net_ok:{debug.get('net_ok',0)}")
        lines.append(f"thresholds: match>={CC_MIN_MATCH_SCORE} | net>={CC_MIN_NET_EUR} OR roi>={CC_MIN_NET_ROI} | haircut:{CLOSE_HAIRCUT}")
        lines.append(f"stop: max_items:{CC_MAX_ITEMS} | good_brands_target:{CC_GOOD_BRANDS_TARGET}")
        lines.append(f"targets: items:{debug.get('targets_items','?')} | valid:{debug.get('targets_valid','?')}")
        lines.append(f"verify_mode:{int(CC_VERIFY_MODE)}")
    return "\n".join(lines).strip()


# =========================
# Main run
# =========================
def run() -> None:
    # start ping only in debug (avoid noise)
    if CC_DEBUG:
        telegram_send("🧪 TIMELAB CashConverters scanner: started")

    # load targets
    targets = load_targets("target_list.json")
    dbg_targets_shape = "list" if isinstance(targets, list) else type(targets).__name__
    debug = {"targets_items": len(targets), "targets_valid": len(targets), "match_ok": 0, "net_ok": 0}

    items, debug_scan = scan_cashconverters()
    debug.update(debug_scan)

    candidates: List[Dict[str, Any]] = []

    for it in items:
        # best target + score
        trg, match = best_target(it.title, targets)
        if not trg:
            continue

        if match >= CC_MIN_MATCH_SCORE:
            debug["match_ok"] += 1

        # compute close + net anyway
        close_est = estimate_close_eur(it.title, it.condition, trg)
        net, roi = estimate_net_profit(it.price_eur, close_est)

        passes_match = match >= CC_MIN_MATCH_SCORE
        passes_value = (net >= CC_MIN_NET_EUR) or (roi >= CC_MIN_NET_ROI)

        # VERIFY: allow passes_value to be ignored (but keep passes_match)
        if CC_VERIFY_MODE:
            accept = passes_match
        else:
            accept = passes_match and passes_value

        if passes_value:
            debug["net_ok"] += 1

        if accept:
            candidates.append({
                "item": it,
                "target_id": trg.get("id", "UNKNOWN"),
                "match": match,
                "close_est": close_est,
                "net": net,
                "roi": roi
            })

    # Rank: net desc then match desc
    candidates.sort(key=lambda r: (r["net"], r["match"]), reverse=True)

    top = candidates[:10]

    # If VERIFY and still empty, show top matches anyway (diagnosis)
    if CC_VERIFY_MODE and not top:
        # show best matches among reputable items
        diag: List[Dict[str, Any]] = []
        for it in items:
            trg, match = best_target(it.title, targets)
            if not trg:
                continue
            close_est = estimate_close_eur(it.title, it.condition, trg)
            net, roi = estimate_net_profit(it.price_eur, close_est)
            diag.append({
                "item": it,
                "target_id": trg.get("id", "UNKNOWN"),
                "match": match,
                "close_est": close_est,
                "net": net,
                "roi": roi
            })
        diag.sort(key=lambda r: (r["match"], r["net"]), reverse=True)
        top = diag[:10]

    msg = build_message(top, debug)
    telegram_send(msg)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        err = traceback.format_exc()
        telegram_send("❌ TIMELAB CashConverters scanner crashed\n" + err[:3500])
        raise