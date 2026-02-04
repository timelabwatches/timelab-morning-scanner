import os
import re
import json
import time
import math
import random
from typing import Dict, Any, List, Optional, Tuple

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

CC_MIN_MATCH_SCORE = int(os.getenv("CC_MIN_MATCH_SCORE", "65"))
CC_MIN_NET_EUR = float(os.getenv("CC_MIN_NET_EUR", "20"))
CC_MIN_NET_ROI = float(os.getenv("CC_MIN_NET_ROI", "0.08"))

CLOSE_HAIRCUT = float(os.getenv("CLOSE_HAIRCUT", "0.90"))

CC_DEBUG = os.getenv("CC_DEBUG", "0").strip() == "1"
CC_GOOD_BRANDS_TARGET = int(os.getenv("CC_GOOD_BRANDS_TARGET", "60"))

# “verify mode”: baja umbrales para comprobar que hay matches
CC_VERIFY_MODE = os.getenv("CC_VERIFY_MODE", "0").strip() == "1"
if CC_VERIFY_MODE:
    CC_MIN_MATCH_SCORE = min(CC_MIN_MATCH_SCORE, 55)

# Catawiki fee assumptions (conservative)
CATWIKI_COMMISSION_RATE = 0.125  # 12.5%
CATWIKI_COMMISSION_MIN = 3.0     # €
CATWIKI_VAT_ON_COMMISSION = 0.21 # Spain VAT on fee (approx)

# CashConverters shipping: many items show envío, but cost isn’t always clear.
# We keep it conservative: assume 0 unless parsed; allow override
CC_DEFAULT_SHIP_EUR = float(os.getenv("CC_DEFAULT_SHIP_EUR", "0.0"))

# Brand policy
BANNED_BRANDS = set(map(str.lower, [
    # fashion / generic / low acceptance in CW for our arbitrage
    "lotus", "festina", "viceroy", "diesel", "armani", "emporio armani",
    "michael kors", "guess", "skagen", "fossil", "police", "tommy hilfiger",
    "calvin klein", "dkny", "hugo boss", "boss", "lacoste", "swatch",
    "welder", "smartwatch", "samsung", "xiaomi", "huawei", "amazfit", "fitbit",
]))

# If you want to allow Casio/Citizen sometimes, do NOT ban them.
# You can still gate them via target_list (only match if target exists).
# Here we do NOT ban casio/citizen by default.
REPUTABLE_FALLBACK = set(map(str.lower, [
    # We treat these as “reputable” globally (CW generally accepts)
    "omega", "longines", "tissot", "tag heuer", "tag", "heuer", "hamilton",
    "certina", "seiko", "oris", "zenith", "junghans", "baume", "mercier",
    "frederique constant", "raymond weil", "sinn"
]))


# =========================
# UTILS
# =========================
def tg_send(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("WARN: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing. Printing message:\n", text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True
    }
    requests.post(url, json=payload, timeout=15)


def canon(s: str) -> str:
    s = (s or "").strip().lower()
    # normalize accents basic
    s = s.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").replace("ñ", "n")
    # normalize separators
    s = re.sub(r"[\s\-/_,;:|]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # IMPORTANT: normalize common watch tokens in CC titles
    s = s.replace("powermatic80", "powermatic 80")
    s = s.replace("powermatic 80", "powermatic 80")
    s = s.replace(" p80 ", " powermatic 80 ")
    s = s.replace("deville", "de ville")
    s = s.replace("formula1", "formula 1")

    return s


def safe_float_price(txt: str) -> Optional[float]:
    if not txt:
        return None
    t = txt.strip()
    t = t.replace(".", "").replace("€", "").replace("eur", "").strip()
    t = t.replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)", t)
    if not m:
        return None
    try:
        return float(m.group(1))
    except:
        return None


def polite_sleep(base: float) -> None:
    # jitter to reduce pattern
    time.sleep(base + random.uniform(0.05, 0.20))


# =========================
# TARGETS
# =========================
def load_targets(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Accept either list or dict{targets:[...]}
    if isinstance(data, list):
        targets = data
        diag = f"type:list | items:{len(targets)}"
    elif isinstance(data, dict) and isinstance(data.get("targets"), list):
        targets = data["targets"]
        diag = f"type:dict | shape:{{targets:[...]}} | items:{len(targets)}"
    else:
        raise ValueError("target_list.json invalid shape. Expected list OR {targets:[...]}")

    valid = []
    for t in targets:
        if not isinstance(t, dict):
            continue
        if not t.get("id") or not t.get("brand") or not t.get("catawiki_estimate"):
            continue
        # catawiki_estimate must have p50
        ce = t.get("catawiki_estimate")
        if not isinstance(ce, dict) or "p50" not in ce:
            continue
        # normalize arrays
        t["model_keywords"] = t.get("model_keywords") or []
        t["must_include"] = t.get("must_include") or []
        t["must_exclude"] = t.get("must_exclude") or []
        valid.append(t)

    if not valid:
        keys_sample = ""
        if targets and isinstance(targets[0], dict):
            keys_sample = ",".join(list(targets[0].keys())[:20])
        raise ValueError(
            f"No valid targets loaded from target_list.json | diag:{diag} | items:{len(targets)} | valid:0 | keys_sample:{keys_sample}"
        )
    return valid


def detect_brand(title: str, targets: List[Dict[str, Any]]) -> Optional[str]:
    t = canon(title)
    # direct from targets brands first
    brands = sorted({canon(x.get("brand", "")) for x in targets if x.get("brand")}, key=len, reverse=True)
    for b in brands:
        if not b:
            continue
        if b in t:
            return b
    # fallback
    for b in REPUTABLE_FALLBACK:
        if b in t:
            return b
    return None


def is_banned_brand(title: str) -> bool:
    t = canon(title)
    for b in BANNED_BRANDS:
        if b in t:
            return True
    return False


def contains_any(text: str, words: List[str]) -> bool:
    t = canon(text)
    for w in words:
        if canon(w) and canon(w) in t:
            return True
    return False


def contains_all(text: str, words: List[str]) -> bool:
    t = canon(text)
    for w in words:
        ww = canon(w)
        if ww and ww not in t:
            return False
    return True


def compute_match_score(title: str, trg: Dict[str, Any]) -> int:
    """
    Conservative but not overly strict:
    - brand presence is mandatory (handled by best_target)
    - model_keywords add points
    - must_include adds points but missing does NOT nuke to 0 (CashConverters titles are messy)
    - must_exclude is a hard fail
    """
    t = canon(title)

    # hard excludes
    if contains_any(t, trg.get("must_exclude", [])):
        return 0

    score = 0

    # brand already required in best_target, but we still reward it
    b = canon(trg.get("brand", ""))
    if b and b in t:
        score += 25

    # model keywords (soft)
    hits = 0
    for kw in trg.get("model_keywords", []):
        k = canon(kw)
        if k and k in t:
            hits += 1

    # Scale hits to points
    # 0 hits -> 0, 1 hit -> +10, 2 hits -> +20, 3+ hits -> +30
    if hits == 1:
        score += 10
    elif hits == 2:
        score += 20
    elif hits >= 3:
        score += 30

    # must_include: reward presence, small penalty if missing
    includes = trg.get("must_include", [])
    inc_hits = 0
    for inc in includes:
        ii = canon(inc)
        if ii and ii in t:
            inc_hits += 1

    if includes:
        # if all present: +25
        if inc_hits == len(includes):
            score += 25
        else:
            # partial: + (10..20), small penalty for missing
            score += min(20, 10 + inc_hits * 5)
            score -= min(10, (len(includes) - inc_hits) * 3)

    # clamp
    score = max(0, min(100, score))
    return int(score)


def best_target(title: str, targets: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], int]:
    t = canon(title)

    best = None
    best_score = 0

    for trg in targets:
        b = canon(trg.get("brand", ""))
        if not b:
            continue

        # Brand must appear somewhere (strict)
        if b not in t:
            continue

        s = compute_match_score(t, trg)
        if s > best_score:
            best = trg
            best_score = s

    return best, best_score


# =========================
# CASHCONVERTERS SCRAPE
# =========================
BASE = "https://www.cashconverters.es"
# This path may vary; we use the same “relojes” route most CC pages share
START_URL = "https://www.cashconverters.es/es/es/comprar/relojes/"


def fetch(url: str, session: requests.Session) -> Optional[requests.Response]:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
        "Accept-Language": "es-ES,es;q=0.9,en;q=0.7",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
    }
    try:
        r = session.get(url, headers=headers, timeout=CC_TIMEOUT)
        return r
    except Exception as e:
        if CC_DEBUG:
            print("fetch error:", url, e)
        return None


def parse_listing_cards(html: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    out = []

    # CashConverters usually renders product tiles with <a href="/.../segunda-mano/CCxxx_....html">
    # We'll extract any <a> that looks like product detail.
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        if not href:
            continue
        if "/segunda-mano/" not in href:
            continue
        if not href.endswith(".html"):
            continue

        url = href if href.startswith("http") else BASE + href

        # Try to find title around card
        title = a.get_text(" ", strip=True)
        title = title if title else ""

        # price nearby
        price = None
        # check parent containers
        parent = a.parent
        for _ in range(0, 4):
            if not parent:
                break
            txt = parent.get_text(" ", strip=True)
            if "€" in txt:
                # take the first price-like
                m = re.search(r"(\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?)\s*€", txt)
                if m:
                    price = safe_float_price(m.group(1) + "€")
                    break
            parent = parent.parent

        # Avoid duplicates; store raw and refine later by fetching detail
        out.append({
            "url": url,
            "title": title,
            "price": price
        })

    # De-dup by URL preserving order
    seen = set()
    uniq = []
    for x in out:
        if x["url"] in seen:
            continue
        seen.add(x["url"])
        uniq.append(x)
    return uniq


def parse_detail(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")

    # Title
    h1 = soup.select_one("h1")
    title = h1.get_text(" ", strip=True) if h1 else ""

    # Price: try common selectors, then regex fallback
    price = None
    for sel in ["[itemprop='price']", ".product-price", ".price", ".pdp-price"]:
        el = soup.select_one(sel)
        if el:
            price = safe_float_price(el.get_text(" ", strip=True))
            if price is not None:
                break
    if price is None:
        txt = soup.get_text(" ", strip=True)
        m = re.search(r"(\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?)\s*€", txt)
        if m:
            price = safe_float_price(m.group(1) + "€")

    # Condition / availability (best-effort)
    txt = soup.get_text(" ", strip=True).lower()
    cond = None
    for c in ["perfecto", "muy bueno", "bueno", "usado", "aceptable", "nuevo"]:
        if c in txt:
            cond = c
            break

    disp = []
    if "envío" in txt or "envio" in txt:
        disp.append("envío")
    if "tienda" in txt:
        disp.append("tienda")
    availability = " + ".join(sorted(set(disp))) if disp else ""

    # Store/ID: CCxxx_yyyy
    m_id = re.search(r"(CC\d{3}_[A-Z0-9]+_\d+)", html)
    store_id = m_id.group(1) if m_id else ""

    # Shipping cost: hard to parse; keep default
    ship = CC_DEFAULT_SHIP_EUR

    return {
        "detail_title": title,
        "detail_price": price,
        "cond": cond or "",
        "availability": availability or "",
        "store": store_id or "",
        "ship": ship
    }


# =========================
# ECONOMICS
# =========================
def close_estimate_eur(target: Dict[str, Any], cond: str) -> float:
    # Use p50 as base
    p50 = float(target["catawiki_estimate"]["p50"])

    # Very light condition adjust (avoid overfitting)
    c = canon(cond)
    adj = 1.00
    if "perfecto" in c or "excelente" in c or "como nuevo" in c:
        adj = 1.05
    elif "aceptable" in c:
        adj = 0.92
    elif "usado" in c:
        adj = 0.97
    elif "bueno" in c:
        adj = 1.00

    est = p50 * adj * CLOSE_HAIRCUT
    return round(est, 2)


def net_profit_eur(buy: float, ship: float, close_est: float) -> Tuple[float, float]:
    """
    Conservative:
    - You pay buy + ship
    - On sale, Catawiki fee = max(12.5%, 3€) + VAT(21%) on that fee
    """
    fee = max(close_est * CATWIKI_COMMISSION_RATE, CATWIKI_COMMISSION_MIN)
    fee_total = fee * (1.0 + CATWIKI_VAT_ON_COMMISSION)
    net = close_est - fee_total - buy - ship
    roi = net / (buy + ship) if (buy + ship) > 0 else 0.0
    return round(net, 2), roi


# =========================
# OUTPUT
# =========================
def format_top(top: List[Dict[str, Any]]) -> str:
    header = f"🕗 TIMELAB Morning Scan — TOP {len(top)} (CashConverters ES)\n\n"
    if not top:
        return header + "No se encontraron oportunidades que cumplan filtros (marca reputada + match + net/ROI)."

    lines = [header]
    for i, x in enumerate(top, 1):
        lines.append(
            f"{i}) [cc] {x['title']}\n"
            f"   💶 Compra: {x['buy']:.2f}€ | 🎯 Cierre est.: {x['close_est']:.2f}€\n"
            f"   ✅ Neto est.: {x['net']:.2f}€ | ROI: {x['roi']*100:.1f}% | Match: {x['match']} | Cond: {x['cond'] or '-'} | Disp: {x['availability'] or '-'}\n"
            f"   🧩 Target: {x['target_id']}\n"
            f"   📍 {x['store'] or '-'}\n"
            f"   🔗 {x['url']}\n"
        )
    return "\n".join(lines).strip()


def debug_msg(diag: Dict[str, Any]) -> str:
    # compact and readable
    lines = []
    lines.append(f"🧪 TIMELAB CC Debug — scanned:{diag.get('scanned', 0)} | page_bad:{diag.get('page_bad', 0)}")
    b = diag.get("brands", {})
    lines.append(f"brands: reputable:{b.get('reputable', 0)} | banned:{b.get('banned', 0)} | not_reputable:{b.get('not_reputable', 0)} | no_brand:{b.get('no_brand', 0)}")
    p = diag.get("passed", {})
    lines.append(f"passed: match_ok:{p.get('match_ok', 0)} | net_ok:{p.get('net_ok', 0)}")
    lines.append(f"thresholds: match>={CC_MIN_MATCH_SCORE} | net>={CC_MIN_NET_EUR} OR roi>={CC_MIN_NET_ROI} | haircut:{CLOSE_HAIRCUT}")
    s = diag.get("stop", {})
    lines.append(f"stop: max_items:{s.get('max_items', CC_MAX_ITEMS)} | good_brands_target:{s.get('good_brands_target', CC_GOOD_BRANDS_TARGET)}")
    t = diag.get("targets", {})
    lines.append(f"targets: items:{t.get('items', 0)} | valid:{t.get('valid', 0)}")
    lines.append(f"verify_mode:{1 if CC_VERIFY_MODE else 0}")
    return "\n".join(lines)


# =========================
# MAIN
# =========================
def run() -> None:
    # start message (optional)
    if CC_DEBUG:
        tg_send("🧪 TIMELAB CashConverters scanner: started (debug)")

    targets = load_targets("target_list.json")

    diag = {
        "scanned": 0,
        "page_bad": 0,
        "brands": {"reputable": 0, "banned": 0, "not_reputable": 0, "no_brand": 0},
        "passed": {"match_ok": 0, "net_ok": 0},
        "stop": {"max_items": CC_MAX_ITEMS, "good_brands_target": CC_GOOD_BRANDS_TARGET},
        "targets": {"items": len(targets), "valid": len(targets)}
    }

    session = requests.Session()

    # We crawl starting list; CC uses pagination but can vary.
    # We will follow “next” links if present; otherwise we keep collecting from current page only.
    to_visit = [START_URL]
    visited = set()
    candidates: List[Dict[str, Any]] = []
    good_brand_seen = 0

    while to_visit and diag["scanned"] < CC_MAX_ITEMS:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        r = fetch(url, session)
        if not r or r.status_code != 200 or not r.text or len(r.text) < 2000:
            diag["page_bad"] += 1
            continue

        cards = parse_listing_cards(r.text)
        polite_sleep(CC_THROTTLE_S)

        # add next links if any
        soup = BeautifulSoup(r.text, "lxml")
        next_a = soup.find("a", attrs={"rel": "next"})
        if next_a and next_a.get("href"):
            nxt = next_a.get("href")
            nxt_url = nxt if nxt.startswith("http") else BASE + nxt
            if nxt_url not in visited:
                to_visit.append(nxt_url)

        for card in cards:
            if diag["scanned"] >= CC_MAX_ITEMS:
                break

            # fetch detail to enrich
            dr = fetch(card["url"], session)
            polite_sleep(CC_THROTTLE_S)
            if not dr or dr.status_code != 200 or not dr.text or len(dr.text) < 2000:
                diag["page_bad"] += 1
                continue

            detail = parse_detail(dr.text)
            title = detail["detail_title"] or card["title"] or ""
            buy = detail["detail_price"] if detail["detail_price"] is not None else card["price"]

            if not title or buy is None:
                diag["scanned"] += 1
                continue

            # brand filter (global)
            if is_banned_brand(title):
                diag["brands"]["banned"] += 1
                diag["scanned"] += 1
                continue

            brand = detect_brand(title, targets)
            if not brand:
                diag["brands"]["no_brand"] += 1
                diag["scanned"] += 1
                continue

            # treat it as reputable if brand appears in any target brand set
            diag["brands"]["reputable"] += 1
            good_brand_seen += 1

            # match to best target
            trg, match = best_target(title, targets)
            if not trg:
                diag["brands"]["not_reputable"] += 1
                diag["scanned"] += 1
                continue

            # threshold match
            if match >= CC_MIN_MATCH_SCORE:
                diag["passed"]["match_ok"] += 1

            close_est = close_estimate_eur(trg, detail.get("cond", ""))
            ship = float(detail.get("ship", CC_DEFAULT_SHIP_EUR) or 0.0)
            net, roi = net_profit_eur(float(buy), ship, close_est)

            if (match >= CC_MIN_MATCH_SCORE) and (net >= CC_MIN_NET_EUR or roi >= CC_MIN_NET_ROI):
                diag["passed"]["net_ok"] += 1
                candidates.append({
                    "title": title,
                    "url": card["url"],
                    "buy": float(buy),
                    "ship": ship,
                    "cond": detail.get("cond", ""),
                    "availability": detail.get("availability", ""),
                    "store": detail.get("store", ""),
                    "target_id": trg["id"],
                    "match": match,
                    "close_est": close_est,
                    "net": net,
                    "roi": roi
                })

            diag["scanned"] += 1

            # stop early once we have enough reputable items sampled (optional)
            if good_brand_seen >= CC_GOOD_BRANDS_TARGET and diag["scanned"] >= min(CC_MAX_ITEMS, 200):
                # if we already saw enough reputable brands, no need to crawl too much
                break

    # Rank TOP 10 by net, then match
    candidates.sort(key=lambda x: (x["net"], x["match"], x["roi"]), reverse=True)
    top = candidates[:10]

    tg_send(format_top(top))

    if CC_DEBUG:
        tg_send(debug_msg(diag))


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        # separated crash message
        msg = "❌ TIMELAB CashConverters scanner crashed\n" + repr(e)
        try:
            tg_send(msg)
        except Exception:
            pass
        raise