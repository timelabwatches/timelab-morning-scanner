import os
import re
import time
import json
import traceback
from typing import Dict, List, Tuple, Optional
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE = "https://www.cashconverters.es"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

CC_TIMEOUT = float(os.getenv("CC_TIMEOUT", "20"))
CC_THROTTLE_S = float(os.getenv("CC_THROTTLE_S", "0.8"))
CC_MAX_ITEMS = int(os.getenv("CC_MAX_ITEMS", "100"))

MIN_MATCH_SCORE = int(os.getenv("CC_MIN_MATCH_SCORE", "65"))
MIN_NET_EUR = float(os.getenv("CC_MIN_NET_EUR", "20"))
MIN_NET_ROI = float(os.getenv("CC_MIN_NET_ROI", "0.08"))
CLOSE_HAIRCUT = float(os.getenv("CLOSE_HAIRCUT", "0.90"))

DEFAULT_SHIPPING_EUR = float(os.getenv("CC_DEFAULT_SHIPPING_EUR", "0.0"))

DEBUG_CC = os.getenv("CC_DEBUG", "1").strip() == "1"   # default ON for now

PAGE_SIZE = int(os.getenv("CC_PAGE_SIZE", "24"))
SRULE = os.getenv("CC_SRULE", "new")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TIMELAB-WATCHES/1.0)",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
}

SEED_URLS = [
    f"{BASE}/es/es/comprar/relojes/reloj-pulsera/reloj-pulsera-caballero/",
    f"{BASE}/es/es/comprar/relojes/reloj-pulsera/reloj-pulsera-senora/",
    f"{BASE}/es/es/comprar/relojes/reloj-pulsera/reloj-pulsera-unisex/",
    f"{BASE}/es/es/comprar/relojes/reloj-pulsera-premium/reloj-pulsera-premium-caballero/",
    f"{BASE}/es/es/comprar/relojes/reloj-pulsera-premium/reloj-pulsera-premium-senora/",
    f"{BASE}/es/es/comprar/relojes/reloj-pulsera-premium/reloj-pulsera-premium-unisex/",
    f"{BASE}/es/es/comprar/relojes/reloj-alta-gama/reloj-alta-gama-caballero/",
    f"{BASE}/es/es/comprar/relojes/reloj-alta-gama/reloj-alta-gama-senora/",
    f"{BASE}/es/es/comprar/relojes/reloj-alta-gama/reloj-alta-gama-unisex/",
]

# ===== Brand policy =====
REPUTABLE_BRANDS = {
    "rolex", "tudor", "omega", "longines", "tag heuer", "heuer", "breitling",
    "zenith", "jaeger-lecoultre", "jlc", "iwc", "panerai", "cartier",
    "bvlgari", "bulgari", "baume & mercier", "baume mercier",
    "tissot", "hamilton", "certina", "oris", "rado",
    "seiko", "grand seiko", "citizen", "orient", "mido",
    "edox", "doxa", "eterna", "fortis", "sinn", "nomos",
    "maurice lacroix", "frederique constant", "frédérique constant",
    "vacheron constantin", "audemars piguet", "patek philippe",
    "girard-perregaux", "glashütte", "glashuette", "glashütte original",
    "hublot", "chopard", "montblanc", "u-boat", "ulysse nardin",
    "raymond weil", "alpina", "laco", "stowa",
}

BANNED_BRANDS = {
    "lotus", "festina", "diesel", "armani", "emporio armani", "michael kors",
    "guess", "dkny", "fossil", "police", "hugo boss", "boss", "swatch",
    "samsung", "xiaomi", "huawei", "apple", "garmin", "fitbit",
    "welder", "ice watch", "icewatch", "tommy hilfiger", "calvin klein",
}

BANNED_KEYWORDS = {
    "smartwatch", "reloj inteligente", "galaxy watch", "apple watch", "fitbit",
    "pulsera actividad", "activity", "fitness",
    "sin funcionar", "no funciona", "para piezas", "solo piezas",
    "incompleto", "averiado", "defectuoso", "rotura", "no arranca",
}

POSITIVE_KEYWORDS = {"revisado", "funciona", "buen estado", "perfecto", "recien revisado", "recién revisado"}

PRICE_RE = re.compile(r"(\d{1,3}(?:\.\d{3})*,\d{2})\s*€")
ID_RE = re.compile(r"/segunda-mano/([^/]+)\.html")

def euro_to_float(s: str) -> float:
    s = s.replace(".", "").replace(",", ".")
    return float(s)

def tg_send(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID (GitHub Secrets?)")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    r = requests.post(url, timeout=CC_TIMEOUT, json={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    })
    r.raise_for_status()

def fetch(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=CC_TIMEOUT)
    r.raise_for_status()
    return r.text

def build_page_url(seed: str, start: int) -> str:
    joiner = "&" if "?" in seed else "?"
    return f"{seed}{joiner}srule={SRULE}&start={start}&sz={PAGE_SIZE}"

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def has_banned_keywords(text: str) -> bool:
    t = normalize(text)
    return any(k in t for k in BANNED_KEYWORDS)

def brand_from_text(text: str) -> Optional[str]:
    t = normalize(text)
    for b in sorted(REPUTABLE_BRANDS, key=len, reverse=True):
        if b in t:
            return b
    for b in sorted(BANNED_BRANDS, key=len, reverse=True):
        if b in t:
            return f"__banned__:{b}"
    return None

def condition_boost(cond: str) -> float:
    c = normalize(cond)
    if "perfecto" in c or "muy bueno" in c:
        return 1.00
    if "bueno" in c:
        return 0.97
    if "usado" in c:
        return 0.93
    return 0.95

def compute_match_score_simple(title: str, target: Dict) -> int:
    text = normalize(title)
    score = 0
    t_brand = normalize(target.get("brand", ""))
    if t_brand and t_brand in text:
        score += 55
    kws = target.get("keywords", []) or []
    for kw in kws:
        kw_n = normalize(kw)
        if kw_n and kw_n in text:
            score += 10
    return int(max(0, min(100, score)))

def load_targets(path: str = "target_list.json") -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "targets" in data:
        data = data["targets"]
    if not isinstance(data, list):
        raise ValueError("target_list.json must be a list or {targets:[...]}")
    out = []
    for t in data:
        out.append({
            "id": t.get("id") or t.get("target") or t.get("name"),
            "brand": t.get("brand", ""),
            "base_close_eur": float(t.get("base_close_eur") or t.get("base_close") or 0),
            "fake_risk": (t.get("fake_risk") or "low").lower(),
            "keywords": t.get("keywords") or [],
        })
    return out

def pick_best_target(title: str, targets: List[Dict]) -> Tuple[Optional[Dict], int]:
    best = None
    best_score = -1
    for t in targets:
        score = compute_match_score_simple(title, t)
        if score > best_score:
            best_score = score
            best = t
    return best, best_score

def estimate_close(target: Dict, cond: str) -> float:
    base = float(target.get("base_close_eur") or 0.0)
    if base <= 0:
        return 0.0
    return base * CLOSE_HAIRCUT * condition_boost(cond)

def estimate_net_profit(buy_price: float, shipping: float, close_est: float) -> Tuple[float, float]:
    if close_est <= 0 or buy_price <= 0:
        return -9999.0, -1.0
    proceeds = close_est
    cost = buy_price + shipping
    net = proceeds - cost
    roi = net / cost if cost > 0 else -1.0
    return net, roi

def extract_listing_urls(listing_html: str) -> List[Tuple[str, str]]:
    soup = BeautifulSoup(listing_html, "lxml")
    links = soup.select('a[href*="/segunda-mano/"][href$=".html"]')
    out = []
    for a in links:
        href = (a.get("href") or "").strip()
        if not href:
            continue
        url = urljoin(BASE, href)
        m = ID_RE.search(url)
        cc_id = m.group(1) if m else url
        out.append((cc_id, url))
    return out

def parse_detail_page(detail_html: str) -> Dict:
    soup = BeautifulSoup(detail_html, "lxml")
    h1 = soup.select_one("h1")
    title = h1.get_text(" ", strip=True) if h1 else "Sin título"

    text = soup.get_text(" ", strip=True)
    m = PRICE_RE.search(text)
    price = euro_to_float(m.group(1)) if m else None

    lower = normalize(text)
    estado = ""
    for c in ("perfecto", "muy bueno", "bueno", "usado"):
        if c in lower:
            estado = c
            break

    disponibilidad = "envío" if ("a domicilio" in lower or "envio" in lower) else "tienda"

    tienda = ""
    for s in soup.stripped_strings:
        if "cash converters" in s.lower():
            tienda = s.strip()
            break

    desc = ""
    desc_el = soup.select_one('[class*="description"], [id*="description"]')
    if desc_el:
        desc = desc_el.get_text(" ", strip=True)

    return {
        "title": title,
        "price": price,
        "estado": estado,
        "disponibilidad": disponibilidad,
        "tienda": tienda,
        "raw_text": text,
        "description": desc,
    }

def is_reputable_listing(title: str, raw_text: str) -> Tuple[bool, str]:
    t = normalize(title + " " + raw_text)
    if has_banned_keywords(t):
        return False, "banned_keywords"
    b = brand_from_text(t)
    if b is None:
        return False, "no_brand"
    if b.startswith("__banned__"):
        return False, "banned_brand"
    return True, "ok"

def cond_label(estado: str, raw_text: str) -> str:
    t = normalize((estado or "") + " " + (raw_text or ""))
    if any(k in t for k in POSITIVE_KEYWORDS):
        return f"{estado or '—'}+"
    return f"{estado or '—'}"

def run():
    targets = load_targets("target_list.json")

    dedup = set()
    scanned = 0

    # Debug counters
    c_brand_ok = 0
    c_match_ok = 0
    c_net_ok = 0

    # For debug: keep best candidates even if rejected by net/roi
    candidates = []

    opportunities = []

    for seed in SEED_URLS:
        start = 0
        while scanned < CC_MAX_ITEMS:
            page_url = build_page_url(seed, start)
            listing_html = fetch(page_url)
            pairs = extract_listing_urls(listing_html)
            if not pairs:
                break

            for cc_id, url in pairs:
                if cc_id in dedup:
                    continue
                dedup.add(cc_id)

                detail_html = fetch(url)
                data = parse_detail_page(detail_html)

                scanned += 1
                time.sleep(CC_THROTTLE_S)

                title = data.get("title", "")
                raw_text = data.get("raw_text", "")

                ok_brand, why_brand = is_reputable_listing(title, raw_text)
                if not ok_brand:
                    continue
                c_brand_ok += 1

                buy = data.get("price") or 0.0
                shipping = DEFAULT_SHIPPING_EUR

                best_target, match = pick_best_target(title, targets)
                if not best_target:
                    continue

                close_est = estimate_close(best_target, data.get("estado", ""))
                net, roi = estimate_net_profit(buy, shipping, close_est)

                # candidate snapshot for debugging
                candidates.append({
                    "title": title,
                    "url": url,
                    "buy": buy,
                    "close_est": close_est,
                    "net": net,
                    "roi": roi,
                    "match": match,
                    "target": best_target.get("id", "—"),
                    "cond": cond_label(data.get("estado", ""), raw_text),
                    "shop": data.get("tienda") or "—",
                })

                if match < MIN_MATCH_SCORE:
                    continue
                c_match_ok += 1

                if not (net >= MIN_NET_EUR or roi >= MIN_NET_ROI):
                    continue
                c_net_ok += 1

                opportunities.append(candidates[-1])

                if scanned >= CC_MAX_ITEMS:
                    break

            start += PAGE_SIZE
            time.sleep(CC_THROTTLE_S)

            if scanned >= CC_MAX_ITEMS:
                break

        if scanned >= CC_MAX_ITEMS:
            break

    opportunities.sort(key=lambda x: (x["net"], x["match"]), reverse=True)
    top = opportunities[:10]

    header = f"🕗 TIMELAB Morning Scan — TOP {len(top)} (CashConverters ES)"
    lines = [header]

    for i, it in enumerate(top, 1):
        buy = f"{it['buy']:.2f}€"
        close = f"{it['close_est']:.0f}€" if it["close_est"] > 0 else "—"
        net = f"{it['net']:.0f}€"
        roi = f"{it['roi']*100:.0f}%"
        lines.append(
            f"{i}) [cc] {it['title']}\n"
            f"   💶 Compra: {buy} | 🚚 Envío: {DEFAULT_SHIPPING_EUR:.2f}€ | 🎯 Cierre est.: {close}\n"
            f"   ✅ Neto est.: {net} | ROI: {roi} | Match: {it['match']} | Cond: {it['cond']}\n"
            f"   🧩 Target: {it['target']}\n"
            f"   📍 {it['shop']}\n"
            f"   🔗 {it['url']}"
        )

    if not top:
        lines.append("\nNo se encontraron oportunidades que cumplan filtros (marca + match + net/ROI).")

    tg_send("\n\n".join(lines))

    # DEBUG SUMMARY MESSAGE (separado)
    if DEBUG_CC:
        candidates.sort(key=lambda x: (x["match"], x["net"]), reverse=True)
        cand_top = candidates[:10]
        d = [
            f"🧪 TIMELAB CC Debug — scanned:{scanned} | brand_ok:{c_brand_ok} | match_ok:{c_match_ok} | net_ok:{c_net_ok}",
            f"thresholds: match>={MIN_MATCH_SCORE} | net>={MIN_NET_EUR} OR roi>={MIN_NET_ROI} | haircut:{CLOSE_HAIRCUT}",
            "",
            "Top candidates (even if rejected):"
        ]
        for i, it in enumerate(cand_top, 1):
            d.append(
                f"{i}) {it['title']}\n"
                f"   buy:{it['buy']:.2f}€ close:{(it['close_est'] or 0):.0f}€ net:{it['net']:.0f}€ roi:{(it['roi']*100 if it['roi']!=-1 else -1):.0f}% match:{it['match']} target:{it['target']}\n"
                f"   {it['url']}"
            )
        tg_send("\n\n".join(d))

if __name__ == "__main__":
    try:
        run()
    except Exception:
        err = traceback.format_exc()
        try:
            tg_send("❌ TIMELAB CashConverters scanner crashed\n\n" + err[:3500])
        except Exception:
            pass
        raise SystemExit(0)