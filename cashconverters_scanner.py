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

# Límite duro de fichas a visitar
CC_MAX_ITEMS = int(os.getenv("CC_MAX_ITEMS", "400"))

# Objetivo de “marcas reputadas encontradas” (si se alcanza antes, se para)
CC_GOOD_BRANDS_TARGET = int(os.getenv("CC_GOOD_BRANDS_TARGET", "60"))

MIN_MATCH_SCORE = int(os.getenv("CC_MIN_MATCH_SCORE", "65"))
MIN_NET_EUR = float(os.getenv("CC_MIN_NET_EUR", "20"))
MIN_NET_ROI = float(os.getenv("CC_MIN_NET_ROI", "0.08"))
CLOSE_HAIRCUT = float(os.getenv("CLOSE_HAIRCUT", "0.90"))

DEFAULT_SHIPPING_EUR = float(os.getenv("CC_DEFAULT_SHIPPING_EUR", "0.0"))
DEBUG_CC = os.getenv("CC_DEBUG", "1").strip() == "1"

PAGE_SIZE = int(os.getenv("CC_PAGE_SIZE", "24"))
SRULE = os.getenv("CC_SRULE", "new")

SESSION = requests.Session()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Connection": "keep-alive",
}

# Seeds (pulsera y premium/alta gama)
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

# Whitelist (marcas “Catawiki-friendly”)
REPUTABLE_BRANDS = {
    "rolex", "tudor", "omega", "longines", "tag heuer", "breitling",
    "zenith", "jaeger lecoultre", "iwc", "panerai", "cartier",
    "bulgari", "baume mercier",
    "tissot", "hamilton", "certina", "oris", "rado",
    "seiko", "grand seiko", "citizen", "orient", "mido",
    "edox", "doxa", "eterna", "fortis", "sinn", "nomos",
    "maurice lacroix", "frederique constant",
    "vacheron constantin", "audemars piguet", "patek philippe",
    "girard perregaux", "glashutte", "glashutte original",
    "hublot", "chopard", "montblanc", "u boat", "ulysse nardin",
    "raymond weil", "alpina", "laco", "stowa",
}

# Blacklist (moda / smart / low-value para Catawiki)
BANNED_BRANDS = {
    "lotus", "festina", "diesel", "armani", "emporio armani", "michael kors",
    "guess", "dkny", "fossil", "police", "hugo boss", "boss", "swatch",
    "samsung", "xiaomi", "huawei", "apple", "garmin", "fitbit",
    "welder", "ice watch", "icewatch", "tommy hilfiger", "calvin klein",
}

# Keywords negativos fuertes
BANNED_KEYWORDS = {
    "smartwatch", "reloj inteligente", "galaxy watch", "apple watch", "fitbit",
    "pulsera actividad", "activity", "fitness",
    "sin funcionar", "no funciona", "para piezas", "solo piezas",
    "incompleto", "averiado", "defectuoso", "rotura", "no arranca",
}

# Keywords positivos suaves (solo ajustan “cond”)
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
    r = SESSION.post(url, timeout=CC_TIMEOUT, json={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    })
    r.raise_for_status()

def fetch(url: str) -> Tuple[str, int]:
    r = SESSION.get(url, headers=HEADERS, timeout=CC_TIMEOUT, allow_redirects=True)
    return r.text, r.status_code

def build_page_url(seed: str, start: int) -> str:
    joiner = "&" if "?" in seed else "?"
    return f"{seed}{joiner}srule={SRULE}&start={start}&sz={PAGE_SIZE}"

def canon(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("&", " ")
    s = re.sub(r"[-_/\.]", " ", s)
    s = re.sub(r"[^a-z0-9áéíóúñü ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    trans = str.maketrans("áéíóúñü", "aeiounu")
    s = s.translate(trans)
    return s

REPUTABLE_CANON = {canon(b) for b in REPUTABLE_BRANDS}
BANNED_CANON = {canon(b) for b in BANNED_BRANDS}

def has_banned_keywords(text: str) -> bool:
    t = canon(text)
    return any(canon(k) in t for k in BANNED_KEYWORDS)

def extract_brand_from_text(text: str) -> Optional[str]:
    t = canon(text)
    for b in sorted(REPUTABLE_CANON, key=len, reverse=True):
        if b and b in t:
            return b
    for b in sorted(BANNED_CANON, key=len, reverse=True):
        if b and b in t:
            return f"__banned__:{b}"
    return None

def condition_boost(cond: str) -> float:
    c = canon(cond)
    if "perfecto" in c or "muy bueno" in c:
        return 1.00
    if "bueno" in c:
        return 0.97
    if "usado" in c:
        return 0.93
    return 0.95

def compute_match_score_simple(title: str, target: Dict) -> int:
    text = canon(title)
    score = 0
    t_brand = canon(target.get("brand", ""))
    if t_brand and t_brand in text:
        score += 55
    for kw in target.get("keywords", []) or []:
        kw_n = canon(kw)
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

def extract_brand_from_breadcrumbs(soup: BeautifulSoup) -> str:
    """
    CashConverters suele poner la marca en el breadcrumb:
    'Relojes > ... > Lotus' o similar.
    Intentamos encontrar cualquier anchor cuyo texto sea una marca conocida (reputable o banned).
    """
    crumbs = soup.select("nav a, nav span, ol li a, ol li span, ul li a, ul li span")
    for node in crumbs:
        txt = canon(node.get_text(" ", strip=True))
        if not txt:
            continue
        if txt in REPUTABLE_CANON:
            return txt
        if txt in BANNED_CANON:
            return f"__banned__:{txt}"
    return ""

def parse_detail_page(url: str) -> Dict:
    html, status = fetch(url)
    soup = BeautifulSoup(html, "lxml")

    title_tag = soup.title.get_text(" ", strip=True) if soup.title else ""
    h1 = soup.select_one("h1")
    title = h1.get_text(" ", strip=True) if h1 else ""

    text = soup.get_text(" ", strip=True)

    # Precio
    m = PRICE_RE.search(text)
    price = euro_to_float(m.group(1)) if m else None

    # Estado
    lower = canon(text)
    estado = ""
    for c in ("perfecto", "muy bueno", "bueno", "usado"):
        if c in lower:
            estado = c
            break

    # Disponibilidad (heurística simple)
    disponibilidad = "envío" if ("a domicilio" in lower or "envio" in lower) else "tienda"

    # Tienda (heurística simple)
    tienda = ""
    for s in soup.stripped_strings:
        if "cash converters" in s.lower():
            tienda = s.strip()
            break

    # ✅ Marca: primero breadcrumbs (más fiable), luego fallback a texto
    brand = extract_brand_from_breadcrumbs(soup)
    if not brand:
        brand = extract_brand_from_text(title + " " + title_tag + " " + text) or ""

    # Page ok
    page_ok = bool(title) and (price is not None)

    return {
        "url": url,
        "status": status,
        "html_len": len(html),
        "title_tag": title_tag[:180],
        "page_ok": page_ok,
        "title": title or title_tag,
        "brand": brand,
        "price": price,
        "estado": estado,
        "disponibilidad": disponibilidad,
        "tienda": tienda,
        "raw_text": text,
    }

def classify_brand(brand: str) -> str:
    if not brand:
        return "no_brand"
    if brand.startswith("__banned__"):
        return "banned_brand"
    b = canon(brand)
    if b in REPUTABLE_CANON:
        return "reputable"
    if b in BANNED_CANON:
        return "banned_brand"
    return "not_reputable"

def cond_label(estado: str, raw_text: str) -> str:
    t = canon((estado or "") + " " + (raw_text or ""))
    if any(canon(k) in t for k in POSITIVE_KEYWORDS):
        return f"{estado or '—'}+"
    return f"{estado or '—'}"

def run():
    targets = load_targets("target_list.json")

    dedup = set()
    scanned = 0

    c_page_bad = 0
    c_brand_reputable = 0
    c_brand_banned = 0
    c_brand_notrep = 0
    c_brand_none = 0
    c_match_ok = 0
    c_net_ok = 0

    opportunities = []

    for seed in SEED_URLS:
        start = 0
        while scanned < CC_MAX_ITEMS and c_brand_reputable < CC_GOOD_BRANDS_TARGET:
            page_url = build_page_url(seed, start)
            listing_html, _ = fetch(page_url)
            pairs = extract_listing_urls(listing_html)
            if not pairs:
                break

            for cc_id, url in pairs:
                if scanned >= CC_MAX_ITEMS or c_brand_reputable >= CC_GOOD_BRANDS_TARGET:
                    break
                if cc_id in dedup:
                    continue
                dedup.add(cc_id)

                data = parse_detail_page(url)
                scanned += 1
                time.sleep(CC_THROTTLE_S)

                if not data.get("page_ok"):
                    c_page_bad += 1
                    continue

                # keywords negativos (smartwatch / piezas / no funciona)
                if has_banned_keywords(data.get("title", "") + " " + data.get("raw_text", "")):
                    continue

                brand_class = classify_brand(data.get("brand", ""))

                # ✅ contadores consistentes
                if brand_class == "reputable":
                    c_brand_reputable += 1
                elif brand_class == "banned_brand":
                    c_brand_banned += 1
                    continue
                elif brand_class == "not_reputable":
                    c_brand_notrep += 1
                    continue
                else:
                    c_brand_none += 1
                    continue

                title = data.get("title", "")
                raw_text = data.get("raw_text", "")
                buy = float(data.get("price") or 0.0)
                shipping = DEFAULT_SHIPPING_EUR

                best_target, match = pick_best_target(title, targets)
                if not best_target:
                    continue

                close_est = estimate_close(best_target, data.get("estado", ""))
                net, roi = estimate_net_profit(buy, shipping, close_est)

                if match < MIN_MATCH_SCORE:
                    continue
                c_match_ok += 1

                if not (net >= MIN_NET_EUR or roi >= MIN_NET_ROI):
                    continue
                c_net_ok += 1

                opportunities.append({
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

            start += PAGE_SIZE
            time.sleep(CC_THROTTLE_S)

        if scanned >= CC_MAX_ITEMS or c_brand_reputable >= CC_GOOD_BRANDS_TARGET:
            break

    opportunities.sort(key=lambda x: (x["net"], x["match"]), reverse=True)
    top = opportunities[:10]

    header = f"🕗 TIMELAB Morning Scan — TOP {len(top)} (CashConverters ES)"
    lines = [header]

    for i, it in enumerate(top, 1):
        buy = f"{it['buy']:.2f}€"
        close = f"{it['close_est']:.0f}€" if it["close_est"] > 0 else "—"
        net_s = f"{it['net']:.0f}€"
        roi_s = f"{it['roi']*100:.0f}%"
        lines.append(
            f"{i}) [cc] {it['title']}\n"
            f"   💶 Compra: {buy} | 🚚 Envío: {DEFAULT_SHIPPING_EUR:.2f}€ | 🎯 Cierre est.: {close}\n"
            f"   ✅ Neto est.: {net_s} | ROI: {roi_s} | Match: {it['match']} | Cond: {it['cond']}\n"
            f"   🧩 Target: {it['target']}\n"
            f"   📍 {it['shop']}\n"
            f"   🔗 {it['url']}"
        )

    if not top:
        lines.append("\nNo se encontraron oportunidades que cumplan filtros (marca reputada + match + net/ROI).")

    tg_send("\n\n".join(lines))

    if DEBUG_CC:
        d = [
            f"🧪 TIMELAB CC Debug — scanned:{scanned} | page_bad:{c_page_bad}",
            f"brands: reputable:{c_brand_reputable} | banned:{c_brand_banned} | not_reputable:{c_brand_notrep} | no_brand:{c_brand_none}",
            f"passed: match_ok:{c_match_ok} | net_ok:{c_net_ok}",
            f"thresholds: match>={MIN_MATCH_SCORE} | net>={MIN_NET_EUR} OR roi>={MIN_NET_ROI} | haircut:{CLOSE_HAIRCUT}",
            f"stop: max_items:{CC_MAX_ITEMS} | good_brands_target:{CC_GOOD_BRANDS_TARGET}",
        ]
        tg_send("\n".join(d))

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