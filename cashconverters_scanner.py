import os
import re
import time
import json
import math
import traceback
from typing import Dict, List, Tuple, Optional
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# =========================
# CONFIG (TIMELAB CC)
# =========================
BASE = "https://www.cashconverters.es"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

CC_TIMEOUT = float(os.getenv("CC_TIMEOUT", "20"))
CC_THROTTLE_S = float(os.getenv("CC_THROTTLE_S", "0.8"))
CC_MAX_ITEMS = int(os.getenv("CC_MAX_ITEMS", "100"))

MIN_MATCH_SCORE = int(os.getenv("CC_MIN_MATCH_SCORE", "65"))
MIN_NET_EUR = float(os.getenv("CC_MIN_NET_EUR", "20"))
MIN_NET_ROI = float(os.getenv("CC_MIN_NET_ROI", "0.08"))  # 8%
CLOSE_HAIRCUT = float(os.getenv("CLOSE_HAIRCUT", "0.90"))

# Si no detectamos envío, asumimos 0 (CashConverters suele incluir envío/recogida variable)
DEFAULT_SHIPPING_EUR = float(os.getenv("CC_DEFAULT_SHIPPING_EUR", "0.0"))

# Comisión Catawiki aprox. (para estimación neta simple)
# Ajusta si en tu TIMELAB tienes otra fórmula exacta.
CATWIKI_BUYER_PREMIUM = float(os.getenv("CATWIKI_BUYER_PREMIUM", "0.125"))  # 12.5%

PAGE_SIZE = int(os.getenv("CC_PAGE_SIZE", "24"))
SRULE = os.getenv("CC_SRULE", "new")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TIMELAB-WATCHES/1.0)",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
}

# Seeds: reloj pulsera / premium / alta gama
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

# =========================
# BRAND FILTERS
# =========================
# Lista “reputada / Catawiki-friendly” (puedes ampliarla)
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

# Excluir explícitamente moda/baratas / smartwatches / electrónicas
BANNED_BRANDS = {
    "lotus", "festina", "diesel", "armani", "emporio armani", "michael kors",
    "guess", "dkny", "fossil", "police", "hugo boss", "boss", "swatch",
    "samsung", "xiaomi", "huawei", "apple", "garmin", "fitbit",
    "welder", "ice watch", "icewatch", "tommy hilfiger", "calvin klein",
}

BANNED_KEYWORDS = {
    "smartwatch", "watch", "fitness", "pulsera actividad", "activity",
    "reloj inteligente", "galaxy watch", "apple watch",
    "sin funcionar", "no funciona", "para piezas", "solo piezas",
    "incompleto", "averiado", "defectuoso", "rotura", "no arranca",
}

POSITIVE_KEYWORDS = {"revisado", "funciona", "buen estado", "perfecto", "recien revisado", "recién revisado"}

# =========================
# PARSING UTILS
# =========================
PRICE_RE = re.compile(r"(\d{1,3}(?:\.\d{3})*,\d{2})\s*€")
ID_RE = re.compile(r"/segunda-mano/([^/]+)\.html")

def euro_to_float(s: str) -> float:
    s = s.replace(".", "").replace(",", ".")
    return float(s)

def tg_send(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID (GitHub Secrets?)")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    r = requests.post(
        url,
        timeout=CC_TIMEOUT,
        json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True},
    )
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

def brand_from_text(text: str) -> Optional[str]:
    t = normalize(text)
    # Si contiene alguna marca reputada, devuélvela
    for b in sorted(REPUTABLE_BRANDS, key=len, reverse=True):
        if b in t:
            return b
    # Si contiene alguna marca baneada, marcamos como baneada
    for b in sorted(BANNED_BRANDS, key=len, reverse=True):
        if b in t:
            return f"__banned__:{b}"
    return None

def has_banned_keywords(text: str) -> bool:
    t = normalize(text)
    return any(k in t for k in BANNED_KEYWORDS)

def condition_boost(cond: str) -> float:
    # Ajuste leve del cierre por estado (conservador)
    c = normalize(cond)
    if "perfecto" in c or "muy bueno" in c:
        return 1.00
    if "bueno" in c:
        return 0.97
    if "usado" in c:
        return 0.93
    return 0.95

def compute_match_score_simple(title: str, target: Dict) -> int:
    """
    Match simple y robusto:
    - suma puntos si aparecen brand / model keywords del target
    - penaliza si no aparece la marca
    """
    text = normalize(title)
    score = 0

    t_brand = normalize(target.get("brand", ""))
    if t_brand and t_brand in text:
        score += 55
    else:
        score += 0

    # keywords opcionales
    kws = target.get("keywords", []) or []
    for kw in kws:
        kw_n = normalize(kw)
        if kw_n and kw_n in text:
            score += 10

    # cap
    return int(max(0, min(100, score)))

def load_targets(path: str = "target_list.json") -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Formatos soportados:
    # - lista de targets
    # - dict {targets:[...]}
    if isinstance(data, dict) and "targets" in data:
        data = data["targets"]

    if not isinstance(data, list):
        raise ValueError("target_list.json must be a list or {targets:[...]}")

    # Normaliza esperados:
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
    est = base * CLOSE_HAIRCUT * condition_boost(cond)
    return float(est)

def estimate_net_profit(buy_price: float, shipping: float, close_est: float) -> Tuple[float, float]:
    """
    Estimación neta SIMPLE:
    - close_est: martillo estimado (lo que paga comprador a Catawiki por el reloj)
    - aproximamos que el vendedor recibe close_est (sin premium) y coste total = buy + shipping
    - si quieres la versión exacta TIMELAB (comisiones, IVA, arbitraje envío), lo metemos en el siguiente commit
    """
    if close_est <= 0 or buy_price <= 0:
        return -9999.0, -1.0

    proceeds = close_est  # aproximación conservadora a "martillo"
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
    price = euro_to_float(m