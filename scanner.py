import os
import re
import time
import math
import json
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import requests
from bs4 import BeautifulSoup

# =========================
# CONFIG (TIMELAB)
# =========================

MIN_NET_EUR = 120
MIN_NET_ROI = 0.25
MIN_MATCH_SCORE = 75
ALLOW_FAKE_RISK = {"low", "medium"}  # never "high"

# TIMELAB cost model (ajustable)
CATWIKI_COMMISSION = 0.125          # 12.5% sobre martillo (aprox)
PAYMENT_PROCESSING = 0.0            # si quieres modelar PayPal/Stripe etc
PACKAGING_EUR = 5.0                 # coste medio empaquetado
MISC_EUR = 5.0                      # limpieza, consumibles, etc
SHIP_ARBITRAGE_EUR = 35.0           # tu arbitraje a favor (ajústalo si cambia)

# Impuestos (placeholder: ajústalo según tu régimen real / cálculo neto)
# Si estás en REBU, el IVA va sobre margen. Aquí ponemos un "impuesto efectivo" conservador.
EFFECTIVE_TAX_RATE_ON_PROFIT = 0.15  # 15% sobre beneficio estimado (conservador; ajustable)

# eBay: dominios UE (preferimos España primero)
EBAY_SEARCH_BASES = [
    "https://www.ebay.es/sch/i.html",
    "https://www.ebay.fr/sch/i.html",
    "https://www.ebay.de/sch/i.html",
    "https://www.ebay.it/sch/i.html",
]

EU_COUNTRIES = {
    "spain", "españa", "france", "francia", "germany", "alemania", "italy", "italia",
    "portugal", "belgium", "belgica", "netherlands", "países bajos", "austria", "ireland",
    "finland", "sweden", "denmark", "poland", "czech", "slovakia", "slovenia", "croatia",
    "hungary", "romania", "bulgaria", "greece", "luxembourg", "latvia", "lithuania", "estonia",
    "cyprus", "malta"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TIMELABScanner/1.0; +https://github.com/)",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}

# =========================
# TARGET LIST v1.0 (MVP)
# =========================

@dataclass
class TargetModel:
    key: str
    keywords: List[str]        # must include brand/model tokens
    refs: List[str]            # ref patterns (optional)
    tier: str                  # A-E
    fake_risk: str             # low/medium/high
    catwiki_close_med: float   # benchmark cierre medio (EUR)
    buy_max: float             # compra máxima objetivo (EUR)

TARGETS: List[TargetModel] = [
    # ---- TISSOT ----
    TargetModel("tissot_seastar_pr516", ["tissot", "seastar", "pr", "516"], [], "A", "low", 750, 380),
    TargetModel("tissot_pr516_chrono", ["tissot", "pr", "516", "chrono"], ["7733", "7734"], "B", "low", 900, 450),
    TargetModel("tissot_visodate_auto", ["tissot", "visodate", "automatic"], [], "A", "low", 650, 320),
    TargetModel("tissot_navigator", ["tissot", "navigator"], [], "B", "low", 700, 350),

    # ---- TAG HEUER ----
    TargetModel("tag_f1_waz1110", ["tag", "heuer", "formula", "1"], ["waz1110", "waz1010", "caz101"], "A", "medium", 850, 450),
    TargetModel("tag_2000", ["tag", "heuer", "2000"], [], "A", "medium", 800, 420),
    TargetModel("tag_4000", ["tag", "heuer", "4000"], [], "B", "medium", 700, 380),
    TargetModel("tag_kirium", ["tag", "heuer", "kirium"], [], "B", "medium", 700, 380),

    # ---- LONGINES ----
    TargetModel("longines_conquest_vintage", ["longines", "conquest"], [], "A", "low", 900, 450),
    TargetModel("longines_admiral_hf", ["longines", "admiral", "hf"], [], "A", "low", 850, 420),
    TargetModel("longines_flagship_auto", ["longines", "flagship", "automatic"], [], "B", "low", 800, 400),

    # ---- OMEGA (más riesgo, controlado) ----
    TargetModel("omega_seamaster_cosmic", ["omega", "seamaster", "cosmic"], [], "B", "high", 1100, 520),
    TargetModel("omega_geneve_auto", ["omega", "geneve", "automatic"], [], "B", "high", 950, 480),
    TargetModel("omega_deville_prestige", ["omega", "de", "ville", "prestige"], [], "B", "high", 900, 450),

    # ---- ZENITH ----
    TargetModel("zenith_cosmopolitan", ["zenith", "cosmopolitan"], [], "B", "medium", 850, 420),
    TargetModel("zenith_defy_quartz", ["zenith", "defy", "quartz"], [], "C", "medium", 750, 380),

    # ---- SEIKO CHRONO (más riesgo de piezas/franken, pero no “fakes” típicos) ----
    TargetModel("seiko_6139_pogue", ["seiko", "6139", "pogue"], ["6139"], "B", "medium", 1200, 600),
    TargetModel("seiko_6138", ["seiko", "6138"], ["6138"], "C", "medium", 1100, 550),
]

# Importante: nunca permitir "high" (Omega) en el filtro final según tu regla
# => en el filtro final, Omega quedará fuera automáticamente hasta que definamos una lógica anti-fake más fuerte.

# =========================
# DATA STRUCTURES
# =========================

@dataclass
class Listing:
    source: str
    country_site: str
    title: str
    price_eur: float
    shipping_eur: float
    url: str
    location_text: str

@dataclass
class Candidate:
    listing: Listing
    target: TargetModel
    match_score: int
    fake_risk: str
    expected_close: float
    net_profit: float
    net_roi: float

# =========================
# HELPERS
# =========================

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def extract_price(text: str) -> Optional[float]:
    if not text:
        return None
    t = text.replace("\xa0", " ").replace(".", "").replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)", t)
    if not m:
        return None
    try:
        return float(m.group(1))
    except:
        return None

def is_eu_location(loc: str) -> bool:
    l = norm(loc)
    if not l:
        return True  # si no tenemos ubicación, no descartamos en MVP
    return any(c in l for c in EU_COUNTRIES)

def compute_match_score(title: str, target: TargetModel) -> int:
    t = norm(title)
    score = 0

    # keywords
    kw_hits = 0
    for kw in target.keywords:
        if norm(kw) in t:
            kw_hits += 1
    score += int(60 * (kw_hits / max(1, len(target.keywords))))

    # refs
    if target.refs:
        ref_hits = 0
        for r in target.refs:
            if norm(r) in t:
                ref_hits += 1
        score += int(25 * (ref_hits / max(1, len(target.refs))))
    else:
        score += 10  # pequeño bonus si no hay ref requerida

    # penalties for suspicious words
    suspicious = ["replica", "copy", "imitacion", "imitation", "fake"]
    if any(w in t for w in suspicious):
        score -= 50

    # cap
    score = max(0, min(100, score))
    return score

def estimate_net_profit(buy_price: float, ship_in: float, expected_close: float) -> Tuple[float, float]:
    """
    buy_price: precio anuncio
    ship_in: envío entrada
    expected_close: cierre esperado catawiki (martillo aprox)
    """
    cost_in = buy_price + ship_in + PACKAGING_EUR + MISC_EUR
    revenue_gross = expected_close + SHIP_ARBITRAGE_EUR  # arbitraje a favor
    fees = expected_close * CATWIKI_COMMISSION + PAYMENT_PROCESSING
    profit_before_tax = revenue_gross - fees - cost_in
    tax = max(0.0, profit_before_tax) * EFFECTIVE_TAX_RATE_ON_PROFIT
    net = profit_before_tax - tax
    roi = net / max(1.0, cost_in)
    return net, roi

# =========================
# EBAY SCRAPER (HTML)
# =========================

def ebay_search(model: TargetModel, base_url: str, max_pages: int = 1) -> List[Listing]:
    out: List[Listing] = []
    query = " ".join(model.keywords)

    for page in range(1, max_pages + 1):
        params = {
            "_nkw": query,
            "_sop": "10",   # Newly listed
            "LH_BIN": "1",  # Buy It Now (reduce auction noise)
            "_pgn": str(page),
        }

        r = requests.get(base_url, params=params, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            continue

        soup = BeautifulSoup(r.text, "lxml")
        items = soup.select("li.s-item")

        for it in items:
            title_el = it.select_one(".s-item__title")
            price_el = it.select_one(".s-item__price")
            ship_el = it.select_one(".s-item__shipping, .s-item__logisticsCost")
            link_el = it.select_one("a.s-item__link")
            loc_el = it.select_one(".s-item__location")

            if not title_el or not price_el or not link_el:
                continue

            title = title_el.get_text(" ", strip=True)
            if title.lower() in {"shop on ebay", "explore more options"}:
                continue

            price = extract_price(price_el.get_text(" ", strip=True))
            if price is None:
                continue

            shipping = 0.0
            if ship_el:
                txt = ship_el.get_text(" ", strip=True)
                if "free" in txt.lower() or "gratis" in txt.lower():
                    shipping = 0.0
                else:
                    sp = extract_price(txt)
                    shipping = sp if sp is not None else 0.0

            url = link_el.get("href", "")
            loc = loc_el.get_text(" ", strip=True) if loc_el else ""

            out.append(Listing(
                source="ebay",
                country_site=base_url.split("//")[1].split("/")[0],
                title=title,
                price_eur=float(price),
                shipping_eur=float(shipping),
                url=url,
                location_text=loc
            ))

        time.sleep(1.0)  # pequeño throttle

    return out

# =========================
# TELEGRAM
# =========================

def tg_send(text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
    r = requests.post(url, json=payload, timeout=30)
    print("Telegram status:", r.status_code)
    print("Telegram response:", r.text)
    r.raise_for_status()

# =========================
# MAIN
# =========================

def main():
    now_utc = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # 1) Collect listings (eBay UE)
    all_listings: List[Listing] = []
    for t in TARGETS:
        # MVP: solo 1 página por marketplace para no saturar
        for base in EBAY_SEARCH_BASES:
            try:
                all_listings.extend(ebay_search(t, base_url=base, max_pages=1))
            except Exception as e:
                print("EBAY ERROR:", base, t.key, str(e))

    # 2) Score & filter
    candidates: List[Candidate] = []
    for li in all_listings:
        if not is_eu_location(li.location_text):
            continue

        for t in TARGETS:
            ms = compute_match_score(li.title, t)
            if ms < 50:
                continue

            # Comprar por debajo de objetivo (flexible)
            if li.price_eur > (t.buy_max * 1.15):
                continue

            expected_close = t.catwiki_close_med
            net, roi = estimate_net_profit(li.price_eur, li.shipping_eur, expected_close)

            candidates.append(Candidate(
                listing=li,
                target=t,
                match_score=ms,
                fake_risk=t.fake_risk,
                expected_close=expected_close,
                net_profit=net,
                net_roi=roi
            ))

    # 3) Apply TIMELAB rules
    filtered: List[Candidate] = []
    for c in candidates:
        if c.match_score < MIN_MATCH_SCORE:
            continue
        if c.fake_risk not in ALLOW_FAKE_RISK:
            continue
        if not (c.net_profit >= MIN_NET_EUR or c.net_roi >= MIN_NET_ROI):
            continue
        filtered.append(c)

    # 4) Rank
    filtered.sort(key=lambda x: (x.net_profit, x.match_score), reverse=True)
    top = filtered[:10]

    # 5) Telegram report
    if not top:
        msg = f"🕗 TIMELAB Morning Scan (eBay UE)\nTimestamp: {now_utc}\n\nNo hay oportunidades que pasen los filtros:\n- ≥ {MIN_NET_EUR}€ neto o ≥ {int(MIN_NET_ROI*100)}% ROI neto\n- match ≥ {MIN_MATCH_SCORE}/100\n- fake risk low/medium"
        tg_send(msg)
        return

    lines = [
        f"🕗 TIMELAB Morning Scan (eBay UE) — TOP {len(top)}",
        f"Timestamp: {now_utc}",
        "",
        f"Filtros: net≥{MIN_NET_EUR}€ o ROI≥{int(MIN_NET_ROI*100)}% | match≥{MIN_MATCH_SCORE} | fake: low/medium",
        ""
    ]

    for i, c in enumerate(top, 1):
        li = c.listing
        lines.append(
            f"{i}) {li.title}\n"
            f"   💶 Compra: {li.price_eur:.0f}€ + envío {li.shipping_eur:.0f}€ | 🎯 Cierre est.: {c.expected_close:.0f}€\n"
            f"   ✅ Neto est.: {c.net_profit:.0f}€ | ROI: {int(c.net_roi*100)}% | Match: {c.match_score}/100 | Src: {li.country_site}\n"
            f"   🔗 {li.url}\n"
        )

    tg_send("\n".join(lines))

if __name__ == "__main__":
    main()