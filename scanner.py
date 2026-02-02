import os
import re
import time
import json
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

# =========================
# CONFIG (TIMELAB)
# =========================

MIN_NET_EUR = 120
MIN_NET_ROI = 0.25
MIN_MATCH_SCORE = 75
ALLOW_FAKE_RISK = {"low", "medium"}  # nunca "high"

# TIMELAB cost model (ajustable)
CATWIKI_COMMISSION = 0.125          # 12.5% sobre martillo (aprox)
PAYMENT_PROCESSING = 0.0            # si quieres modelar procesado pago
PACKAGING_EUR = 5.0                 # empaquetado medio
MISC_EUR = 5.0                      # consumibles / limpieza
SHIP_ARBITRAGE_EUR = 35.0           # arbitraje envío a tu favor (ajústalo)

# Impuestos (placeholder conservador; ajusta a tu régimen real)
EFFECTIVE_TAX_RATE_ON_PROFIT = 0.15

# eBay: dominios UE
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
# DATA STRUCTURES
# =========================

@dataclass
class TargetModel:
    key: str
    keywords: List[str]        # tokens para búsqueda y scoring
    refs: List[str]            # refs opcionales
    tier: str                  # A-E (informativo)
    fake_risk: str             # low/medium/high
    catwiki_close_med: float   # benchmark cierre medio (EUR)
    buy_max: float             # compra máxima objetivo (EUR)

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
    # Normaliza separadores: eBay puede usar puntos/commas según país
    t = text.replace("\xa0", " ").replace(".", "").replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)", t)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def is_eu_location(loc: str) -> bool:
    l = norm(loc)
    if not l:
        return True  # si no hay ubicación, no descartamos en MVP
    return any(c in l for c in EU_COUNTRIES)

def compute_match_score(title: str, target: TargetModel) -> int:
    t = norm(title)
    score = 0

    # keywords (peso fuerte)
    kw_hits = 0
    for kw in target.keywords:
        if norm(kw) in t:
            kw_hits += 1
    score += int(60 * (kw_hits / max(1, len(target.keywords))))

    # refs (peso medio)
    if target.refs:
        ref_hits = 0
        for r in target.refs:
            if norm(r) in t:
                ref_hits += 1
        score += int(25 * (ref_hits / max(1, len(target.refs))))
    else:
        score += 10

    # penalizaciones por palabras sospechosas
    suspicious = ["replica", "copy", "imitacion", "imitación", "imitation", "fake"]
    if any(w in t for w in suspicious):
        score -= 50

    return max(0, min(100, score))

def estimate_net_profit(buy_price: float, ship_in: float, expected_close: float) -> Tuple[float, float]:
    """
    Modelo económico TIMELAB (placeholder conservador).
    - Revenue: martillo esperado + arbitraje envío
    - Fees: comisión Catawiki sobre martillo
    - Costs: compra + envío + packaging + misc
    - Taxes: % efectivo sobre beneficio positivo
    """
    cost_in = buy_price + ship_in + PACKAGING_EUR + MISC_EUR
    revenue_gross = expected_close + SHIP_ARBITRAGE_EUR
    fees = expected_close * CATWIKI_COMMISSION + PAYMENT_PROCESSING
    profit_before_tax = revenue_gross - fees - cost_in
    tax = max(0.0, profit_before_tax) * EFFECTIVE_TAX_RATE_ON_PROFIT
    net = profit_before_tax - tax
    roi = net / max(1.0, cost_in)
    return net, roi


# =========================
# TARGET LIST LOADER (JSON)
# =========================

def load_target_list_json(path: str = "target_list.json") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Create it next to scanner.py")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "targets" not in data or not isinstance(data["targets"], list):
        raise ValueError("target_list.json must contain a 'targets' array")
    return data

def keywords_from_json(model_keywords: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for kw in model_keywords or []:
        kw2 = norm(kw)
        if not kw2:
            continue
        if kw2 not in seen:
            seen.add(kw2)
            out.append(kw2)
    return out

def load_targets_from_json(path: str = "target_list.json") -> List[TargetModel]:
    """
    Convierte target_list.json -> TargetModel compatible con tu pipeline actual.
    - catwiki_close_med: media del rango expected_catawiki_eur
    - buy_max: max_buy_eur
    - fake_risk: risk
    """
    data = load_target_list_json(path)
    targets_json = data["targets"]

    targets: List[TargetModel] = []
    for tj in targets_json:
        brand = (tj.get("brand") or "").strip()
        risk = (tj.get("risk") or "medium").strip().lower()
        tier = (tj.get("tier") or "B").strip().upper()

        kws = keywords_from_json(tj.get("model_keywords") or [])
        if not brand or not kws:
            continue

        # Asegura que la marca está en keywords (para eBay_search y scoring)
        if norm(brand) not in kws:
            kws = [norm(brand)] + kws

        exp = tj.get("expected_catawiki_eur") or [0, 0]
        try:
            lo = float(exp[0]) if len(exp) > 0 else 0.0
            hi = float(exp[1]) if len(exp) > 1 else 0.0
        except Exception:
            lo, hi = 0.0, 0.0

        close_med = (lo + hi) / 2.0 if (lo > 0 and hi > 0) else max(lo, hi, 0.0)

        try:
            buy_max = float(tj.get("max_buy_eur") or 0.0)
        except Exception:
            buy_max = 0.0

        key = (tj.get("id") or f"{brand}_{kws[0]}").strip()

        targets.append(TargetModel(
            key=key,
            keywords=kws,
            refs=[],            # fase 2: meter refs por modelo
            tier=tier,
            fake_risk=risk,
            catwiki_close_med=close_med,
            buy_max=buy_max
        ))

    if not targets:
        raise ValueError("target_list.json loaded but produced 0 valid targets")

    return targets

def fallback_targets_min() -> List[TargetModel]:
    # Fallback mínimo por si el JSON falla (para no romper el bot)
    return [
        TargetModel("tag_f1", ["tag", "heuer", "formula", "1"], ["waz1110", "waz1010", "caz101"], "A", "medium", 850, 500),
        TargetModel("longines_conquest", ["longines", "conquest"], [], "A", "low", 900, 550),
        TargetModel("oris_aquis", ["oris", "aquis", "date"], [], "A", "low", 1000, 750),
    ]


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
            "LH_BIN": "1",  # Buy It Now
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

        time.sleep(1.0)  # throttle suave

    return out


# =========================
# CASHCONVERTERS (placeholder hook)
# =========================
def cashconverters_search(_model: TargetModel) -> List[Listing]:
    """
    PASO SIGUIENTE: aquí implementamos scraping de CashConverters (relojes de pulsera).
    Lo dejo como stub para integrar sin tocar el resto del pipeline.
    """
    return []


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

    # Load targets from JSON
    try:
        targets = load_targets_from_json("target_list.json")
        print(f"[OK] Loaded {len(targets)} targets from target_list.json")
    except Exception as e:
        print("[WARN] Using fallback targets due to target_list.json error:", str(e))
        targets = fallback_targets_min()

    # 1) Collect listings (eBay UE + (future) CashConverters)
    all_listings: List[Listing] = []

    # eBay
    for t in targets:
        for base in EBAY_SEARCH_BASES:
            try:
                all_listings.extend(ebay_search(t, base_url=base, max_pages=1))
            except Exception as e:
                print("EBAY ERROR:", base, t.key, str(e))

    # CashConverters (stub ahora; en el siguiente paso lo activamos)
    # for t in targets:
    #     try:
    #         all_listings.extend(cashconverters_search(t))
    #     except Exception as e:
    #         print("CC ERROR:", t.key, str(e))

    # 2) Score & filter
    candidates: List[Candidate] = []
    for li in all_listings:
        if not is_eu_location(li.location_text):
            continue

        for t in targets:
            ms = compute_match_score(li.title, t)
            if ms < 50:
                continue

            # Comprar por debajo de objetivo (flexible)
            if t.buy_max > 0 and li.price_eur > (t.buy_max * 1.15):
                continue

            expected_close = t.catwiki_close_med
            if expected_close <= 0:
                continue

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
        msg = (
            f"🕗 TIMELAB Morning Scan (eBay UE)\n"
            f"Timestamp: {now_utc}\n\n"
            f"No hay oportunidades que pasen los filtros:\n"
            f"- ≥ {MIN_NET_EUR}€ neto o ≥ {int(MIN_NET_ROI*100)}% ROI neto\n"
            f"- match ≥ {MIN_MATCH_SCORE}/100\n"
            f"- fake risk low/medium"
        )
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