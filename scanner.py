import os
import re
import time
import json
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import requests
from bs4 import BeautifulSoup

# =========================
# CONFIG (TIMELAB)
# =========================

MIN_NET_EUR = 20
MIN_NET_ROI = 0.05
MIN_MATCH_SCORE = 50
ALLOW_FAKE_RISK = {"low", "medium"}  # nunca "high"

CATWIKI_COMMISSION = 0.125
PAYMENT_PROCESSING = 0.0
PACKAGING_EUR = 5.0
MISC_EUR = 5.0
SHIP_ARBITRAGE_EUR = 35.0

EFFECTIVE_TAX_RATE_ON_PROFIT = 0.15

EBAY_SEARCH_BASES = [
    "https://www.ebay.es/sch/i.html",
    "https://www.ebay.fr/sch/i.html",
    "https://www.ebay.de/sch/i.html",
    "https://www.ebay.it/sch/i.html",
]

EU_COUNTRIES = {
    "spain","españa","france","francia","germany","alemania","italy","italia",
    "portugal","belgium","belgica","netherlands","países bajos","austria","ireland",
    "finland","sweden","denmark","poland","czech","slovakia","slovenia","croatia",
    "hungary","romania","bulgaria","greece","luxembourg","latvia","lithuania",
    "estonia","cyprus","malta"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TIMELABScanner/1.0)",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}

HTTP_TIMEOUT = 25
THROTTLE_S = 0.8

# =========================
# DATA STRUCTURES
# =========================

@dataclass
class TargetModel:
    key: str
    keywords: List[str]        # incluye marca + modelo keywords
    refs: List[str]
    tier: str
    fake_risk: str
    catwiki_close_med: float
    buy_max: float

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
    expected_close: float
    net_profit: float
    net_roi: float

# =========================
# HELPERS
# =========================

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def extract_price(text: str) -> Optional[float]:
    """Extrae el primer número razonable de un string tipo '1.234,56 €'."""
    if not text:
        return None
    t = text.replace("\xa0", " ").strip()
    # Quita símbolos comunes, deja números, punto, coma
    t = re.sub(r"[^\d\.,]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return None
    # Normaliza 1.234,56 -> 1234.56
    # Estrategia: si hay coma y punto, asumimos punto miles y coma decimal
    if "," in t and "." in t:
        t = t.replace(".", "").replace(",", ".")
    else:
        # si solo coma, puede ser decimal
        t = t.replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)", t)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def is_eu_location(loc: str) -> bool:
    l = norm(loc)
    return (not l) or any(c in l for c in EU_COUNTRIES)

def compute_match_score(title: str, target: TargetModel) -> int:
    """
    Score más robusto:
    - Marca (primer keyword normalmente) pesa mucho
    - Modelo keywords pesan medio
    - Penaliza palabras de falsificación
    """
    t = norm(title)
    if not t:
        return 0

    kws = [norm(k) for k in (target.keywords or []) if norm(k)]
    if not kws:
        return 0

    brand = kws[0]  # por construcción insertamos brand al inicio
    model_kws = kws[1:] if len(kws) > 1 else []

    score = 0

    # Marca (peso alto)
    if brand and brand in t:
        score += 45
    else:
        score += 0

    # Modelo keywords (peso medio)
    if model_kws:
        hits = sum(1 for kw in model_kws if kw and kw in t)
        score += int(45 * (hits / max(1, len(model_kws))))
    else:
        score += 15

    # refs (si las añades en el futuro)
    if target.refs:
        rhits = sum(1 for r in target.refs if norm(r) in t)
        score += int(10 * (rhits / max(1, len(target.refs))))
    else:
        score += 5

    # Penalizaciones
    suspicious = ["replica","copy","fake","imitacion","imitación","imitation"]
    if any(w in t for w in suspicious):
        score -= 70

    return max(0, min(100, score))

def estimate_net_profit(buy: float, ship: float, close: float) -> Tuple[float, float]:
    cost = buy + ship + PACKAGING_EUR + MISC_EUR
    revenue = close + SHIP_ARBITRAGE_EUR
    fees = close * CATWIKI_COMMISSION + PAYMENT_PROCESSING
    profit_bt = revenue - fees - cost
    tax = max(0.0, profit_bt) * EFFECTIVE_TAX_RATE_ON_PROFIT
    net = profit_bt - tax
    roi = net / max(1.0, cost)
    return net, roi

# =========================
# TARGET LIST
# =========================

def load_targets_from_json(path: str = "target_list.json") -> List[TargetModel]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data = raw.get("targets", [])
    if not isinstance(data, list) or not data:
        raise ValueError("target_list.json: 'targets' vacío o inválido")

    targets: List[TargetModel] = []
    for t in data:
        brand = norm(t.get("brand", ""))
        mkws = t.get("model_keywords", []) or []
        if not brand or not mkws:
            continue

        kws = [norm(k) for k in mkws if norm(k)]
        if brand not in kws:
            kws.insert(0, brand)
        else:
            # asegúrate brand al inicio
            kws = [brand] + [k for k in kws if k != brand]

        exp = t.get("expected_catawiki_eur", [0, 0])
        lo = float(exp[0]) if isinstance(exp, list) and len(exp) > 0 else 0.0
        hi = float(exp[1]) if isinstance(exp, list) and len(exp) > 1 else 0.0
        close_med = (lo + hi) / 2.0 if (lo > 0 and hi > 0) else max(lo, hi, 0.0)

        buy_max = float(t.get("max_buy_eur", 0.0) or 0.0)

        targets.append(TargetModel(
            key=str(t.get("id", f"{brand}_{kws[1] if len(kws)>1 else 'model'}")),
            keywords=kws,
            refs=[],
            tier=str(t.get("tier", "B")),
            fake_risk=str(t.get("risk", "medium")).lower(),
            catwiki_close_med=close_med,
            buy_max=buy_max
        ))

    if not targets:
        raise ValueError("target_list.json cargó pero no generó targets válidos")

    return targets

# =========================
# EBAY
# =========================

def ebay_search(model: TargetModel, base_url: str, max_items: int = 25) -> List[Listing]:
    out: List[Listing] = []

    params = {
        "_nkw": " ".join(model.keywords),
        "_sop": "10",     # Newly listed
        "LH_BIN": "1",    # Buy It Now
        "_ipg": "50",     # items per page
    }

    r = requests.get(base_url, params=params, headers=HEADERS, timeout=HTTP_TIMEOUT)
    if r.status_code != 200:
        return out

    soup = BeautifulSoup(r.text, "lxml")
    for it in soup.select("li.s-item"):
        title_el = it.select_one(".s-item__title")
        price_el = it.select_one(".s-item__price")
        link_el = it.select_one("a.s-item__link")
        ship_el = it.select_one(".s-item__shipping, .s-item__logisticsCost")
        loc_el = it.select_one(".s-item__location")

        if not title_el or not price_el or not link_el:
            continue

        title = title_el.get_text(" ", strip=True)
        if not title or title.lower() in {"shop on ebay", "explore more options"}:
            continue

        price = extract_price(price_el.get_text(" ", strip=True))
        if price is None:
            continue

        shipping = 0.0
        if ship_el:
            st = ship_el.get_text(" ", strip=True).lower()
            if ("free" in st) or ("gratis" in st):
                shipping = 0.0
            else:
                sp = extract_price(st)
                shipping = float(sp) if sp is not None else 0.0

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

        if len(out) >= max_items:
            break

    return out

# =========================
# CASHCONVERTERS
# =========================

def _cc_fetch(url: str, params: Dict[str, str]) -> Optional[BeautifulSoup]:
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            return None
        return BeautifulSoup(r.text, "lxml")
    except Exception:
        return None

def cashconverters_search(model: TargetModel, max_pages: int = 2) -> List[Listing]:
    """
    Implementación tolerante:
    - Intenta buscar en categoria relojes + query
    - Si no hay cards, intenta buscador general /buscar
    - Selectores alternativos
    """
    out: List[Listing] = []
    query = " ".join(model.keywords).strip()
    if not query:
        return out

    base_cat = "https://www.cashconverters.es/es/es/comprar/relojes/relojes-de-pulsera"
    base_search = "https://www.cashconverters.es/es/es/buscar"

    def parse_cards(soup: BeautifulSoup) -> List[Listing]:
        found: List[Listing] = []

        # Selectores típicos (cambian a veces)
        cards = soup.select("article.product-card")
        if not cards:
            cards = soup.select("article[class*='product']")  # fallback

        for c in cards:
            # title
            t_el = c.select_one(".product-card__title") or c.select_one("[class*='title']")
            # price
            p_el = c.select_one(".product-card__price") or c.select_one("[class*='price']")
            # link
            a_el = c.select_one("a.product-card__link") or c.select_one("a[href*='/es/es/comprar/']") or c.select_one("a[href]")
            # store/location (opcional)
            s_el = c.select_one(".product-card__store") or c.select_one("[class*='store']")

            if not t_el or not p_el or not a_el:
                continue

            title = t_el.get_text(" ", strip=True)
            price = extract_price(p_el.get_text(" ", strip=True))
            if not title or price is None:
                continue

            href = a_el.get("href", "")
            if href.startswith("/"):
                href = "https://www.cashconverters.es" + href

            loc = s_el.get_text(" ", strip=True) if s_el else ""

            found.append(Listing(
                source="cashconverters",
                country_site="cashconverters.es",
                title=title,
                price_eur=float(price),
                shipping_eur=0.0,
                url=href,
                location_text=loc
            ))

        return found

    # 1) Intento por categoría
    for page in range(1, max_pages + 1):
        soup = _cc_fetch(base_cat, {"q": query, "page": str(page)})
        if not soup:
            break
        page_items = parse_cards(soup)
        if not page_items:
            break
        out.extend(page_items)
        time.sleep(THROTTLE_S)

    # 2) Fallback por buscador general si no salió nada
    if not out:
        for page in range(1, max_pages + 1):
            soup = _cc_fetch(base_search, {"q": query, "page": str(page)})
            if not soup:
                break
            page_items = parse_cards(soup)
            if not page_items:
                break
            out.extend(page_items)
            time.sleep(THROTTLE_S)

    return out

# =========================
# TELEGRAM
# =========================

def tg_send(msg: str) -> None:
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    if not token or not chat:
        print("[WARN] Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID")
        return

    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": msg, "disable_web_page_preview": True},
            timeout=HTTP_TIMEOUT
        )
        # Para debug en Actions
        print("Telegram:", r.status_code, r.text[:300])
    except Exception as e:
        print("Telegram error:", str(e))

# =========================
# MAIN
# =========================

def main():
    now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    targets = load_targets_from_json()

    # 1) Collect listings
    listings: List[Listing] = []
    counts = {"ebay": 0, "cashconverters": 0}

    for t in targets:
        # eBay
        for base in EBAY_SEARCH_BASES:
            try:
                got = ebay_search(t, base_url=base, max_items=20)
                listings.extend(got)
                counts["ebay"] += len(got)
                time.sleep(THROTTLE_S)
            except Exception as e:
                print("EBAY ERROR:", base, t.key, str(e))

        # CashConverters
        try:
            got = cashconverters_search(t, max_pages=2)
            listings.extend(got)
            counts["cashconverters"] += len(got)
        except Exception as e:
            print("CC ERROR:", t.key, str(e))

    # Si no hay listings, avisa claro (problema scraping/red)
    if not listings:
        tg_send(
            f"🕗 TIMELAB Morning Scan\n{now}\n\n"
            f"⚠️ 0 listings recogidos.\n"
            f"Esto indica que el scraping no está trayendo resultados (cambio HTML / bloqueo / query).\n"
        )
        return

    # 2) Build candidates: asigna a cada listing su mejor target (evita duplicados locos)
    candidates: List[Candidate] = []
    raw_scored: List[Tuple[int, Listing, TargetModel, float, float]] = []  # ms, li, t, net, roi

    for li in listings:
        if not is_eu_location(li.location_text):
            continue

        best_ms = -1
        best_t: Optional[TargetModel] = None

        for t in targets:
            ms = compute_match_score(li.title, t)
            if ms > best_ms:
                best_ms = ms
                best_t = t

        if not best_t:
            continue

        # Benchmark close: si viene 0, hacemos fallback suave para no matar el bot en fase test
        expected_close = best_t.catwiki_close_med
        if expected_close <= 0:
            expected_close = max(li.price_eur * 1.8, li.price_eur + 150)  # fallback debug

        net, roi = estimate_net_profit(li.price_eur, li.shipping_eur, expected_close)
        raw_scored.append((best_ms, li, best_t, net, roi))

        # filtro de compra (lo aflojamos un poco para test)
        if best_t.buy_max > 0 and li.price_eur > (best_t.buy_max * 1.25):
            continue

        # filtros duros
        if best_ms < MIN_MATCH_SCORE:
            continue
        if best_t.fake_risk not in ALLOW_FAKE_RISK:
            continue
        if not (net >= MIN_NET_EUR or roi >= MIN_NET_ROI):
            continue

        candidates.append(Candidate(
            listing=li,
            target=best_t,
            match_score=best_ms,
            expected_close=expected_close,
            net_profit=net,
            net_roi=roi
        ))

    # 3) Rank
    candidates.sort(key=lambda c: (c.net_profit, c.match_score), reverse=True)
    top = candidates[:10]

    # 4) Telegram report
    if not top:
        # SMOKE TEST: manda top RAW por match para probar que hay anuncios reales
        raw_scored.sort(key=lambda x: (x[0], x[3]), reverse=True)
        raw_top = raw_scored[:5]

        lines = [
            f"🕗 TIMELAB Morning Scan\n{now}",
            "",
            f"Recolectado: eBay={counts['ebay']} | CashConverters={counts['cashconverters']}",
            f"Filtros actuales: net≥{MIN_NET_EUR}€ o ROI≥{int(MIN_NET_ROI*100)}% | match≥{MIN_MATCH_SCORE} | fake: low/medium",
            "",
            "❌ Sin oportunidades que pasen filtros.",
            "",
            "🧪 SMOKE TEST (Top RAW por match, aunque no cumplan margen):",
            ""
        ]

        for i, (ms, li, t, net, roi) in enumerate(raw_top, 1):
            lines.append(
                f"{i}) [{li.source}] {li.title}\n"
                f"   💶 {li.price_eur:.0f}€ | Neto est. {net:.0f}€ | ROI {int(roi*100)}% | Match {ms}\n"
                f"   🔗 {li.url}\n"
            )

        tg_send("\n".join(lines))
        return

    msg = [
        f"🕗 TIMELAB Morning Scan — TOP {len(top)}",
        f"{now}",
        "",
        f"Recolectado: eBay={counts['ebay']} | CashConverters={counts['cashconverters']}",
        f"Filtros: net≥{MIN_NET_EUR}€ o ROI≥{int(MIN_NET_ROI*100)}% | match≥{MIN_MATCH_SCORE} | fake: low/medium",
        ""
    ]

    for i, c in enumerate(top, 1):
        li = c.listing
        t = c.target
        msg.append(
            f"{i}) [{li.source}] {li.title}\n"
            f"   💶 Compra: {li.price_eur:.0f}€ seen | 🎯 Cierre est.: {c.expected_close:.0f}€\n"
            f"   ✅ Neto est.: {c.net_profit:.0f}€ | ROI: {int(c.net_roi*100)}% | Match: {c.match_score}\n"
            f"   🔗 {li.url}\n"
        )

    tg_send("\n".join(msg))

if __name__ == "__main__":
    main()