import os
import re
import time
import json
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import requests

# =========================
# CONFIG (TIMELAB)
# =========================

# 🔧 MODO CALIBRACIÓN (temporal)
# Permite ver "casi oportunidades" para ajustar targets y márgenes reales

MIN_NET_EUR = float(os.getenv("MIN_NET_EUR", "-10"))
MIN_NET_ROI = float(os.getenv("MIN_NET_ROI", "-0.05"))
MIN_MATCH_SCORE = int(os.getenv("MIN_MATCH_SCORE", "45"))

ALLOW_FAKE_RISK = set(
    x.strip()
    for x in os.getenv("ALLOW_FAKE_RISK", "low,medium").split(",")
)
# Cost model (puedes ajustar luego para tu entramado real)
CATWIKI_COMMISSION = float(os.getenv("CATWIKI_COMMISSION", "0.125"))
PAYMENT_PROCESSING = float(os.getenv("PAYMENT_PROCESSING", "0.0"))
PACKAGING_EUR = float(os.getenv("PACKAGING_EUR", "5.0"))
MISC_EUR = float(os.getenv("MISC_EUR", "5.0"))
SHIP_ARBITRAGE_EUR = float(os.getenv("SHIP_ARBITRAGE_EUR", "35.0"))

EFFECTIVE_TAX_RATE_ON_PROFIT = float(os.getenv("EFFECTIVE_TAX_RATE_ON_PROFIT", "0.15"))

# eBay API
EBAY_CLIENT_ID = (os.getenv("EBAY_CLIENT_ID") or "").strip()
EBAY_CLIENT_SECRET = (os.getenv("EBAY_CLIENT_SECRET") or "").strip()
EBAY_MARKETPLACE_ID = (os.getenv("EBAY_MARKETPLACE_ID") or "EBAY_ES").strip()  # EBAY_ES, EBAY_FR, EBAY_DE, EBAY_IT...
EBAY_LIMIT = int(os.getenv("EBAY_LIMIT", "50"))  # items por target
EBAY_THROTTLE_S = float(os.getenv("EBAY_THROTTLE_S", "0.35"))

HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "25"))

EU_COUNTRIES = {
    "spain","españa","france","francia","germany","alemania","italy","italia",
    "portugal","belgium","belgica","netherlands","países bajos","austria","ireland",
    "finland","sweden","denmark","poland","czech","slovakia","slovenia","croatia",
    "hungary","romania","bulgaria","greece","luxembourg","latvia","lithuania",
    "estonia","cyprus","malta"
}

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
    query: str
    ebay_category_id: Optional[str] = None

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

def is_eu_location(loc: str) -> bool:
    l = norm(loc)
    return (not l) or any(c in l for c in EU_COUNTRIES)

def compute_match_score(title: str, target: TargetModel) -> int:
    """
    Score robusto:
    - Marca (primer keyword) pesa mucho
    - Modelo keywords pesan medio
    - refs suman algo
    - Penaliza palabras de falsificación
    """
    t = norm(title)
    if not t:
        return 0

    kws = [norm(k) for k in (target.keywords or []) if norm(k)]
    if not kws:
        return 0

    brand = kws[0]
    model_kws = kws[1:] if len(kws) > 1 else []

    score = 0

    # Marca (peso alto)
    if brand and brand in t:
        score += 45

    # Modelo keywords (peso medio)
    if model_kws:
        hits = sum(1 for kw in model_kws if kw and kw in t)
        score += int(45 * (hits / max(1, len(model_kws))))
    else:
        score += 15

    # refs (si las añades)
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
    """
    Modelo neto simplificado TIMELAB:
    - coste: compra + envío compra + packaging + misc
    - revenue: cierre esperado + arbitraje de envío
    - fees: comisión catawiki + payment processing
    - tax: sobre profit positivo
    """
    cost = buy + ship + PACKAGING_EUR + MISC_EUR
    revenue = close + SHIP_ARBITRAGE_EUR
    fees = close * CATWIKI_COMMISSION + PAYMENT_PROCESSING
    profit_bt = revenue - fees - cost
    tax = max(0.0, profit_bt) * EFFECTIVE_TAX_RATE_ON_PROFIT
    net = profit_bt - tax
    roi = net / max(1.0, cost)
    return net, roi

def now_utc() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

# =========================
# TARGET LIST
# =========================

def load_targets_from_json(path: str = "target_list.json") -> List[TargetModel]:
    """
    Soporta 2 formatos:
    A) {"targets":[...]} (tu formato viejo)
    B) [...] (lista simple) (formato nuevo)
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and isinstance(raw.get("targets"), list):
        data = raw["targets"]
    elif isinstance(raw, list):
        data = raw
    else:
        raise ValueError("target_list.json inválido: debe ser lista o dict con 'targets'.")

    targets: List[TargetModel] = []
    for t in data:
        brand = norm(t.get("brand", ""))
        mkws = t.get("model_keywords", []) or []
        query = (t.get("query") or "").strip()

        # Si viene en formato nuevo (brand/model/query), permitimos construir keywords
        model_name = (t.get("model") or "").strip()
        if not mkws and (brand and model_name):
            mkws = [brand, model_name]

        if not brand:
            continue

        kws = [norm(k) for k in mkws if norm(k)]
        if brand not in kws:
            kws.insert(0, brand)
        else:
            kws = [brand] + [k for k in kws if k != brand]

        # expected close
        close_med = 0.0
        if "expected_catawiki_eur" in t and isinstance(t["expected_catawiki_eur"], list):
            exp = t["expected_catawiki_eur"]
            lo = float(exp[0]) if len(exp) > 0 else 0.0
            hi = float(exp[1]) if len(exp) > 1 else 0.0
            close_med = (lo + hi) / 2.0 if (lo > 0 and hi > 0) else max(lo, hi, 0.0)
        else:
            # formato nuevo
            close_med = float(t.get("expected_hammer_eur", 0.0) or 0.0)

        buy_max = float(t.get("max_buy_eur", 0.0) or 0.0)

        # Si no te dan query explícito, lo construimos
        if not query:
            # algo razonable: brand + segundo keyword si existe
            query = " ".join([k for k in [brand, model_name] if k]).strip()
            if not query:
                query = " ".join(kws[:3]).strip()

        targets.append(TargetModel(
            key=str(t.get("id", f"{brand}_{model_name or (kws[1] if len(kws)>1 else 'model')}")),
            keywords=kws,
            refs=t.get("refs", []) or [],
            tier=str(t.get("tier", "B")),
            fake_risk=str(t.get("risk", "medium")).lower(),
            catwiki_close_med=close_med,
            buy_max=buy_max,
            query=query,
            ebay_category_id=(t.get("ebay_category_id") or None)
        ))

    if not targets:
        raise ValueError("target_list.json cargó pero no generó targets válidos")

    return targets

# =========================
# EBAY API (NO SCRAPING)
# =========================

def ebay_oauth_app_token() -> str:
    """
    Client Credentials Grant:
    POST https://api.ebay.com/identity/v1/oauth2/token
    """
    if not EBAY_CLIENT_ID or not EBAY_CLIENT_SECRET:
        raise RuntimeError("Faltan EBAY_CLIENT_ID / EBAY_CLIENT_SECRET (GitHub Secrets).")

    url = "https://api.ebay.com/identity/v1/oauth2/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "client_credentials",
        "scope": "https://api.ebay.com/oauth/api_scope"
    }

    r = requests.post(url, headers=headers, data=data, auth=(EBAY_CLIENT_ID, EBAY_CLIENT_SECRET), timeout=HTTP_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"OAuth error {r.status_code}: {r.text[:500]}")
    return r.json()["access_token"]

def ebay_search(token: str, query: str, category_id: Optional[str] = None, limit: int = 50) -> List[Listing]:
    """
    Browse API search (active listings):
    GET https://api.ebay.com/buy/browse/v1/item_summary/search?q=...
    """
    base = "https://api.ebay.com/buy/browse/v1/item_summary/search"
    headers = {
        "Authorization": f"Bearer {token}",
        "X-EBAY-C-MARKETPLACE-ID": EBAY_MARKETPLACE_ID,
        "Accept": "application/json"
    }

    params: Dict[str, str] = {
        "q": query,
        "limit": str(min(max(limit, 1), 200))
    }
    if category_id:
        params["category_ids"] = category_id

    r = requests.get(base, headers=headers, params=params, timeout=HTTP_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"Browse search error {r.status_code}: {r.text[:500]}")

    data = r.json()
    items = data.get("itemSummaries", []) or []
    out: List[Listing] = []

    for it in items:
        title = it.get("title") or ""
        url = it.get("itemWebUrl") or ""

        # Price
        price = it.get("price") or {}
        price_val = price.get("value")
        currency = (price.get("currency") or "").upper()
        try:
            price_eur = float(price_val) if price_val is not None else None
        except Exception:
            price_eur = None
        if price_eur is None:
            continue

        # Shipping (si no viene, 0.0)
        ship_eur = 0.0
        ship = it.get("shippingOptions") or []
        # La Browse API no siempre trae shipping cost en search; lo dejamos 0.
        # Si quieres estimar shipping real, luego añadimos "getItem" por ID para top candidatos.

        # Location
        loc = ""
        item_loc = it.get("itemLocation") or {}
        cc = item_loc.get("country") or ""
        city = item_loc.get("city") or ""
        if city and cc:
            loc = f"{city}, {cc}"
        elif cc:
            loc = cc

        # Nota: currency puede no ser EUR en marketplaces. Para ES suele ser EUR.
        # Si quieres soporte multi-moneda, lo añadimos después.
        if currency and currency != "EUR" and EBAY_MARKETPLACE_ID in ("EBAY_ES","EBAY_FR","EBAY_DE","EBAY_IT"):
            # Aún así puede venir EUR; si no es EUR, por ahora lo dejamos pasar pero marcado
            pass

        out.append(Listing(
            source="ebay",
            country_site=EBAY_MARKETPLACE_ID,
            title=title,
            price_eur=price_eur,
            shipping_eur=ship_eur,
            url=url,
            location_text=loc
        ))

    return out

# =========================
# TELEGRAM
# =========================

def tg_send(msg: str) -> None:
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    if not token or not chat:
        print("[WARN] Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID")
        print(msg)
        return

    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": msg, "disable_web_page_preview": True},
            timeout=HTTP_TIMEOUT
        )
        print("Telegram:", r.status_code, r.text[:300])
    except Exception as e:
        print("Telegram error:", str(e))

# =========================
# MAIN
# =========================

def main():
    now = now_utc()

    # Load targets
    try:
        targets = load_targets_from_json("target_list.json")
    except Exception as e:
        tg_send(f"🕗 TIMELAB Morning Scan\n{now}\n\n❌ Error leyendo target_list.json:\n{str(e)[:800]}")
        raise

    # Get eBay token
    try:
        token = ebay_oauth_app_token()
    except Exception as e:
        tg_send(f"🕗 TIMELAB Morning Scan\n{now}\n\n❌ Error OAuth eBay:\n{str(e)[:800]}")
        raise

    # 1) Collect listings from eBay API
    listings: List[Listing] = []
    counts = {"ebay": 0}

    for t in targets:
        try:
            got = ebay_search(token, query=t.query, category_id=t.ebay_category_id, limit=EBAY_LIMIT)
            listings.extend(got)
            counts["ebay"] += len(got)
            time.sleep(EBAY_THROTTLE_S)
        except Exception as e:
            print("EBAY ERROR:", t.key, str(e)[:300])

    # If no listings, alert clearly
    if not listings:
        tg_send(
            f"🕗 TIMELAB Morning Scan\n{now}\n\n"
            f"⚠️ 0 listings recogidos desde eBay API.\n"
            f"Esto suele indicar: query demasiado restrictiva, marketplace incorrecto ({EBAY_MARKETPLACE_ID}), o error de token.\n"
        )
        return

    # 2) Build candidates: asigna a cada listing su mejor target
    candidates: List[Candidate] = []
    raw_scored: List[Tuple[int, Listing, TargetModel, float, float]] = []  # ms, li, t, net, roi

    for li in listings:
        # Filtro localización (suave)
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

        expected_close = best_t.catwiki_close_med
        if expected_close <= 0:
            expected_close = max(li.price_eur * 1.8, li.price_eur + 150)  # fallback debug

        net, roi = estimate_net_profit(li.price_eur, li.shipping_eur, expected_close)
        raw_scored.append((best_ms, li, best_t, net, roi))

        # filtro de compra (aflojado para test)
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
        raw_scored.sort(key=lambda x: (x[0], x[3]), reverse=True)
        raw_top = raw_scored[:5]

        lines = [
            f"🕗 TIMELAB Morning Scan (eBay API {EBAY_MARKETPLACE_ID})\n{now}",
            "",
            f"Recolectado: eBay={counts['ebay']}",
            f"Filtros: net≥{MIN_NET_EUR:.0f}€ o ROI≥{int(MIN_NET_ROI*100)}% | match≥{MIN_MATCH_SCORE} | fake: {','.join(sorted(ALLOW_FAKE_RISK))}",
            "",
            "❌ Sin oportunidades que pasen filtros.",
            "",
            "🧪 SMOKE TEST (Top RAW por match, aunque no cumplan margen):",
            ""
        ]

        for i, (ms, li, t, net, roi) in enumerate(raw_top, 1):
            lines.append(
                f"{i}) [ebay] {li.title}\n"
                f"   💶 {li.price_eur:.0f}€ | Neto est. {net:.0f}€ | ROI {int(roi*100)}% | Match {ms}\n"
                f"   🔗 {li.url}\n"
            )

        tg_send("\n".join(lines))
        return

    msg = [
        f"🕗 TIMELAB Morning Scan — TOP {len(top)} (eBay API {EBAY_MARKETPLACE_ID})",
        f"{now}",
        "",
        f"Recolectado: eBay={counts['ebay']}",
        f"Filtros: net≥{MIN_NET_EUR:.0f}€ o ROI≥{int(MIN_NET_ROI*100)}% | match≥{MIN_MATCH_SCORE} | fake: {','.join(sorted(ALLOW_FAKE_RISK))}",
        ""
    ]

    for i, c in enumerate(top, 1):
        li = c.listing
        msg.append(
            f"{i}) [ebay] {li.title}\n"
            f"   💶 Compra: {li.price_eur:.0f}€ | 🎯 Cierre est.: {c.expected_close:.0f}€\n"
            f"   ✅ Neto est.: {c.net_profit:.0f}€ | ROI: {int(c.net_roi*100)}% | Match: {c.match_score}\n"
            f"   🔗 {li.url}\n"
        )

    tg_send("\n".join(msg))


if __name__ == "__main__":
    main()