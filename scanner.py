import os
import re
import time
import json
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import requests


# =========================
# CONFIG (TIMELAB)
# =========================

MIN_NET_EUR = float(os.getenv("MIN_NET_EUR", "20"))
MIN_NET_ROI = float(os.getenv("MIN_NET_ROI", "0.05"))
MIN_MATCH_SCORE = int(os.getenv("MIN_MATCH_SCORE", "50"))
ALLOW_FAKE_RISK = set(x.strip().lower() for x in os.getenv("ALLOW_FAKE_RISK", "low,medium").split(","))

BUY_MAX_MULT = float(os.getenv("BUY_MAX_MULT", "1.25"))  # producción 1.25–1.5 | calibración 10

CATWIKI_COMMISSION = float(os.getenv("CATWIKI_COMMISSION", "0.125"))
PAYMENT_PROCESSING = float(os.getenv("PAYMENT_PROCESSING", "0.0"))
PACKAGING_EUR = float(os.getenv("PACKAGING_EUR", "5.0"))
MISC_EUR = float(os.getenv("MISC_EUR", "5.0"))
SHIP_ARBITRAGE_EUR = float(os.getenv("SHIP_ARBITRAGE_EUR", "35.0"))
EFFECTIVE_TAX_RATE_ON_PROFIT = float(os.getenv("EFFECTIVE_TAX_RATE_ON_PROFIT", "0.15"))

HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "25"))

# eBay API
EBAY_CLIENT_ID = (os.getenv("EBAY_CLIENT_ID") or "").strip()
EBAY_CLIENT_SECRET = (os.getenv("EBAY_CLIENT_SECRET") or "").strip()
EBAY_MARKETPLACE_ID = (os.getenv("EBAY_MARKETPLACE_ID") or "EBAY_ES").strip()
EBAY_LIMIT = int(os.getenv("EBAY_LIMIT", "50"))
EBAY_THROTTLE_S = float(os.getenv("EBAY_THROTTLE_S", "0.35"))


# =========================
# EU HELPERS (IMPORTANT)
# =========================

EU_ISO2 = {
    "ES","FR","DE","IT","PT","BE","NL","LU","IE","AT","FI","SE","DK","PL","CZ","SK","SI","HR",
    "HU","RO","BG","GR","CY","MT","LV","LT","EE"
}

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def is_eu_location(loc: str) -> bool:
    """
    eBay Browse API suele dar itemLocation.country como ISO2 (p.ej. 'DE')
    y lo montamos como 'city, DE'. Reconocemos esos códigos y también texto.
    """
    if not loc:
        return True

    raw = loc.strip()
    # intenta sacar ISO2 al final: "Madrid, ES"
    m = re.search(r"\b([A-Z]{2})\b\s*$", raw)
    if m:
        return m.group(1).upper() in EU_ISO2

    l = norm(raw)
    # fallback muy permisivo: si contiene nombres de países típicos
    common = ["spain","españa","france","francia","germany","alemania","italy","italia","portugal",
              "belgium","bélgica","netherlands","países bajos","austria","ireland","finland","sweden",
              "denmark","poland","czech","slovakia","slovenia","croatia","hungary","romania","bulgaria",
              "greece","luxembourg","latvia","lithuania","estonia","cyprus","malta"]
    return any(c in l for c in common)


# =========================
# DATA STRUCTURES
# =========================

@dataclass
class TargetModel:
    key: str
    keywords: List[str]
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
# MATCH + PROFIT
# =========================

def compute_match_score(title: str, target: TargetModel) -> int:
    t = norm(title)
    if not t:
        return 0

    kws = [norm(k) for k in (target.keywords or []) if norm(k)]
    if not kws:
        return 0

    brand = kws[0]
    model_kws = kws[1:] if len(kws) > 1 else []

    score = 0
    if brand and brand in t:
        score += 45

    if model_kws:
        hits = sum(1 for kw in model_kws if kw and kw in t)
        score += int(45 * (hits / max(1, len(model_kws))))
    else:
        score += 15

    if target.refs:
        rhits = sum(1 for r in target.refs if norm(r) in t)
        score += int(10 * (rhits / max(1, len(target.refs))))
    else:
        score += 5

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

def now_utc() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")


# =========================
# TARGET LIST
# =========================

def load_targets_from_json(path: str = "target_list.json") -> List[TargetModel]:
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
        if not brand or not mkws:
            continue

        kws = [norm(k) for k in mkws if norm(k)]
        if brand not in kws:
            kws.insert(0, brand)
        else:
            kws = [brand] + [k for k in kws if k != brand]

        exp = t.get("expected_catawiki_eur", [0, 0])
        lo = float(exp[0]) if isinstance(exp, list) and len(exp) > 0 else 0.0
        hi = float(exp[1]) if isinstance(exp, list) and len(exp) > 1 else 0.0
        close_med = (lo + hi) / 2.0 if (lo > 0 and hi > 0) else max(lo, hi, 0.0)

        buy_max = float(t.get("max_buy_eur", 0.0) or 0.0)

        # Query para eBay: marca + keywords principales (sin duplicar)
        query = " ".join(kws[:4]).strip()

        targets.append(TargetModel(
            key=str(t.get("id", f"{brand}_{kws[1] if len(kws)>1 else 'model'}")),
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
# EBAY API
# =========================

def ebay_oauth_app_token() -> str:
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
    base = "https://api.ebay.com/buy/browse/v1/item_summary/search"
    headers = {
        "Authorization": f"Bearer {token}",
        "X-EBAY-C-MARKETPLACE-ID": EBAY_MARKETPLACE_ID,
        "Accept": "application/json"
    }

    params: Dict[str, str] = {"q": query, "limit": str(min(max(limit, 1), 200))}
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
        price = it.get("price") or {}
        currency = (price.get("currency") or "").upper()
        try:
            price_eur = float(price.get("value"))
        except Exception:
            continue

        # En search, shipping suele no venir: lo dejamos 0
        ship_eur = 0.0

        item_loc = it.get("itemLocation") or {}
        cc = (item_loc.get("country") or "").upper()
        city = (item_loc.get("city") or "")
        loc = f"{city}, {cc}".strip(", ") if (city or cc) else ""

        # Si viene una moneda rara, seguimos (más adelante podemos convertir)
        _ = currency

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

    # IMPORTANTE: si faltan, fallamos el job (así lo ves en Actions)
    if not token or not chat:
        raise RuntimeError("Faltan TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID en env (Secrets no inyectados).")

    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat, "text": msg, "disable_web_page_preview": True},
        timeout=HTTP_TIMEOUT
    )
    if r.status_code != 200:
        raise RuntimeError(f"Telegram error {r.status_code}: {r.text[:500]}")


# =========================
# MAIN
# =========================

def main():
    now = now_utc()

    targets = load_targets_from_json("target_list.json")
    token = ebay_oauth_app_token()

    listings: List[Listing] = []
    counts = {"ebay": 0}

    for t in targets:
        got = ebay_search(token, query=t.query, category_id=t.ebay_category_id, limit=EBAY_LIMIT)
        listings.extend(got)
        counts["ebay"] += len(got)
        time.sleep(EBAY_THROTTLE_S)

    if not listings:
        tg_send(
            f"🕗 TIMELAB Morning Scan (eBay API {EBAY_MARKETPLACE_ID})\n{now}\n\n"
            f"⚠️ 0 listings recogidos desde eBay API.\n"
            f"Revisa: marketplace={EBAY_MARKETPLACE_ID} | queries | token.\n"
        )
        return

    candidates: List[Candidate] = []
    raw_scored: List[Tuple[int, Listing, TargetModel, float, float]] = []

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

        expected_close = best_t.catwiki_close_med if best_t.catwiki_close_med > 0 else max(li.price_eur * 1.8, li.price_eur + 150)
        net, roi = estimate_net_profit(li.price_eur, li.shipping_eur, expected_close)
        raw_scored.append((best_ms, li, best_t, net, roi))

        # buy_max controlable
        if best_t.buy_max > 0 and li.price_eur > (best_t.buy_max * BUY_MAX_MULT):
            continue

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

    candidates.sort(key=lambda c: (c.net_profit, c.match_score), reverse=True)
    top = candidates[:10]

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

        for i, (ms, li, _, net, roi) in enumerate(raw_top, 1):
            lines.append(
                f"{i}) [ebay] {li.title}\n"
                f"   💶 {li.price_eur:.0f}€ | Neto est. {net:.0f}€ | ROI {int(roi*100)}% | Match {ms}\n"
                f"   📍 {li.location_text}\n"
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
            f"   📍 {li.location_text}\n"
            f"   🔗 {li.url}\n"
        )

    tg_send("\n".join(msg))


if __name__ == "__main__":
    main()