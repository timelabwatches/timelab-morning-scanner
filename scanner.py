import os
import re
import time
import json
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Set
from urllib.parse import quote

import requests


# =========================
# CONFIG (TIMELAB)
# =========================

MIN_NET_EUR = float(os.getenv("MIN_NET_EUR", "20"))
MIN_NET_ROI = float(os.getenv("MIN_NET_ROI", "0.05"))
MIN_MATCH_SCORE = int(os.getenv("MIN_MATCH_SCORE", "50"))
ALLOW_FAKE_RISK = set(x.strip().lower() for x in os.getenv("ALLOW_FAKE_RISK", "low,medium").split(","))

BUY_MAX_MULT = float(os.getenv("BUY_MAX_MULT", "1.25"))

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

EBAY_DEFAULT_CATEGORY_ID = (os.getenv("EBAY_DEFAULT_CATEGORY_ID") or "31387").strip()

DETAIL_FETCH_N = int(os.getenv("DETAIL_FETCH_N", "35"))
EBAY_DETAIL_THROTTLE_S = float(os.getenv("EBAY_DETAIL_THROTTLE_S", "0.20"))

# ✅ Categorías permitidas (default: solo relojes)
EBAY_ALLOWED_CATEGORY_IDS = set(
    x.strip() for x in (os.getenv("EBAY_ALLOWED_CATEGORY_IDS", EBAY_DEFAULT_CATEGORY_ID) or "").split(",") if x.strip()
)

# Telegram
TG_MAX_LEN = int(os.getenv("TG_MAX_LEN", "3500"))


# =========================
# NORMALIZATION
# =========================

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def norm_tokens(items: List[str]) -> List[str]:
    out = []
    for x in items or []:
        nx = norm(x)
        if nx:
            out.append(nx)
    return out


# =========================
# URL CLEANING
# =========================

def clean_url(url: str) -> str:
    if not url:
        return ""
    m = re.search(r"/itm/(\d+)", url)
    if m:
        return f"https://www.ebay.es/itm/{m.group(1)}"
    return url.split("?", 1)[0]


# =========================
# EU HELPERS
# =========================

EU_ISO2 = {
    "ES","FR","DE","IT","PT","BE","NL","LU","IE","AT","FI","SE","DK","PL","CZ","SK","SI","HR",
    "HU","RO","BG","GR","CY","MT","LV","LT","EE"
}

def is_eu_location(loc: str) -> bool:
    if not loc:
        return True
    raw = loc.strip()
    m = re.search(r"\b([A-Z]{2})\b\s*$", raw)
    if m:
        return m.group(1).upper() in EU_ISO2
    l = norm(raw)
    common = ["spain","españa","france","francia","germany","alemania","italy","italia","portugal",
              "belgium","bélgica","netherlands","países bajos","austria","ireland","finland","sweden",
              "denmark","poland","czech","slovakia","slovenia","croatia","hungary","romania","bulgaria",
              "greece","luxembourg","latvia","lithuania","estonia","cyprus","malta"]
    return any(c in l for c in common)


# =========================
# GLOBAL FILTERS
# =========================

WATCH_LIKELY_TERMS = {
    "watch", "orologio", "montre", "uhr", "reloj",
    "automatic", "automatique", "automatik",
    "chronograph", "cronografo", "chronographe",
    "cal.", "calibre", "caliber"
}

GLOBAL_PARTS_TERMS = {
    "crown", "corona",
    "dial", "cadran",
    "case", "cassa", "boitier", "boîtier",
    "hands", "aiguilles",
    "bracelet", "bracciale", "armband", "armis",
    "movement", "movimiento", "mouvement", "werk",
    "box", "scatola",
    "papers", "documenti", "paperwork",
    "parts", "spares", "spare",
    "ricambi", "repuestos", "pieza", "piezas",
    "lot of", "bundle"
}

# ✅ NUEVO: rechazo duro de “incompleto / solo piezas” (aunque ponga automatic)
INCOMPLETE_HARD_TERMS = {
    # EN
    "case only", "watch case", "empty case", "no movement", "without movement", "movement missing",
    "dial only", "hands only", "bracelet only",
    "for parts", "parts only", "spares only",
    # ES
    "solo caja", "caja sola", "sin movimiento", "sin maquinaria", "sin máquina", "sin mecanismo",
    "solo armis", "solo correa", "solo brazalete", "para piezas", "para repuestos",
    # IT
    "cassa per", "cassa originale per", "cassa in acciaio per", "solo cassa",
    "senza movimento", "manca movimento", "solo bracciale",
    "ricambi", "riparazione", "da riparare",
    # FR/DE
    "sans mouvement", "pour pièces", "zum basteln", "ohne werk"
}

GLOBAL_HARD_BAD_TERMS = {
    "broken", "not working", "doesn't work", "does not work",
    "defect", "defective",
    "as is", "untested", "not tested",
    "no funciona", "averiado", "averiada", "sin funcionar",
    "non funziona", "guasto",
    "ne fonctionne pas",
    "defekt", "funktioniert nicht"
}

GLOBAL_BOOST_TERMS = {
    "nos", "new old stock", "mint", "full set",
    "serviced", "service", "revised", "revisionato", "revisado", "revisión",
    "working", "works", "runs", "tested", "fonctionne", "funziona", "läuft",
    "caja y papeles", "box and papers"
}

GLOBAL_BAD_TERMS = {
    "scratches", "scratch", "heavily used", "worn",
    "rust", "corrosion",
    "cracked", "broken glass", "glass cracked",
    "missing", "no crown", "without strap", "no strap",
    "read the description", "see description", "please read",
    "balance ok", "balance wheel ok"
}

AUTO_CONTRADICTIONS = {
    "quartz", "battery", "pile",
    "manual", "hand-wound", "hand wound", "handwound",
    "carica manuale", "a carica manuale", "remontage manuel", "handaufzug"
}

def title_has_any(t: str, terms: Set[str]) -> bool:
    return any(x in t for x in terms if x)

def text_is_likely_watch(t: str) -> bool:
    return title_has_any(t, WATCH_LIKELY_TERMS)

def has_incomplete_hard_terms(text: str) -> bool:
    t = norm(text)
    return any(x in t for x in INCOMPLETE_HARD_TERMS)

def global_noise_reject(text: str) -> Optional[str]:
    t = norm(text)
    if not t:
        return "empty_text"
    # ✅ primero: incompletos / piezas
    if has_incomplete_hard_terms(t):
        return "incomplete_parts"
    # ✅ luego: averiado / no funciona
    if title_has_any(t, GLOBAL_HARD_BAD_TERMS):
        return "hard_bad_terms"
    return None


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

    must_include: List[str] = None
    must_exclude: List[str] = None
    condition_boost_terms: List[str] = None
    condition_bad_terms: List[str] = None

@dataclass
class Listing:
    source: str
    country_site: str
    item_id: str
    title: str
    price_eur: float
    shipping_eur: float
    url: str
    location_text: str
    condition: str = ""
    condition_id: str = ""
    short_desc: str = ""
    category_id: str = ""  # ✅ NUEVO

@dataclass
class Candidate:
    listing: Listing
    target: TargetModel
    match_score: int
    condition_score: int
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
# TARGET FILTERS
# =========================

def title_passes_target_filters(text: str, target: TargetModel) -> bool:
    t = norm(text)

    mi = norm_tokens(target.must_include or [])
    if mi and not all(x in t for x in mi):
        return False

    me = norm_tokens(target.must_exclude or [])
    if me and any(x in t for x in me):
        return False

    if any(x in {"automatic", "automatique", "automatik"} for x in mi):
        if title_has_any(t, AUTO_CONTRADICTIONS):
            return False

    return True

def compute_condition_score(text: str, target: TargetModel, condition_str: str = "", condition_id: str = "") -> int:
    t = norm(text + " " + (condition_str or ""))

    hard = global_noise_reject(t)
    if hard is not None:
        return -999

    boost = set(GLOBAL_BOOST_TERMS) | set(norm_tokens(target.condition_boost_terms or []))
    bad = set(GLOBAL_BAD_TERMS) | set(norm_tokens(target.condition_bad_terms or []))

    score = 0
    if any(x in t for x in boost):
        score += 15
    if any(x in t for x in bad):
        score -= 15

    uncertainty = {"untested", "not tested", "as is", "read the description", "see description", "balance ok"}
    if any(x in t for x in uncertainty):
        score -= 20

    return max(-50, min(25, score))


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

        query = (t.get("query") or "").strip()
        if not query:
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
            ebay_category_id=(t.get("ebay_category_id") or None),

            must_include=t.get("must_include", []) or [],
            must_exclude=t.get("must_exclude", []) or [],
            condition_boost_terms=t.get("condition_boost_terms", []) or [],
            condition_bad_terms=t.get("condition_bad_terms", []) or []
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
    data = {"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"}

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

    cat = (category_id or "").strip() or EBAY_DEFAULT_CATEGORY_ID
    params: Dict[str, str] = {
        "q": query,
        "limit": str(min(max(limit, 1), 200)),
        "category_ids": cat
    }

    r = requests.get(base, headers=headers, params=params, timeout=HTTP_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"Browse search error {r.status_code}: {r.text[:500]}")

    data = r.json()
    items = data.get("itemSummaries", []) or []
    out: List[Listing] = []

    for it in items:
        title = it.get("title") or ""
        url = clean_url(it.get("itemWebUrl") or "")
        item_id = it.get("itemId") or ""
        if not item_id:
            continue

        price = it.get("price") or {}
        try:
            price_eur = float(price.get("value"))
        except Exception:
            continue

        ship_eur = 0.0
        item_loc = it.get("itemLocation") or {}
        cc = (item_loc.get("country") or "").upper()
        city = (item_loc.get("city") or "")
        loc = f"{city}, {cc}".strip(", ") if (city or cc) else (cc or "")

        cond = it.get("condition") or ""
        cond_id = str(it.get("conditionId") or "")

        out.append(Listing(
            source="ebay",
            country_site=EBAY_MARKETPLACE_ID,
            item_id=item_id,
            title=title,
            price_eur=price_eur,
            shipping_eur=ship_eur,
            url=url,
            location_text=loc,
            condition=cond,
            condition_id=cond_id,
            short_desc="",
            category_id=""  # se rellena en detalle
        ))

    return out

def ebay_get_item_detail(token: str, item_id: str) -> Dict:
    safe_id = quote(item_id, safe="")
    url = f"https://api.ebay.com/buy/browse/v1/item/{safe_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "X-EBAY-C-MARKETPLACE-ID": EBAY_MARKETPLACE_ID,
        "Accept": "application/json"
    }
    try:
        r = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
    except Exception:
        return {}
    if r.status_code != 200:
        return {}
    return r.json()

def eur_value(money: Dict) -> Optional[float]:
    if not isinstance(money, dict):
        return None
    v = money.get("value")
    cur = (money.get("currency") or "").upper()
    try:
        fv = float(v)
    except Exception:
        return None
    if cur and cur != "EUR":
        return None
    return fv

def extract_category_id(detail: Dict) -> str:
    # eBay no siempre es consistente; probamos varios campos
    for k in ("primaryCategoryId", "categoryId"):
        v = detail.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # a veces viene en categoryPath (string)
    cp = detail.get("categoryPath")
    if isinstance(cp, str):
        m = re.search(r"\b(\d{4,})\b", cp)
        if m:
            return m.group(1)
    # a veces como lista de categorías
    cats = detail.get("categories")
    if isinstance(cats, list) and cats:
        c0 = cats[0]
        if isinstance(c0, dict):
            cid = c0.get("categoryId")
            if isinstance(cid, str) and cid.strip():
                return cid.strip()
    return ""

def enrich_listing_from_detail(li: Listing, detail: Dict) -> Listing:
    if not detail:
        return li

    li.condition = detail.get("condition") or li.condition or ""
    li.condition_id = str(detail.get("conditionId") or li.condition_id or "")

    sd = detail.get("shortDescription")
    if isinstance(sd, str) and sd.strip():
        li.short_desc = sd.strip()

    ship_opts = detail.get("shippingOptions") or []
    if isinstance(ship_opts, list) and ship_opts:
        for opt in ship_opts:
            sc = opt.get("shippingCost") or {}
            val = eur_value(sc)
            if val is not None:
                li.shipping_eur = float(val)
                break

    loc = detail.get("itemLocation") or {}
    cc = (loc.get("country") or "").upper()
    city = (loc.get("city") or "")
    new_loc = f"{city}, {cc}".strip(", ") if (city or cc) else (cc or "")
    if new_loc:
        li.location_text = new_loc

    li.url = clean_url(li.url)

    # ✅ categoría
    li.category_id = extract_category_id(detail) or li.category_id

    return li


# =========================
# TELEGRAM
# =========================

def _tg_send_one(token: str, chat: str, text: str) -> None:
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat, "text": text, "disable_web_page_preview": True},
        timeout=HTTP_TIMEOUT
    )
    if r.status_code != 200:
        raise RuntimeError(f"Telegram error {r.status_code}: {r.text[:500]}")

def tg_send(msg: str) -> None:
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    if not token or not chat:
        raise RuntimeError("Faltan TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID en env.")

    text = (msg or "").strip()
    if not text:
        return

    if len(text) <= TG_MAX_LEN:
        _tg_send_one(token, chat, text)
        return

    lines = text.split("\n")
    chunk = ""
    for ln in lines:
        candidate = (chunk + "\n" + ln) if chunk else ln
        if len(candidate) > TG_MAX_LEN:
            if chunk:
                _tg_send_one(token, chat, chunk)
            chunk = ln
        else:
            chunk = candidate
    if chunk:
        _tg_send_one(token, chat, chunk)


# =========================
# MAIN
# =========================

def main():
    now = now_utc()

    targets = load_targets_from_json("target_list.json")
    token = ebay_oauth_app_token()

    listings: List[Listing] = []
    counts = {"ebay": 0}

    # FASE 1: SEARCH
    for t in targets:
        got = ebay_search(token, query=t.query, category_id=t.ebay_category_id, limit=EBAY_LIMIT)
        listings.extend(got)
        counts["ebay"] += len(got)
        time.sleep(EBAY_THROTTLE_S)

    # Dedup por item_id
    seen = set()
    unique: List[Listing] = []
    for li in listings:
        if not li.item_id:
            continue
        if li.item_id in seen:
            continue
        seen.add(li.item_id)
        unique.append(li)
    listings = unique

    if not listings:
        tg_send(
            f"🕗 TIMELAB Morning Scan (eBay API {EBAY_MARKETPLACE_ID})\n{now}\n\n"
            f"⚠️ 0 listings recogidos desde eBay API.\n"
        )
        return

    # Pre-score para decidir cuáles enriquecer
    prescored: List[Tuple[int, float, Listing, TargetModel]] = []
    for li in listings:
        if not is_eu_location(li.location_text):
            continue
        if global_noise_reject(li.title) is not None:
            continue

        best_ms = -1
        best_t = None
        for t in targets:
            if not title_passes_target_filters(li.title, t):
                continue
            ms = compute_match_score(li.title, t)
            if ms > best_ms:
                best_ms = ms
                best_t = t
        if not best_t:
            continue

        expected_close = best_t.catwiki_close_med if best_t.catwiki_close_med > 0 else max(li.price_eur * 1.8, li.price_eur + 150)
        net, _ = estimate_net_profit(li.price_eur, li.shipping_eur, expected_close)
        prescored.append((best_ms, net, li, best_t))

    prescored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    to_enrich = prescored[:max(0, DETAIL_FETCH_N)]
    enrich_ids = {li.item_id for _, _, li, _ in to_enrich}

    # FASE 2: DETAIL ENRICHMENT (solo TOP N)
    enriched_count = 0
    enriched_map: Dict[str, Listing] = {}
    for _, _, li, _ in to_enrich:
        detail = ebay_get_item_detail(token, li.item_id)
        new_li = enrich_listing_from_detail(li, detail)
        enriched_map[new_li.item_id] = new_li
        enriched_count += 1
        time.sleep(EBAY_DETAIL_THROTTLE_S)

    listings = [enriched_map.get(li.item_id, li) for li in listings]

    # Selección final
    candidates: List[Candidate] = []
    raw_scored: List[Tuple[int, int, Listing, TargetModel, float, float]] = []

    for li in listings:
        if not is_eu_location(li.location_text):
            continue

        # ✅ filtro duro por categoría (si ya la tenemos)
        if li.category_id and EBAY_ALLOWED_CATEGORY_IDS and li.category_id not in EBAY_ALLOWED_CATEGORY_IDS:
            continue

        detail_text = li.title
        if li.short_desc:
            detail_text += " " + li.short_desc
        if li.condition:
            detail_text += " " + li.condition

        # ✅ filtro duro por incompleto / piezas
        if global_noise_reject(detail_text) is not None:
            continue

        best_ms = -1
        best_t: Optional[TargetModel] = None
        for t in targets:
            if not title_passes_target_filters(detail_text, t):
                continue
            ms = compute_match_score(li.title, t)
            if ms > best_ms:
                best_ms = ms
                best_t = t
        if not best_t:
            continue

        cscore = compute_condition_score(detail_text, best_t, condition_str=li.condition, condition_id=li.condition_id)
        if cscore == -999:
            continue

        expected_close = best_t.catwiki_close_med if best_t.catwiki_close_med > 0 else max(li.price_eur * 1.8, li.price_eur + 150)
        if cscore >= 15:
            expected_close *= 1.05
        elif cscore <= -35:
            expected_close *= 0.85
        elif cscore <= -20:
            expected_close *= 0.92

        net, roi = estimate_net_profit(li.price_eur, li.shipping_eur, expected_close)
        raw_scored.append((best_ms, cscore, li, best_t, net, roi))

        if best_t.buy_max > 0 and li.price_eur > (best_t.buy_max * BUY_MAX_MULT):
            continue
        if best_ms < MIN_MATCH_SCORE:
            continue
        if best_t.fake_risk not in ALLOW_FAKE_RISK:
            continue
        if cscore <= -35:
            continue
        if not (net >= MIN_NET_EUR or roi >= MIN_NET_ROI):
            continue

        candidates.append(Candidate(li, best_t, best_ms, cscore, expected_close, net, roi))

    candidates.sort(key=lambda c: (c.net_profit, c.match_score, c.condition_score), reverse=True)
    top = candidates[:10]

    header = [
        f"🕗 TIMELAB Morning Scan — TOP {len(top) if top else 0} (eBay API {EBAY_MARKETPLACE_ID})",
        f"{now}",
        "",
        f"Recolectado: eBay={counts['ebay']}",
        f"Filtros: net≥{MIN_NET_EUR:.0f}€ o ROI≥{int(MIN_NET_ROI*100)}% | match≥{MIN_MATCH_SCORE} | fake: {','.join(sorted(ALLOW_FAKE_RISK))}",
        f"BUY_MAX_MULT: {BUY_MAX_MULT}",
        f"EBAY_DEFAULT_CATEGORY_ID: {EBAY_DEFAULT_CATEGORY_ID}",
        f"DETAIL_FETCH_N: {DETAIL_FETCH_N} | details_fetched: {enriched_count}",
        f"EBAY_ALLOWED_CATEGORY_IDS: {','.join(sorted(EBAY_ALLOWED_CATEGORY_IDS))}",
        ""
    ]

    if not top:
        raw_scored.sort(key=lambda x: (x[0], x[4], x[1]), reverse=True)
        raw_top = raw_scored[:5]
        lines = header + ["❌ Sin oportunidades que pasen filtros.", "", "🧪 SMOKE TEST (Top RAW):", ""]
        for i, (ms, cs, li, t, net, roi) in enumerate(raw_top, 1):
            lines.append(
                f"{i}) [ebay] {li.title}\n"
                f"   💶 {li.price_eur:.0f}€ | 🚚 {li.shipping_eur:.0f}€ | Neto {net:.0f}€ | ROI {int(roi*100)}% | Match {ms} | Cond {cs}\n"
                f"   🧩 Target: {t.key}\n"
                f"   📍 {li.location_text}\n"
                f"   🧾 eBay cond: {li.condition or 'n/a'} | cat: {li.category_id or 'n/a'}\n"
                f"   🔗 {li.url}\n"
            )
        tg_send("\n".join(lines))
        return

    msg = header
    for i, c in enumerate(top, 1):
        li = c.listing
        msg.append(
            f"{i}) [ebay] {li.title}\n"
            f"   💶 Compra: {li.price_eur:.0f}€ | 🚚 Envío: {li.shipping_eur:.0f}€ | 🎯 Cierre est.: {c.expected_close:.0f}€\n"
            f"   ✅ Neto est.: {c.net_profit:.0f}€ | ROI: {int(c.net_roi*100)}% | Match: {c.match_score} | Cond: {c.condition_score}\n"
            f"   🧩 Target: {c.target.key}\n"
            f"   📍 {li.location_text}\n"
            f"   🧾 eBay cond: {li.condition or 'n/a'} | cat: {li.category_id or 'n/a'}\n"
            f"   🔗 {li.url}\n"
        )

    tg_send("\n".join(msg))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        err = f"❌ TIMELAB scanner crashed\n{now_utc()}\n\n{type(e).__name__}: {str(e)[:800]}"
        try:
            tg_send(err)
        except Exception:
            pass
        raise