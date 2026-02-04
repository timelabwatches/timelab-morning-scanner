import os
import re
import time
import json
import traceback
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import Any, Dict, List, Tuple, Optional

BASE = "https://www.cashconverters.es"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

CC_TIMEOUT = float(os.getenv("CC_TIMEOUT", "20"))
CC_THROTTLE_S = float(os.getenv("CC_THROTTLE_S", "0.8"))
CC_MAX_ITEMS = int(os.getenv("CC_MAX_ITEMS", "500"))
CC_GOOD_BRANDS_TARGET = int(os.getenv("CC_GOOD_BRANDS_TARGET", "60"))

MIN_MATCH_SCORE = int(os.getenv("CC_MIN_MATCH_SCORE", "65"))
MIN_NET_EUR = float(os.getenv("CC_MIN_NET_EUR", "20"))
MIN_NET_ROI = float(os.getenv("CC_MIN_NET_ROI", "0.08"))
CLOSE_HAIRCUT = float(os.getenv("CLOSE_HAIRCUT", "0.90"))

DEFAULT_SHIPPING_EUR = float(os.getenv("CC_DEFAULT_SHIPPING_EUR", "0.0"))
DEBUG_CC = os.getenv("CC_DEBUG", "1") == "1"

PAGE_SIZE = int(os.getenv("CC_PAGE_SIZE", "24"))
SRULE = os.getenv("CC_SRULE", "new")

SESSION = requests.Session()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

SEED_URLS = [
    f"{BASE}/es/es/comprar/relojes/reloj-pulsera/",
    f"{BASE}/es/es/comprar/relojes/reloj-pulsera-premium/",
    f"{BASE}/es/es/comprar/relojes/reloj-alta-gama/",
]

REPUTABLE_BRANDS = {
    "rolex","tudor","omega","longines","tag heuer","breitling","zenith",
    "jaeger lecoultre","iwc","panerai","cartier","bulgari","chopard",
    "tissot","hamilton","certina","oris","rado","mido",
    "seiko","grand seiko","citizen","orient",
    "doxa","eterna","fortis","sinn","nomos","frederique constant",
    "vacheron constantin","audemars piguet","patek philippe",
    "girard perregaux","glashutte","glashutte original","hublot","montblanc",
    "maurice lacroix","raymond weil","alpina",
}

BANNED_BRANDS = {
    "lotus","festina","diesel","armani","emporio armani","fossil","guess","dkny",
    "tommy hilfiger","calvin klein","police","boss","hugo boss",
    "welder","ice watch","icewatch",
    "samsung","xiaomi","apple","garmin","fitbit","huawei",
    "casio",
}

BANNED_KEYWORDS_TITLE = {
    "smartwatch","reloj inteligente","pulsera actividad","fitness",
    "apple watch","galaxy watch","fitbit","garmin",
    "sin funcionar","no funciona","para piezas","solo piezas","averiado","defectuoso","incompleto",
}

PRICE_RE = re.compile(r"(\d{1,3}(?:\.\d{3})*,\d{2})\s*€")
ID_RE = re.compile(r"/segunda-mano/([^/]+)\.html")

def canon(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("&", " ")
    s = re.sub(r"[-_/\.]", " ", s)
    s = re.sub(r"[^a-z0-9áéíóúñü ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    trans = str.maketrans("áéíóúñü", "aeiounu")
    return s.translate(trans)

REPUTABLE_CANON = {canon(b) for b in REPUTABLE_BRANDS}
BANNED_CANON = {canon(b) for b in BANNED_BRANDS}
BANNED_KEYWORDS_CANON = {canon(k) for k in BANNED_KEYWORDS_TITLE}

def euro(s: str) -> float:
    return float(s.replace(".", "").replace(",", "."))

def tg_send(msg: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    r = SESSION.post(url, timeout=CC_TIMEOUT, json={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg,
        "disable_web_page_preview": True
    })
    r.raise_for_status()

def fetch(url: str) -> str:
    r = SESSION.get(url, headers=HEADERS, timeout=CC_TIMEOUT, allow_redirects=True)
    r.raise_for_status()
    return r.text

def build_page_url(seed: str, start: int) -> str:
    joiner = "&" if "?" in seed else "?"
    return f"{seed}{joiner}srule={SRULE}&start={start}&sz={PAGE_SIZE}"

def detect_brand_from_text(text: str) -> Tuple[str, str]:
    t = canon(text)
    for b in sorted(REPUTABLE_CANON, key=len, reverse=True):
        if b and b in t:
            return b, "reputable"
    for b in sorted(BANNED_CANON, key=len, reverse=True):
        if b and b in t:
            return b, "banned"
    return "", "no_brand"

def title_has_banned_keywords(title: str) -> bool:
    t = canon(title)
    return any(k in t for k in BANNED_KEYWORDS_CANON)

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
    seen = set()
    uniq = []
    for cc_id, url in out:
        if cc_id in seen:
            continue
        seen.add(cc_id)
        uniq.append((cc_id, url))
    return uniq

def parse_detail_page(url: str) -> Dict[str, Any]:
    html = fetch(url)
    soup = BeautifulSoup(html, "lxml")

    h1 = soup.select_one("h1")
    title = h1.get_text(" ", strip=True) if h1 else ""

    text = soup.get_text(" ", strip=True)
    m = PRICE_RE.search(text)
    price = euro(m.group(1)) if m else None

    page_ok = bool(title) and (price is not None)

    shop = ""
    for s in soup.stripped_strings:
        if "cash converters" in s.lower():
            shop = s.strip()
            break

    return {
        "url": url,
        "title": title,
        "text": text,
        "price": price,
        "shop": shop,
        "page_ok": page_ok
    }

def _to_float(v: Any) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        vv = v.strip()
        # permite "650", "650.0", "650,00"
        vv = vv.replace("€", "").strip()
        vv = vv.replace(".", "").replace(",", ".")
        try:
            return float(vv)
        except Exception:
            return 0.0
    return 0.0

def _pick_first(d: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def load_targets(path: str = "target_list.json") -> Tuple[List[Dict[str, Any]], str]:
    """
    Devuelve (targets, diag_text).
    Soporta múltiples formatos:
      - [ {...}, {...} ]
      - { "targets": [ {...} ] }
      - { "TISSOT_PRX_AUTO": {...}, "OMEGA_SEAMASTER": {...} }  (mapa)
    Soporta claves alternativas para base close:
      base_close_eur, base_close, baseClose, close, close_eur, catawiki_close, catawiki_close_eur, base
    Soporta brand en:
      brand, make, manufacturer
    Soporta id en:
      id, target, name, key (si venía como dict-map)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    diag_parts = [f"type:{type(data).__name__}"]

    items: List[Any] = []
    if isinstance(data, dict):
        if "targets" in data and isinstance(data["targets"], list):
            items = data["targets"]
            diag_parts.append("shape:{targets:[...]}")
        else:
            # dict-map: id -> obj
            items = []
            for k, v in data.items():
                if isinstance(v, dict):
                    vv = dict(v)
                    vv["_key_from_map"] = k
                    items.append(vv)
            diag_parts.append("shape:{id:{...}}")
    elif isinstance(data, list):
        items = data
        diag_parts.append("shape:[...]")
    else:
        return [], " | ".join(diag_parts) + " | unsupported"

    valid: List[Dict[str, Any]] = []
    sample_keys = set()

    for it in items:
        if not isinstance(it, dict):
            continue

        for k in it.keys():
            sample_keys.add(str(k))

        tid = _pick_first(it, ["id", "target", "name"])
        if not tid:
            tid = it.get("_key_from_map", "")
        tid = str(tid).strip()

        brand = _pick_first(it, ["brand", "make", "manufacturer"])
        brand = str(brand).strip() if brand else ""

        base_raw = _pick_first(it, [
            "base_close_eur", "base_close", "baseClose",
            "close", "close_eur",
            "catawiki_close", "catawiki_close_eur",
            "base", "baseCloseEur"
        ])
        base = _to_float(base_raw)

        keywords = it.get("keywords") or it.get("kw") or it.get("tags") or []
        if isinstance(keywords, str):
            keywords = [keywords]
        if not isinstance(keywords, list):
            keywords = []

        fake_risk = (it.get("fake_risk") or it.get("risk") or "low")
        fake_risk = str(fake_risk).lower()

        # VALIDACIÓN mínima (más laxa, pero útil)
        # - id requerido
        # - brand requerido
        # - base_close > 0 requerido (si no, no podemos estimar net)
        if tid and brand and base > 0:
            valid.append({
                "id": tid,
                "brand": brand,
                "base_close_eur": base,
                "keywords": keywords,
                "fake_risk": fake_risk,
            })

    diag_parts.append(f"items:{len(items)}")
    diag_parts.append(f"valid:{len(valid)}")
    if sample_keys:
        ks = sorted(list(sample_keys))[:25]
        diag_parts.append("keys_sample:" + ",".join(ks))

    return valid, " | ".join(diag_parts)

def best_target(title: str, targets: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], int]:
    t = canon(title)
    best = None
    best_score = -1
    for trg in targets:
        score = 0
        brand = canon(trg.get("brand", ""))
        if brand and brand in t:
            score += 60
        for kw in (trg.get("keywords") or []):
            kwc = canon(str(kw))
            if kwc and kwc in t:
                score += 10
        if score > best_score:
            best_score = score
            best = trg
    return best, max(0, best_score)

def estimate_close(trg: Dict[str, Any]) -> float:
    base = float(trg.get("base_close_eur") or 0.0)
    if base <= 0:
        return 0.0
    return base * CLOSE_HAIRCUT

def estimate_net(buy: float, shipping: float, close: float) -> Tuple[float, float]:
    if buy <= 0 or close <= 0:
        return -9999.0, -1.0
    cost = buy + shipping
    net = close - cost
    roi = net / cost if cost > 0 else -1.0
    return net, roi

def run() -> None:
    targets, diag = load_targets("target_list.json")

    # ✅ NO CRASH: si targets inválidos, avisamos por Telegram y salimos limpio
    if not targets:
        tg_send(
            "❌ TIMELAB CashConverters scanner: target_list.json inválido (no se pudieron cargar targets válidos)\n\n"
            f"diag: {diag}\n\n"
            "Acción: revisa el formato o pásame una captura/pegado del inicio de target_list.json (primeros 30-50 lines) y lo adapto."
        )
        return

    seen = set()
    scanned = 0
    page_bad = 0

    c_rep = 0
    c_ban = 0
    c_nb = 0

    c_match_ok = 0
    c_net_ok = 0

    opps: List[Dict[str, Any]] = []

    for seed in SEED_URLS:
        start = 0
        while scanned < CC_MAX_ITEMS and c_rep < CC_GOOD_BRANDS_TARGET:
            page_url = build_page_url(seed, start)
            listing_html = fetch(page_url)
            pairs = extract_listing_urls(listing_html)
            if not pairs:
                break

            for cc_id, url in pairs:
                if scanned >= CC_MAX_ITEMS or c_rep >= CC_GOOD_BRANDS_TARGET:
                    break
                if cc_id in seen:
                    continue
                seen.add(cc_id)

                data = parse_detail_page(url)
                scanned += 1
                time.sleep(CC_THROTTLE_S)

                if not data.get("page_ok"):
                    page_bad += 1
                    continue

                title = data.get("title", "")
                text = data.get("text", "")
                buy = float(data.get("price") or 0.0)

                brand, cls = detect_brand_from_text(title + " " + text)
                if cls == "banned":
                    c_ban += 1
                    continue
                if cls != "reputable":
                    c_nb += 1
                    continue
                c_rep += 1

                if title_has_banned_keywords(title):
                    continue

                trg, match = best_target(title, targets)
                if not trg or match < MIN_MATCH_SCORE:
                    continue
                c_match_ok += 1

                close_est = estimate_close(trg)
                net, roi = estimate_net(buy, DEFAULT_SHIPPING_EUR, close_est)

                if not (net >= MIN_NET_EUR or roi >= MIN_NET_ROI):
                    continue
                c_net_ok += 1

                opps.append({
                    "net": net,
                    "roi": roi,
                    "match": match,
                    "title": title,
                    "buy": buy,
                    "close": close_est,
                    "url": url,
                    "target": trg.get("id", ""),
                    "shop": data.get("shop") or "—",
                    "brand": brand,
                })

            start += PAGE_SIZE
            time.sleep(CC_THROTTLE_S)

        if scanned >= CC_MAX_ITEMS or c_rep >= CC_GOOD_BRANDS_TARGET:
            break

    opps.sort(key=lambda x: (x["net"], x["match"]), reverse=True)
    top = opps[:10]

    header = f"🕗 TIMELAB Morning Scan — TOP {len(top)} (CashConverters ES)"
    lines = [header]

    if not top:
        lines.append("\nNo se encontraron oportunidades que cumplan filtros (marca reputada + match + net/ROI).")
    else:
        for i, it in enumerate(top, 1):
            lines.append(
                f"{i}) [cc] {it['title']}\n"
                f"   💶 Compra: {it['buy']:.2f}€ | 🚚 Envío: {DEFAULT_SHIPPING_EUR:.2f}€ | 🎯 Cierre est.: {it['close']:.0f}€\n"
                f"   ✅ Neto est.: {it['net']:.0f}€ | ROI: {it['roi']*100:.0f}% | Match: {it['match']}\n"
                f"   🧩 Target: {it['target']}\n"
                f"   📍 {it['shop']}\n"
                f"   🔗 {it['url']}"
            )

    tg_send("\n\n".join(lines))

    if DEBUG_CC:
        dbg = (
            f"🧪 TIMELAB CC Debug — scanned:{scanned} | page_bad:{page_bad}\n"
            f"brands: reputable:{c_rep} | banned:{c_ban} | no_brand:{c_nb}\n"
            f"passed: match_ok:{c_match_ok} | net_ok:{c_net_ok}\n"
            f"thresholds: match>={MIN_MATCH_SCORE} | net>={MIN_NET_EUR} OR roi>={MIN_NET_ROI} | haircut:{CLOSE_HAIRCUT}\n"
            f"stop: max_items:{CC_MAX_ITEMS} | good_brands_target:{CC_GOOD_BRANDS_TARGET}\n"
            f"targets: {diag}"
        )
        tg_send(dbg)

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