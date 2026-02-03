import os
import re
import time
import traceback
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE = "https://www.cashconverters.es"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

DEFAULT_TIMEOUT = float(os.getenv("CC_TIMEOUT", "20"))
THROTTLE_S = float(os.getenv("CC_THROTTLE_S", "1.2"))
MAX_ITEMS = int(os.getenv("CC_MAX_ITEMS", "20"))
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

PRICE_RE = re.compile(r"(\d{1,3}(?:\.\d{3})*,\d{2})\s*€")
ID_RE = re.compile(r"/segunda-mano/([^/]+)\.html")


def euro_to_float(s: str) -> float:
    # "2.160,95" -> 2160.95
    s = s.replace(".", "").replace(",", ".")
    return float(s)


def tg_send(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID (GitHub Secrets?)")

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    r = requests.post(
        url,
        timeout=DEFAULT_TIMEOUT,
        json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "disable_web_page_preview": True,
        },
    )
    r.raise_for_status()


def fetch(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return r.text


def build_page_url(seed: str, start: int) -> str:
    joiner = "&" if "?" in seed else "?"
    return f"{seed}{joiner}srule={SRULE}&start={start}&sz={PAGE_SIZE}"


def extract_listing_urls(html: str):
    soup = BeautifulSoup(html, "lxml")
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


def parse_title_price_from_page(html: str):
    soup = BeautifulSoup(html, "lxml")

    h1 = soup.select_one("h1")
    title = h1.get_text(" ", strip=True) if h1 else "Sin título"

    text = soup.get_text(" ", strip=True)
    m = PRICE_RE.search(text)
    price = euro_to_float(m.group(1)) if m else None

    # señales simples
    lower = text.lower()
    estado = None
    for c in ("perfecto", "muy bueno", "bueno", "usado"):
        if c in lower:
            estado = c
            break

    disponibilidad = "envío" if ("a domicilio" in lower or "envio" in lower) else "tienda"

    tienda = None
    # intento simple: cualquier línea que contenga "Cash Converters"
    for s in soup.stripped_strings:
        if "cash converters" in s.lower():
            tienda = s.strip()
            break

    return {
        "title": title,
        "price": price,
        "estado": estado,
        "disponibilidad": disponibilidad,
        "tienda": tienda,
    }


def run():
    # 1) mensaje de arranque (para confirmar que llega a Telegram)
    tg_send("🧪 TIMELAB CashConverters scanner: started (debug)")

    items = []
    seen_ids = set()

    for seed in SEED_URLS:
        start = 0
        while len(items) < MAX_ITEMS:
            page_url = build_page_url(seed, start)
            html = fetch(page_url)

            pairs = extract_listing_urls(html)
            if not pairs:
                break

            for cc_id, url in pairs:
                if cc_id in seen_ids:
                    continue
                seen_ids.add(cc_id)

                # entrar a ficha para title/price/estado real
                detail_html = fetch(url)
                data = parse_title_price_from_page(detail_html)
                data["url"] = url
                data["id"] = cc_id
                items.append(data)

                time.sleep(THROTTLE_S)
                if len(items) >= MAX_ITEMS:
                    break

            start += PAGE_SIZE
            time.sleep(THROTTLE_S)

        if len(items) >= MAX_ITEMS:
            break

    header = f"🕗 TIMELAB Morning Scan — TOP {len(items)} (CashConverters ES)"
    lines = [header]

    for i, it in enumerate(items, 1):
        p = f"{it['price']:.2f}€" if isinstance(it.get("price"), (int, float)) else "—"
        cond = it.get("estado") or "—"
        disp = it.get("disponibilidad") or "—"
        shop = it.get("tienda") or "—"
        lines.append(
            f"{i}) [cc] {it.get('title','')}\n"
            f"   💶 Precio: {p} | Cond: {cond} | Disp: {disp}\n"
            f"   📍 {shop}\n"
            f"   🔗 {it.get('url','')}"
        )

    tg_send("\n\n".join(lines))


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