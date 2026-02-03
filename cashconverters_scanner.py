# cashconverters_scanner.py
import os
import re
import time
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE = "https://www.cashconverters.es"
DEFAULT_TIMEOUT = float(os.getenv("CC_TIMEOUT", "20"))
THROTTLE_S = float(os.getenv("CC_THROTTLE_S", "1.2"))
MAX_ITEMS = int(os.getenv("CC_MAX_ITEMS", "20"))          # commit 1: 20
PAGE_SIZE = int(os.getenv("CC_PAGE_SIZE", "24"))          # sz
SRULE = os.getenv("CC_SRULE", "new")                      # newest first

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TIMELAB-WATCHES/1.0; +https://github.com/)",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}

SEED_URLS = [
    # Reloj pulsera
    f"{BASE}/es/es/comprar/relojes/reloj-pulsera/reloj-pulsera-caballero/",
    f"{BASE}/es/es/comprar/relojes/reloj-pulsera/reloj-pulsera-senora/",
    f"{BASE}/es/es/comprar/relojes/reloj-pulsera/reloj-pulsera-unisex/",
    # Premium
    f"{BASE}/es/es/comprar/relojes/reloj-pulsera-premium/reloj-pulsera-premium-caballero/",
    f"{BASE}/es/es/comprar/relojes/reloj-pulsera-premium/reloj-pulsera-premium-senora/",
    f"{BASE}/es/es/comprar/relojes/reloj-pulsera-premium/reloj-pulsera-premium-unisex/",
    # Alta gama
    f"{BASE}/es/es/comprar/relojes/reloj-alta-gama/reloj-alta-gama-caballero/",
    f"{BASE}/es/es/comprar/relojes/reloj-alta-gama/reloj-alta-gama-senora/",
    f"{BASE}/es/es/comprar/relojes/reloj-alta-gama/reloj-alta-gama-unisex/",
]

PRICE_RE = re.compile(r"(\d{1,3}(?:\.\d{3})*,\d{2})\s*€")

def euro_to_float(s: str) -> float:
    # "2.160,95" -> 2160.95
    s = s.replace(".", "").replace(",", ".")
    return float(s)

def tg_send(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, timeout=DEFAULT_TIMEOUT, json={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True
    }).raise_for_status()

def fetch(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return r.text

def build_page_url(seed: str, start: int) -> str:
    joiner = "&" if "?" in seed else "?"
    return f"{seed}{joiner}srule={SRULE}&start={start}&sz={PAGE_SIZE}"

def extract_listing_cards(html: str):
    """
    Estrategia robusta sin depender de clases:
    - buscamos links a /segunda-mano/<ID>.html
    - alrededor suele aparecer precio + estado
    """
    soup = BeautifulSoup(html, "html.parser")
    links = soup.select('a[href*="/segunda-mano/"][href$=".html"]')
    seen = set()
    items = []

    for a in links:
        href = a.get("href", "").strip()
        if not href:
            continue
        url = urljoin(BASE, href)
        if url in seen:
            continue
        seen.add(url)

        # title: intenta alt de imagen dentro del link, si no texto del link
        img = a.select_one("img[alt]")
        title = (img.get("alt") if img else a.get_text(" ", strip=True)) or "Sin título"

        # intenta capturar precio/estado en un contenedor cercano
        container_text = a.parent.get_text(" ", strip=True) if a.parent else ""
        m = PRICE_RE.search(container_text)
        price = euro_to_float(m.group(1)) if m else None

        # estado (Perfecto/Bueno/Usado) si aparece cerca
        cond = None
        for c in ("Perfecto", "Bueno", "Usado"):
            if c.lower() in container_text.lower():
                cond = c
                break

        items.append({
            "title": title,
            "price": price,
            "url": url,
            "cond": cond,
        })
    return items

def run():
    all_items = []
    dedup = set()

    for seed in SEED_URLS:
        start = 0
        while len(all_items) < MAX_ITEMS and start <= 500:  # hard safety
            page_url = build_page_url(seed, start)
            html = fetch(page_url)
            batch = extract_listing_cards(html)

            if not batch:
                break

            for it in batch:
                if it["url"] in dedup:
                    continue
                dedup.add(it["url"])
                all_items.append(it)
                if len(all_items) >= MAX_ITEMS:
                    break

            start += PAGE_SIZE
            time.sleep(THROTTLE_S)

        if len(all_items) >= MAX_ITEMS:
            break

    # Telegram output (separado del eBay SIEMPRE)
    lines = [f"🕗 TIMELAB Morning Scan — TOP {len(all_items)} (CashConverters ES)"]
    for i, it in enumerate(all_items, 1):
        p = f"{it['price']:.2f}€" if isinstance(it["price"], (int, float)) else "—"
        cond = it["cond"] or "—"
        lines.append(
            f"{i}) [cc] {it['title']}\n"
            f"   💶 Precio: {p} | Cond: {cond}\n"
            f"   🔗 {it['url']}"
        )

    tg_send("\n\n".join(lines))

if __name__ == "__main__":
    try:
        run()
    except Exception:
        # mensaje de crash separado
        try:
            tg_send("❌ TIMELAB CashConverters scanner crashed")
        finally:
            # Importantísimo: salir 0 para NO romper el job de eBay
            raise SystemExit(0)