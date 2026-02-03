import os
import re
import time
import json
import traceback
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

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

DEFAULT_SHIPPING_EUR = 0.0
DEBUG_CC = os.getenv("CC_DEBUG", "1") == "1"

PAGE_SIZE = 24
SESSION = requests.Session()

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "es-ES,es;q=0.9",
}

SEED_URLS = [
    f"{BASE}/es/es/comprar/relojes/reloj-pulsera/",
    f"{BASE}/es/es/comprar/relojes/reloj-pulsera-premium/",
    f"{BASE}/es/es/comprar/relojes/reloj-alta-gama/",
]

REPUTABLE_BRANDS = {
    "rolex","tudor","omega","longines","tag heuer","breitling","zenith",
    "jaeger lecoultre","iwc","panerai","cartier","tissot","hamilton",
    "certina","oris","rado","seiko","grand seiko","citizen","orient",
    "mido","doxa","eterna","fortis","sinn","nomos","frederique constant",
    "vacheron constantin","audemars piguet","patek philippe",
    "girard perregaux","glashutte","hublot","chopard","montblanc",
}

BANNED_BRANDS = {
    "lotus","festina","diesel","armani","fossil","guess","dkny",
    "swatch","welder","ice watch","samsung","xiaomi","apple",
    "garmin","fitbit","huawei","casio"
}

BANNED_KEYWORDS = {
    "smartwatch","reloj inteligente","fitness","pulsera actividad",
    "sin funcionar","no funciona","para piezas","averiado"
}

def canon(s):
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

REPUTABLE_CANON = {canon(b) for b in REPUTABLE_BRANDS}
BANNED_CANON = {canon(b) for b in BANNED_BRANDS}

PRICE_RE = re.compile(r"(\d{1,3}(?:\.\d{3})*,\d{2})\s*€")
ID_RE = re.compile(r"/segunda-mano/([^/]+)\.html")

def euro(s):
    return float(s.replace(".", "").replace(",", "."))

def tg(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    SESSION.post(url, json={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg,
        "disable_web_page_preview": True
    })

def fetch(url):
    r = SESSION.get(url, headers=HEADERS, timeout=CC_TIMEOUT)
    return r.text

def detect_brand(text):
    t = canon(text)
    for b in REPUTABLE_CANON:
        if b in t:
            return b, "reputable"
    for b in BANNED_CANON:
        if b in t:
            return b, "banned"
    return "", "no_brand"

def load_targets():
    with open("target_list.json", encoding="utf-8") as f:
        return json.load(f)

def best_target(title, targets):
    t = canon(title)
    best, score = None, 0
    for trg in targets:
        s = 0
        if canon(trg["brand"]) in t:
            s += 60
        for kw in trg.get("keywords", []):
            if canon(kw) in t:
                s += 10
        if s > score:
            best, score = trg, s
    return best, score

def estimate_net(buy, close):
    net = close - buy
    roi = net / buy if buy else -1
    return net, roi

def run():
    targets = load_targets()
    seen = set()
    scanned = 0

    c_rep = c_ban = c_nb = 0
    c_match = c_net = 0
    bad = 0

    opps = []

    for seed in SEED_URLS:
        start = 0
        while scanned < CC_MAX_ITEMS and c_rep < CC_GOOD_BRANDS_TARGET:
            url = f"{seed}?start={start}&sz={PAGE_SIZE}"
            html = fetch(url)
            soup = BeautifulSoup(html, "lxml")
            links = soup.select('a[href*="/segunda-mano/"]')

            if not links:
                break

            for a in links:
                href = a.get("href")
                if not href:
                    continue
                full = urljoin(BASE, href)
                m = ID_RE.search(full)
                if not m or m.group(1) in seen:
                    continue
                seen.add(m.group(1))

                page = fetch(full)
                s = BeautifulSoup(page, "lxml")

                title = s.select_one("h1")
                title = title.get_text(strip=True) if title else ""
                text = s.get_text(" ", strip=True)

                scanned += 1
                time.sleep(CC_THROTTLE_S)

                if not title:
                    bad += 1
                    continue

                brand, cls = detect_brand(title + " " + text)

                if cls == "banned":
                    c_ban += 1
                    continue
                if cls != "reputable":
                    c_nb += 1
                    continue

                c_rep += 1

                if any(k in canon(title) for k in BANNED_KEYWORDS):
                    continue

                mprice = PRICE_RE.search(text)
                if not mprice:
                    bad += 1
                    continue
                buy = euro(mprice.group(1))

                target, match = best_target(title, targets)
                if not target or match < MIN_MATCH_SCORE:
                    continue
                c_match += 1

                close = target["base_close_eur"] * CLOSE_HAIRCUT
                net, roi = estimate_net(buy, close)

                if not (net >= MIN_NET_EUR or roi >= MIN_NET_ROI):
                    continue
                c_net += 1

                opps.append((net, title, buy, close, roi, match, full, target["id"]))

            start += PAGE_SIZE

    opps.sort(reverse=True)
    top = opps[:10]

    msg = f"🕗 TIMELAB Morning Scan — TOP {len(top)} (CashConverters ES)\n"
    if not top:
        msg += "\nNo se encontraron oportunidades que cumplan filtros (marca reputada + match + net/ROI)."
    else:
        for i,o in enumerate(top,1):
            msg += f"\n{i}) [cc] {o[1]}\n💶 Compra: {o[2]:.2f}€ | 🎯 Cierre est.: {o[3]:.0f}€\n✅ Neto: {o[0]:.0f}€ | ROI: {o[4]*100:.0f}% | Match:{o[5]}\n🧩 Target:{o[7]}\n🔗 {o[6]}\n"

    tg(msg)

    if DEBUG_CC:
        dbg = (
            f"🧪 TIMELAB CC Debug — scanned:{scanned} | page_bad:{bad}\n"
            f"brands: reputable:{c_rep} | banned:{c_ban} | no_brand:{c_nb}\n"
            f"passed: match_ok:{c_match} | net_ok:{c_net}"
        )
        tg(dbg)

if __name__ == "__main__":
    try:
        run()
    except Exception:
        tg("❌ TIMELAB CashConverters scanner crashed\n" + traceback.format_exc()[:3000])