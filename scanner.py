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

MIN_NET_EUR = 20
MIN_NET_ROI = 0.05
MIN_MATCH_SCORE = 50
ALLOW_FAKE_RISK = {"low", "medium"}

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
    return float(m.group(1)) if m else None

def is_eu_location(loc: str) -> bool:
    l = norm(loc)
    return not l or any(c in l for c in EU_COUNTRIES)

def compute_match_score(title: str, target: TargetModel) -> int:
    t = norm(title)
    kw_hits = sum(1 for kw in target.keywords if norm(kw) in t)
    score = int(60 * (kw_hits / max(1, len(target.keywords))))
    score += 10 if not target.refs else 0
    if any(w in t for w in ["replica","copy","fake","imitacion","imitation"]):
        score -= 50
    return max(0, min(100, score))

def estimate_net_profit(buy: float, ship: float, close: float) -> Tuple[float, float]:
    cost = buy + ship + PACKAGING_EUR + MISC_EUR
    revenue = close + SHIP_ARBITRAGE_EUR
    fees = close * CATWIKI_COMMISSION
    profit_bt = revenue - fees - cost
    tax = max(0, profit_bt) * EFFECTIVE_TAX_RATE_ON_PROFIT
    net = profit_bt - tax
    return net, net / max(1.0, cost)

# =========================
# TARGET LIST
# =========================

def load_targets_from_json(path="target_list.json") -> List[TargetModel]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)["targets"]

    targets = []
    for t in data:
        kws = [norm(k) for k in t["model_keywords"]]
        if norm(t["brand"]) not in kws:
            kws.insert(0, norm(t["brand"]))
        lo, hi = t["expected_catawiki_eur"]
        targets.append(TargetModel(
            key=t["id"],
            keywords=kws,
            refs=[],
            tier=t.get("tier","B"),
            fake_risk=t["risk"],
            catwiki_close_med=(lo+hi)/2,
            buy_max=t["max_buy_eur"]
        ))
    return targets

# =========================
# EBAY
# =========================

def ebay_search(model: TargetModel, base_url: str) -> List[Listing]:
    out = []
    r = requests.get(base_url, params={"_nkw": " ".join(model.keywords), "LH_BIN": 1}, headers=HEADERS)
    soup = BeautifulSoup(r.text, "lxml")

    for it in soup.select("li.s-item"):
        t = it.select_one(".s-item__title")
        p = it.select_one(".s-item__price")
        a = it.select_one("a.s-item__link")
        if not t or not p or not a:
            continue
        price = extract_price(p.text)
        if price is None:
            continue
        out.append(Listing(
            source="ebay",
            country_site=base_url.split("//")[1].split("/")[0],
            title=t.text.strip(),
            price_eur=price,
            shipping_eur=0.0,
            url=a["href"],
            location_text=""
        ))
    return out

# =========================
# CASHCONVERTERS (ACTIVO)
# =========================

def cashconverters_search(model: TargetModel, max_pages: int = 2) -> List[Listing]:
    out = []
    base = "https://www.cashconverters.es/es/es/comprar/relojes/relojes-de-pulsera"
    query = " ".join(model.keywords)

    for page in range(1, max_pages + 1):
        r = requests.get(base, params={"q": query, "page": page}, headers=HEADERS)
        soup = BeautifulSoup(r.text, "lxml")
        cards = soup.select("article.product-card")
        if not cards:
            break

        for c in cards:
            t = c.select_one(".product-card__title")
            p = c.select_one(".product-card__price")
            a = c.select_one("a.product-card__link")
            s = c.select_one(".product-card__store")

            if not t or not p or not a:
                continue

            price = extract_price(p.text)
            if price is None:
                continue

            url = a["href"]
            if url.startswith("/"):
                url = "https://www.cashconverters.es" + url

            out.append(Listing(
                source="cashconverters",
                country_site="cashconverters.es",
                title=t.text.strip(),
                price_eur=price,
                shipping_eur=0.0,
                url=url,
                location_text=s.text.strip() if s else ""
            ))

        time.sleep(1)

    return out

# =========================
# TELEGRAM
# =========================

def tg_send(msg: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat = os.getenv("TELEGRAM_CHAT_ID")
    requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat, "text": msg, "disable_web_page_preview": True}
    )

# =========================
# MAIN
# =========================

def main():
    now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    targets = load_targets_from_json()

    listings = []
    for t in targets:
        for base in EBAY_SEARCH_BASES:
            listings += ebay_search(t, base)
        listings += cashconverters_search(t)

    candidates = []
    for li in listings:
        if not is_eu_location(li.location_text):
            continue
        for t in targets:
            ms = compute_match_score(li.title, t)
            if ms < MIN_MATCH_SCORE:
                continue
            if li.price_eur > t.buy_max * 1.15:
                continue
            net, roi = estimate_net_profit(li.price_eur, li.shipping_eur, t.catwiki_close_med)
            if net >= MIN_NET_EUR or roi >= MIN_NET_ROI:
                candidates.append((net, ms, li, t, roi))

    candidates.sort(reverse=True, key=lambda x: (x[0], x[1]))
    top = candidates[:10]

    if not top:
        tg_send(f"🕗 TIMELAB Morning Scan\n{now}\n\nSin oportunidades válidas.")
        return

    msg = [f"🕗 TIMELAB Morning Scan\n{now}\n"]
    for i,(net,ms,li,t,roi) in enumerate(top,1):
        msg.append(
            f"{i}) {li.title}\n"
            f"💶 {li.price_eur:.0f}€ | 🎯 {t.catwiki_close_med:.0f}€\n"
            f"✅ Neto {net:.0f}€ | ROI {int(roi*100)}% | Match {ms}\n"
            f"{li.url}\n"
        )

    tg_send("\n".join(msg))

if __name__ == "__main__":
    main()