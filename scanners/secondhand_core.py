"""
TIMELAB — Shared scanning core for second-hand watch shops.
Used by Bilbotruke, RealCash, Locotoo scanners.

Design: each shop scanner provides a SiteConfig with URLs/selectors.
This module handles:
  - HTTP fetching with retries
  - Search page → product URL extraction
  - Detail page → Listing parsing
  - Target matching, scoring, gate decision
  - Telegram alert formatting
  - Cooldown state per-shop

Each shop is fully isolated:
  - Its own state file (state_<shop>.json)
  - Its own Telegram channel option (env: TELEGRAM_CHAT_ID_<SHOP>)
  - Its own catalog URLs

The scanner pattern mirrors cashconverters_scanner.py but is config-driven
rather than copy-pasted, so adding a 4th shop is ~50 lines.
"""
from __future__ import annotations
import os, re, json, time, random, logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup


# =============================================================================
# 1) LOGGING + ENV HELPERS
# =============================================================================
log = logging.getLogger("secondhand")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)


def env_int(name: str, default: int) -> int:
    try: return int(os.environ.get(name, str(default)))
    except (ValueError, TypeError): return default


def env_float(name: str, default: float) -> float:
    try: return float(os.environ.get(name, str(default)))
    except (ValueError, TypeError): return default


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default) or default


# =============================================================================
# 2) DATA TYPES
# =============================================================================
@dataclass
class SiteConfig:
    """Per-shop configuration. Build one of these for each scraper."""
    shop_id: str                                    # e.g. "bilbotruke"
    shop_label: str                                 # e.g. "Bilbotruke"
    base_url: str                                   # e.g. "https://bilbotruke.net"
    catalog_urls: List[str]                         # category pages to scan
    product_link_pattern: str                       # regex: which links are products
    product_url_prefix: str = ""                    # optional prefix to prepend if relative
    paginate: bool = True                           # follow paginated catalog
    max_pages: int = 5                              # max pages per category
    pagination_format: str = "?p={page}"            # how to build page N URL: ?p={page}, ?page={page}, /page/{page}/, etc.
    state_file: str = ""                            # if empty, derived from shop_id
    telegram_chat_env: str = "TELEGRAM_CHAT_ID"     # env var for Telegram chat
    user_agent: str = (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
    )

    def __post_init__(self):
        if not self.state_file:
            self.state_file = f"state_{self.shop_id}.json"


@dataclass
class Listing:
    title: str
    price_eur: float
    url: str
    shop_id: str
    cond: str = "desconocido"
    sku: str = ""
    raw_text: str = ""           # full page text for movement detection
    images: List[str] = field(default_factory=list)


# =============================================================================
# 3) HTTP / FETCH
# =============================================================================
def make_session(user_agent: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
    })
    return s


def polite_sleep(seconds: float) -> None:
    time.sleep(max(0.0, seconds + random.uniform(-0.2, 0.2)))


def fetch(url: str, session: requests.Session, timeout: int = 15) -> Optional[requests.Response]:
    try:
        r = session.get(url, timeout=timeout, allow_redirects=True)
        if r.status_code != 200:
            log.warning(f"HTTP {r.status_code} on {url}")
            return None
        return r
    except Exception as e:
        log.warning(f"Fetch error {url}: {e}")
        return None


# =============================================================================
# 4) GENERIC PARSING UTILITIES (reused across shops)
# =============================================================================
def canon(s: str) -> str:
    if not s: return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_price(text: str) -> Optional[float]:
    """Extract first price like '189,95 €' or '€199.95' or '129,00 EUR'."""
    if not text: return None
    # Common European format: 1.234,56 €
    # Try EUR-prefix format first: €189.50 or €1,234.56
    m = re.search(r"€\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)", text)
    if not m:
        # Suffix format: 199,95 € or 1.234,56 €
        m = re.search(r"(\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?|\d+(?:[.,]\d{1,2})?)\s*€", text)
    if not m: return None
    num = m.group(1)
    # Normalize to float
    if "," in num and "." in num:  # 1.234,56 → 1234.56
        num = num.replace(".", "").replace(",", ".")
    elif "," in num:  # 199,95 → 199.95
        num = num.replace(",", ".")
    try:
        return float(num)
    except ValueError:
        return None


def text_extract_price(soup: BeautifulSoup, html: str) -> Optional[float]:
    """Best-effort price extraction: meta → JSON-LD → DOM → regex fallback."""
    # 1) meta tags
    for sel in ['meta[property="product:price:amount"]', 'meta[itemprop="price"]', 'meta[name="price"]']:
        el = soup.select_one(sel)
        if el and el.get("content"):
            p = parse_price(el["content"]) or parse_price(f"{el['content']}€")
            if p and p > 0: return p

    # 2) JSON-LD
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(s.string or "{}")
            items = data if isinstance(data, list) else [data]
            for item in items:
                offers = item.get("offers", {}) if isinstance(item, dict) else {}
                if isinstance(offers, dict):
                    p = offers.get("price")
                    if p:
                        try: return float(str(p).replace(",", "."))
                        except ValueError: pass
        except (json.JSONDecodeError, AttributeError):
            continue

    # 3) DOM common selectors
    for sel in [
        '[itemprop="price"]', '.product-price', '.current-price', '.price',
        'span.regular-price', 'span.price-value', '.product_price', '.precio',
    ]:
        for el in soup.select(sel):
            txt = el.get_text(" ", strip=True)
            p = parse_price(txt)
            if p and p >= 5:  # ignore tiny prices that are probably shipping
                return p

    # 4) Regex fallback on full HTML
    p = parse_price(html)
    return p


def detect_condition(text_all: str) -> str:
    """Map common Spanish condition phrases to canonical labels."""
    t = canon(text_all)
    if "perfecto" in t or "impecable" in t or "como nuevo" in t:
        return "perfecto"
    if "excelente" in t or "muy buen estado" in t or "muy bueno" in t:
        return "muy bueno"
    if "buen estado" in t or "estado: bueno" in t:
        return "bueno"
    if "usado" in t or "signos de uso" in t or "de uso" in t:
        return "usado"
    if "aceptable" in t or "para reparar" in t:
        return "aceptable"
    return "desconocido"


# =============================================================================
# 5) MOVEMENT DETECTION (shared with CC)
# =============================================================================
_RE_QUARTZ = re.compile(
    r'\b7t[348][0-9]\b|\b5y2[0-9]\b|\bv739\b|\bv8[0-9]\b'
    r'|t120[._]41[0-9]|\bpr50\b|\btissot v8\b'
)
_RE_AUTO = re.compile(
    r'\bsnk[a-z0-9]{2,6}\b|\bsnab\b|\bsrp[ce][0-9]'
    r'|\b7[6][0-9]{2}[- ][0-9]|\b6[1256][0-9]{2}[- ][0-9]'
    r'|\bra-[a-z]{2}[0-9]'
)


def detect_movement(title: str, raw_text: str = "") -> str:
    """Returns: 'automatic' | 'manual' | 'quartz' | 'solar' | 'kinetic' | 'unknown'."""
    t = canon(title + " " + raw_text)
    if any(w in t for w in [
        "cuarzo", "quartz", "quarzo", "battery", "bateria", "batería",
        "solar", "kinetic", "eco-drive", "eco drive",
        "tipo de movimiento: cuarzo", "movimiento: cuarzo",
    ]):
        if "solar" in t: return "solar"
        if "kinetic" in t: return "kinetic"
        return "quartz"
    if any(w in t for w in ["cuerda manual", "manual wind", "carga manual",
                             "tipo de movimiento: cuerda", "manual de cuerda"]):
        return "manual"
    if any(w in t for w in [
        "automático", "automatico", "automatic", "powermatic", "rotor",
        "tipo de movimiento: automático", "tipo de movimiento: automatico",
        "movimiento: automatico", "movimiento: automático",
        "self-winding", "miyota", "nh35", "nh36", "4r35", "4r36", "6r15",
        "srpc", "srpe", "snk", "snab", "powermatic 80", "pm80",
    ]):
        return "automatic"
    if _RE_QUARTZ.search(t): return "quartz"
    if _RE_AUTO.search(t):   return "automatic"
    return "unknown"


# =============================================================================
# 6) SEARCH/CATALOG → PRODUCT URLs
# =============================================================================
def extract_product_urls(html: str, cfg: SiteConfig) -> List[str]:
    """Extract product detail URLs from a category/search HTML page."""
    soup = BeautifulSoup(html, "lxml")
    pattern = re.compile(cfg.product_link_pattern)
    urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if pattern.search(href):
            full = href
            if full.startswith("/") and cfg.product_url_prefix:
                full = cfg.product_url_prefix + full
            elif not full.startswith("http") and cfg.product_url_prefix:
                full = cfg.product_url_prefix + "/" + full.lstrip("/")
            urls.append(full)
    # Dedup preserving order
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u); out.append(u)
    return out


def collect_listings(cfg: SiteConfig, session: requests.Session,
                     max_items: int = 200, sleep_s: float = 1.0) -> List[str]:
    """Walk all catalog URLs (with pagination) and collect product detail URLs."""
    all_urls = []
    seen = set()
    for cat_url in cfg.catalog_urls:
        for page in range(1, cfg.max_pages + 1):
            if page == 1:
                url = cat_url
            else:
                # Avoid double-pagination: if cat_url already has explicit page in path/query, skip
                if "/page/" in cat_url or "page=" in cat_url:
                    break
                suffix = cfg.pagination_format.format(page=page)
                # If URL already has ? params, use & for query-style pagination
                if suffix.startswith("?") and "?" in cat_url:
                    suffix = "&" + suffix[1:]
                url = cat_url + suffix
            r = fetch(url, session)
            polite_sleep(sleep_s)
            if r is None: break
            urls = extract_product_urls(r.text, cfg)
            new = [u for u in urls if u not in seen]
            if not new:
                # No new URLs on this page → end of pagination
                break
            for u in new:
                seen.add(u); all_urls.append(u)
            if len(all_urls) >= max_items: return all_urls[:max_items]
            if not cfg.paginate: break
    return all_urls


# =============================================================================
# 7) DETAIL PAGE → LISTING
# =============================================================================
def parse_detail(html: str, url: str, cfg: SiteConfig) -> Optional[Listing]:
    soup = BeautifulSoup(html, "lxml")

    # Title (H1 or page title)
    title = ""
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(" ", strip=True)
    if not title and soup.title:
        title = soup.title.get_text(" ", strip=True)
    title = canon(title)
    if not title:
        return None

    # Price
    price = text_extract_price(soup, html)
    if price is None or price < 5:
        log.debug(f"No price for {url}")
        return None

    # Full page text (for movement detection + condition)
    text_all = canon(soup.get_text(" ", strip=True))
    cond = detect_condition(text_all)

    # SKU from URL
    sku = url.rstrip("/").split("/")[-1].replace(".html", "")

    return Listing(
        title=title,
        price_eur=float(price),
        url=url,
        shop_id=cfg.shop_id,
        cond=cond,
        sku=sku,
        raw_text=text_all[:2000],
    )


# =============================================================================
# 8) TARGET MATCHING (delegates to existing logic — reuse from CC)
# =============================================================================
REPUTABLE_BRANDS = {
    "omega", "rolex", "longines", "tudor", "iwc", "breitling", "tag heuer",
    "tag", "heuer", "tissot", "hamilton", "seiko", "citizen", "oris", "zenith",
    "junghans", "certina", "baume", "baume & mercier", "bulova", "cyma",
    "fortis", "cauny", "orient", "mido", "rado", "movado", "frederique constant",
    "raymond weil", "eberhard",
}
BANNED_BRANDS = {
    "lotus", "festina", "calvin klein", "ck", "diesel", "armani", "emporio armani",
    "michael kors", "guess", "tommy", "tommy hilfiger", "fossil", "dkny", "police",
    "welder", "casio", "garmin", "huawei", "amazfit", "fitbit",
    "swatch", "moonswatch", "bioceramic",
    "viceroy", "marea", "adidas", "nike", "puma", "skagen", "daniel wellington",
    "lanscotte", "radiant", "mark maddox", "marina militare",
}
_OVERRIDE_BANNED_TERMS = {"moonswatch", "bioceramic", "swatch"}


def extract_brand(title: str) -> Optional[str]:
    t = canon(title)
    if any(term in t for term in _OVERRIDE_BANNED_TERMS):
        for b in sorted(BANNED_BRANDS, key=lambda x: -len(x)):
            cb = canon(b)
            if cb and cb in t:
                return cb
        return "swatch"
    for b in sorted(REPUTABLE_BRANDS, key=lambda x: -len(x)):
        cb = canon(b)
        if re.search(r"(?<![a-z])" + re.escape(cb) + r"(?![a-z])", t):
            if b in ("tag", "heuer"): return "tag heuer"
            if "baume" in b: return "baume & mercier"
            return cb
    for b in sorted(BANNED_BRANDS, key=lambda x: -len(x)):
        if canon(b) in t: return canon(b)
    return None


def is_banned_brand(brand: Optional[str]) -> bool:
    return bool(brand and brand.lower() in BANNED_BRANDS)


def model_keyword_hits(title: str, target: Dict[str, Any]) -> int:
    t = canon(title)
    kws = [canon(x) for x in (target.get("model_keywords") or []) if isinstance(x, str)]
    return sum(1 for kw in kws if kw and len(kw) > 2 and kw in t)


def violates_must_exclude(title: str, target: Dict[str, Any], raw_text: str = "") -> bool:
    """Check title AND raw page text for exclude terms.
    raw_text typically contains the product spec sheet (where 'cuarzo' lives)."""
    t = canon(title) + " " + canon(raw_text)
    excl = [canon(x) for x in (target.get("must_exclude") or []) if isinstance(x, str)]
    return any(x and x in t for x in excl)


def violates_must_include(title: str, target: Dict[str, Any]) -> bool:
    t = canon(title)
    must = [canon(x) for x in (target.get("must_include") or []) if isinstance(x, str)]
    return bool(must) and not all(x in t for x in must)


def compute_match_score(title: str, target: Dict[str, Any]) -> int:
    t = canon(title)
    brand = canon(target.get("brand", ""))
    score = 0
    if brand and re.search(r"(?<![a-z])" + re.escape(brand) + r"(?![a-z])", t):
        score += 35
    must = [canon(x) for x in (target.get("must_include") or []) if isinstance(x, str)]
    if must:
        p = sum(1 for x in must if x and x in t)
        score += 35 if p == len(must) else int(35 * p / max(1, len(must)))
    else:
        score += 10
    score += min(25, model_keyword_hits(title, target) * 8)
    return score


def best_target(title: str, targets: List[Dict[str, Any]], raw_text: str = "") -> Tuple[Optional[Dict[str, Any]], int, int]:
    best_t, best_s, best_h = None, -1, 0
    for trg in targets:
        if violates_must_exclude(title, trg, raw_text): continue
        if violates_must_include(title, trg): continue
        hits = model_keyword_hits(title, trg)
        kws = trg.get("model_keywords") or []
        # FIXED v16c: require >=1 keyword hit for ALL targets, including _GENERIC.
        # The old _GENERIC bypass caused false positives like Seiko Sea Lion 2205
        # matching SEIKO_CHRONOGRAPH_GENERIC despite having no chrono keyword.
        if isinstance(kws, list) and len(kws) > 0 and hits < 1: continue
        s = compute_match_score(title, trg)
        best_is_gen = str((best_t or {}).get("id", "")).upper().endswith("_GENERIC")
        if (s > best_s) or (s == best_s and best_is_gen and not is_gen):
            best_s, best_t, best_h = s, trg, hits
    return best_t, best_s, best_h


# =============================================================================
# 9) ECONOMICS — same formula as CC
# =============================================================================
HAIRCUT = 0.90       # Catawiki conservatism
RATE = 0.125         # Catawiki commission
VAT_ON_FEE = 0.21
SHIPPING_NET = 35.0  # net shipping arbitrage (charged 50, real 15)


def estimate_close(target: Dict[str, Any]) -> float:
    p50 = float(target.get("catawiki_estimate", {}).get("p50") or 0.0)
    return round(p50 * HAIRCUT, 2)


def compute_net_profit(buy_price: float, close_est: float) -> Tuple[float, float]:
    fees = close_est * RATE * (1 + VAT_ON_FEE)
    net = round(close_est - fees + SHIPPING_NET - buy_price, 2)
    roi = net / buy_price if buy_price > 0 else 0.0
    return net, roi


# =============================================================================
# 10) COOLDOWN STATE
# =============================================================================
def load_state(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(path: str, state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def in_cooldown(sku: str, state: Dict[str, Any], hours: int = 48) -> bool:
    entry = state.get(sku)
    if not entry: return False
    ts = entry.get("last_seen_ts", 0) if isinstance(entry, dict) else 0
    age_h = (datetime.now(timezone.utc).timestamp() - ts) / 3600
    return age_h < hours


def update_state(sku: str, price: float, state: Dict[str, Any]) -> None:
    state[sku] = {
        "last_seen_ts": datetime.now(timezone.utc).timestamp(),
        "last_price": price,
    }


# =============================================================================
# 11) TELEGRAM
# =============================================================================
def telegram_send(text: str, token: str, chat_id: str) -> None:
    if not token or not chat_id:
        log.warning("Telegram credentials missing")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }, timeout=15)
        if r.status_code != 200:
            log.warning(f"Telegram error {r.status_code}: {r.text[:200]}")
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")


def format_alert_card(idx: int, listing: Listing, target: Dict[str, Any],
                       close_est: float, net: float, roi: float,
                       movement: str, shop_label: str) -> str:
    roi_pct = roi * 100
    roi_emoji = "🚀" if roi_pct >= 120 else ("💚" if roi_pct >= 80 else "🟡")
    bucket = "STRONG" if roi_pct >= 100 else "REVIEW"

    MOV_LABEL = {
        "automatic": "⚙️ Automático", "manual": "🔑 Cuerda manual",
        "quartz": "🔋 Cuarzo", "solar": "☀️ Solar",
        "kinetic": "⚡ Kinetic", "unknown": "❓ Mov. desconocido",
    }
    COND_EMOJI = {
        "perfecto": "🌟", "muy bueno": "✨", "bueno": "✔️",
        "usado": "⚠️", "aceptable": "⚠️", "desconocido": "❔",
    }

    title_clean = listing.title.title()[:80]
    raw_p50 = float(target.get("catawiki_estimate", {}).get("p50") or 0)
    target_id = target.get("id", "?")

    lines = [
        "──────────────────────────",
        f"{idx}) {roi_emoji} {bucket}  ·  {shop_label}",
        f"🏷 *{title_clean}*",
        "",
        f"💶 Compra: *{listing.price_eur:.0f}€*",
        f"🎯 Catawiki: ~*{close_est:.0f}€*  ·  +{net:.0f}€ neto  ·  {roi_emoji} {roi_pct:.0f}% ROI",
        "",
        f"{MOV_LABEL.get(movement, '❓')}  ·  {COND_EMOJI.get(listing.cond, '❔')} {listing.cond.title()}",
        f"🎯 Target: `{target_id}` (p50 raw {raw_p50:.0f}€)",
        f"🔗 {listing.url}",
    ]
    return "\n".join(lines)


# =============================================================================
# 12) MAIN SCAN ORCHESTRATOR
# =============================================================================
def run_scan(cfg: SiteConfig, targets_path: str = "target_list.json",
             min_net_eur: float = 30.0, min_roi: float = 0.10,
             max_buy_eur_default: float = 1000.0,
             max_items: int = 300, sleep_s: float = 1.0,
             cooldown_hours: int = 48) -> Dict[str, Any]:
    """Main scanning loop. Returns diagnostic dict."""
    log.info(f"=== {cfg.shop_label} scan starting ===")

    # Load targets
    with open(targets_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    targets = raw.get("targets", raw) if isinstance(raw, dict) else raw
    log.info(f"Loaded {len(targets)} targets")

    # Load state
    state = load_state(cfg.state_file)
    log.info(f"State: {len(state)} cooldown entries")

    # Telegram
    tg_token = env_str("TELEGRAM_BOT_TOKEN", "")
    tg_chat = env_str(cfg.telegram_chat_env, env_str("TELEGRAM_CHAT_ID", ""))

    session = make_session(cfg.user_agent)

    # Step 1: collect product URLs
    log.info(f"Collecting from {len(cfg.catalog_urls)} catalog URLs...")
    urls = collect_listings(cfg, session, max_items=max_items, sleep_s=sleep_s)
    log.info(f"Collected {len(urls)} product URLs")

    # Step 2: parse details + match
    diag = {"scanned": 0, "no_brand": 0, "banned": 0, "no_match": 0,
            "filtered": 0, "cooldown": 0, "alerts": 0}
    candidates = []
    for url in urls:
        diag["scanned"] += 1
        r = fetch(url, session)
        polite_sleep(sleep_s)
        if r is None: continue

        listing = parse_detail(r.text, url, cfg)
        if listing is None:
            diag["filtered"] += 1
            continue

        brand = extract_brand(listing.title)
        if not brand:
            diag["no_brand"] += 1; continue
        if is_banned_brand(brand):
            diag["banned"] += 1; continue

        target, score, hits = best_target(listing.title, targets, listing.raw_text)
        if not target:
            diag["no_match"] += 1; continue

        # Movement detection
        movement = detect_movement(listing.title, listing.raw_text)

        # Economics
        close_est = estimate_close(target)
        if close_est <= 0: continue
        net, roi = compute_net_profit(listing.price_eur, close_est)

        # Sanity check: ROI > 500% is almost always a target mismatch, not a real find.
        if roi > 5.0:
            log.warning(
                f"⚠️  Suspicious ROI {roi*100:.0f}% — target={target.get('id')} "
                f"p50={target.get('catawiki_estimate',{}).get('p50')} "
                f"buy={listing.price_eur} title='{listing.title[:50]}'"
            )

        # FIXED v16c: discount sanity check — buy_price < 18% of p50 is suspicious.
        # A premium watch like a TAG Carrera (~1500€) listed at 180€ in a professional
        # store is almost always: (a) price extraction error, (b) replica, (c) parts/broken.
        # Skip these to avoid false alarms.
        raw_p50 = float(target.get("catawiki_estimate", {}).get("p50") or 0)
        if raw_p50 > 0 and listing.price_eur < raw_p50 * 0.18:
            log.warning(
                f"⚠️  Skipping suspicious low price: target={target.get('id')} "
                f"p50={raw_p50} buy={listing.price_eur} ({listing.price_eur/raw_p50*100:.0f}% of p50) "
                f"title='{listing.title[:50]}'"
            )
            continue

        # Gate
        max_buy = float(target.get("max_buy_eur") or max_buy_eur_default)
        if listing.price_eur > max_buy: continue
        if net < min_net_eur or roi < min_roi: continue

        # Cooldown
        if in_cooldown(listing.sku, state, cooldown_hours):
            diag["cooldown"] += 1; continue

        candidates.append({
            "listing": listing, "target": target, "score": score, "hits": hits,
            "close_est": close_est, "net": net, "roi": roi,
            "movement": movement,
        })

    # Step 3: sort + alert top 5
    candidates.sort(key=lambda x: x["net"], reverse=True)
    top = candidates[:5]
    diag["alerts"] = len(top)

    if top:
        hora = datetime.now().strftime("%H:%M")
        header = (
            f"⌚ TIMELAB · {cfg.shop_label} · {hora}\n"
            f"{'━' * 26}\n"
        )
        body = "\n".join(
            format_alert_card(i + 1, c["listing"], c["target"],
                              c["close_est"], c["net"], c["roi"],
                              c["movement"], cfg.shop_label)
            for i, c in enumerate(top)
        )
        telegram_send(header + body, tg_token, tg_chat)

        # Update cooldown state
        for c in top:
            update_state(c["listing"].sku, c["listing"].price_eur, state)
        save_state(cfg.state_file, state)
    else:
        # Optional: send a "no opportunities" message
        hora = datetime.now().strftime("%H:%M")
        msg = (
            f"⌚ TIMELAB · {cfg.shop_label} · {hora}\n"
            f"{'━' * 26}\n"
            f"😴 Sin oportunidades hoy\n"
            f"Escaneados: {diag['scanned']} | Sin marca: {diag['no_brand']} | "
            f"Sin target: {diag['no_match']} | Cooldown: {diag['cooldown']}"
        )
        telegram_send(msg, tg_token, tg_chat)

    log.info(f"=== {cfg.shop_label} done: {diag} ===")
    return diag
