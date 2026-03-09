import datetime as dt
import re
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup


HEADERS = {"User-Agent": "Mozilla/5.0"}
TIMEOUT = 20

BRANDS = [
    "omega", "seiko", "tissot", "longines", "certina", "zenith",
    "hamilton", "oris", "tag heuer", "citizen", "bulova", "rado",
]

BAD_IMAGE_TERMS = [
    "logo", "icon", "sprite", "banner", "placeholder", "avatar", "favicon",
]


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def extract_listing_id(url: str) -> Optional[str]:
    match = re.search(r"/([A-Z0-9_]+)\.html", url)
    return match.group(1) if match else None


def extract_price_eur(text: str) -> Optional[float]:
    if not text:
        return None

    match = re.search(r"(\d+[.,]\d{2})\s*€", text)
    if not match:
        return None

    raw = match.group(1)
    if "," in raw and "." in raw:
        if raw.rfind(",") > raw.rfind("."):
            raw = raw.replace(".", "").replace(",", ".")
        else:
            raw = raw.replace(",", "")
    else:
        raw = raw.replace(",", ".")

    try:
        return float(raw)
    except ValueError:
        return None


def infer_brand_hint(title: str, description: str) -> Optional[str]:
    text = f"{title} {description}".lower()
    for brand in BRANDS:
        if brand in text:
            return brand
    return None


def extract_image_urls(soup: BeautifulSoup) -> List[str]:
    image_urls: List[str] = []

    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-original")
        if not src:
            continue

        src = src.strip()

        if src.startswith("//"):
            src = "https:" + src

        if not src.startswith("http"):
            continue

        src_lower = src.lower()
        if any(term in src_lower for term in BAD_IMAGE_TERMS):
            continue

        if len(src) <= 30:
            continue

        if src not in image_urls:
            image_urls.append(src)

    return image_urls


def fetch_listing_details(url: str) -> Dict[str, Any]:
    response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    page_text = clean_text(soup.get_text(" ", strip=True))

    title = None
    h1 = soup.find("h1")
    if h1:
        title = clean_text(h1.get_text())

    if not title and soup.title:
        title = clean_text(soup.title.get_text())

    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        description = clean_text(meta_desc["content"])
    else:
        description = page_text[:1500]

    price_eur = extract_price_eur(page_text)
    image_urls = extract_image_urls(soup)
    main_image_url = image_urls[0] if image_urls else None
    brand_hint = infer_brand_hint(title or "", description or "")

    return {
        "source": "cashconverters_es",
        "listing_id": extract_listing_id(url),
        "url": url,
        "title": title,
        "price_eur": price_eur,
        "shipping_eur": None,
        "description": description,
        "brand_hint": brand_hint,
        "main_image_url": main_image_url,
        "image_urls": image_urls,
        "collected_at": dt.datetime.utcnow().isoformat() + "Z",
        "status": "raw_collected",
    }