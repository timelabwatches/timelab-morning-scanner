from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote

import requests

from config import Settings


@dataclass
class EbayListing:
    item_id: str
    title: str
    price_eur: float
    shipping_eur: float
    url: str
    location_text: str
    condition: str = ""
    condition_id: str = ""
    short_desc: str = ""
    category_id: str = ""


def clean_url(url: str) -> str:
    if not url:
        return ""
    if "?" in url:
        url = url.split("?", 1)[0]
    return url


def eur_value(money: dict) -> Optional[float]:
    if not isinstance(money, dict):
        return None

    value = money.get("value")
    currency = str(money.get("currency", "")).upper()

    try:
        parsed = float(value)
    except Exception:
        return None

    if currency and currency != "EUR":
        return None

    return parsed


def extract_location(item_location: dict) -> str:
    if not isinstance(item_location, dict):
        return ""

    city = str(item_location.get("city", "")).strip()
    country = str(item_location.get("country", "")).strip().upper()

    if city and country:
        return f"{city}, {country}"
    if country:
        return country
    return city


def extract_category_id(detail: dict) -> str:
    if not isinstance(detail, dict):
        return ""

    for key in ("primaryCategoryId", "categoryId"):
        value = detail.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    categories = detail.get("categories")
    if isinstance(categories, list) and categories:
        first = categories[0]
        if isinstance(first, dict):
            category_id = first.get("categoryId")
            if isinstance(category_id, str) and category_id.strip():
                return category_id.strip()

    return ""


def get_oauth_token(settings: Settings) -> str:
    response = requests.post(
        "https://api.ebay.com/identity/v1/oauth2/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "client_credentials",
            "scope": "https://api.ebay.com/oauth/api_scope",
        },
        auth=(settings.ebay_client_id, settings.ebay_client_secret),
        timeout=settings.http_timeout,
    )

    if response.status_code != 200:
        raise RuntimeError(f"eBay OAuth error {response.status_code}: {response.text[:500]}")

    data = response.json()
    token = data.get("access_token", "").strip()

    if not token:
        raise RuntimeError("eBay OAuth returned empty access_token")

    return token


def search_listings(
    settings: Settings,
    token: str,
    query: str,
    category_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[EbayListing]:
    final_limit = limit if limit is not None else settings.ebay_limit
    final_category_id = (category_id or "").strip() or settings.ebay_default_category_id

    response = requests.get(
        "https://api.ebay.com/buy/browse/v1/item_summary/search",
        headers={
            "Authorization": f"Bearer {token}",
            "X-EBAY-C-MARKETPLACE-ID": settings.ebay_marketplace_id,
            "Accept": "application/json",
        },
        params={
            "q": query,
            "limit": str(min(max(final_limit, 1), 200)),
            "category_ids": final_category_id,
        },
        timeout=settings.http_timeout,
    )

    if response.status_code != 200:
        raise RuntimeError(f"eBay search error {response.status_code}: {response.text[:500]}")

    data = response.json()
    items = data.get("itemSummaries", []) or []
    results: list[EbayListing] = []

    for item in items:
        item_id = str(item.get("itemId", "")).strip()
        title = str(item.get("title", "")).strip()
        if not item_id or not title:
            continue

        price_block = item.get("price") or {}
        try:
            price_eur = float(price_block.get("value"))
        except Exception:
            continue

        listing = EbayListing(
            item_id=item_id,
            title=title,
            price_eur=price_eur,
            shipping_eur=0.0,
            url=clean_url(str(item.get("itemWebUrl", "")).strip()),
            location_text=extract_location(item.get("itemLocation") or {}),
            condition=str(item.get("condition", "")).strip(),
            condition_id=str(item.get("conditionId", "")).strip(),
        )
        results.append(listing)

    return results


def get_listing_detail(settings: Settings, token: str, item_id: str) -> dict:
    safe_item_id = quote(item_id, safe="")

    response = requests.get(
        f"https://api.ebay.com/buy/browse/v1/item/{safe_item_id}",
        headers={
            "Authorization": f"Bearer {token}",
            "X-EBAY-C-MARKETPLACE-ID": settings.ebay_marketplace_id,
            "Accept": "application/json",
        },
        timeout=settings.http_timeout,
    )

    if response.status_code != 200:
        return {}

    try:
        return response.json()
    except Exception:
        return {}


def enrich_listing_from_detail(listing: EbayListing, detail: dict) -> EbayListing:
    if not isinstance(detail, dict) or not detail:
        return listing

    listing.condition = str(detail.get("condition", "")).strip() or listing.condition
    listing.condition_id = str(detail.get("conditionId", "")).strip() or listing.condition_id

    short_desc = detail.get("shortDescription")
    if isinstance(short_desc, str) and short_desc.strip():
        listing.short_desc = short_desc.strip()

    shipping_options = detail.get("shippingOptions") or []
    best_shipping: Optional[float] = None

    if isinstance(shipping_options, list):
        for option in shipping_options:
            shipping_cost = eur_value(option.get("shippingCost") or {})
            if shipping_cost is None:
                continue
            if best_shipping is None or shipping_cost < best_shipping:
                best_shipping = shipping_cost

    if best_shipping is not None:
        listing.shipping_eur = float(best_shipping)

    location = extract_location(detail.get("itemLocation") or {})
    if location:
        listing.location_text = location

    listing.category_id = extract_category_id(detail) or listing.category_id
    listing.url = clean_url(listing.url)

    return listing