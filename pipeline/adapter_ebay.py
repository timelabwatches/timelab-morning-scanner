from clients.ebay_client import EbayListing


def build_listing_text(listing: EbayListing) -> str:
    parts = [
        listing.title or "",
        listing.short_desc or "",
        listing.condition or "",
    ]
    return " ".join(part.strip() for part in parts if part.strip()).strip()


def ebay_listing_to_candidate(listing: EbayListing) -> dict:
    return {
        "source": "ebay",
        "listing_id": listing.item_id,
        "title": listing.title or "",
        "description": listing.short_desc or "",
        "condition_text": listing.condition or "",
        "price": float(listing.price_eur or 0.0),
        "shipping": float(listing.shipping_eur or 0.0),
        "location": listing.location_text or "",
        "url": listing.url or "",
        "category_id": listing.category_id or "",
        "raw_text": build_listing_text(listing),
    }