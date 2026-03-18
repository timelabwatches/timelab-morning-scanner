# bridge/analyst_adapter.py

def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(str(value).split()).strip()


def ebay_candidate_to_record(candidate: dict) -> dict:
    """
    Convert an eBay candidate into the record structure expected by the
    TIMELAB analyst engine.
    """

    title = clean_text(candidate.get("title"))
    description = clean_text(candidate.get("description"))
    condition_text = clean_text(candidate.get("condition_text"))
    raw_text = clean_text(candidate.get("raw_text"))
    location = clean_text(candidate.get("location"))
    price = candidate.get("price")

    analysis_text_parts = [
        title,
        description,
        condition_text,
        location,
    ]
    analysis_text = " | ".join(part for part in analysis_text_parts if part)

    return {
        "source": candidate.get("source", "ebay"),
        "listing_id": candidate.get("listing_id"),
        "url": candidate.get("url"),
        "title": title,
        "description": description,
        "condition_text": condition_text,
        "location": location,
        "price": price,
        "shipping": candidate.get("shipping"),
        "category_id": candidate.get("category_id"),
        "raw_text": raw_text or analysis_text,
        "raw_text_clean": raw_text or analysis_text,
        "analysis_text": analysis_text,
        "discovery_context": title,
        "analysis_ready": True,
        "title_found": bool(title),
        "price_found": price is not None,
        "forced_brand_hint": candidate.get("forced_brand_hint"),
    }