from analyzer.text_utils import build_analysis_text


BRANDS = [
    "omega",
    "seiko",
    "tissot",
    "longines",
    "certina",
    "zenith",
    "hamilton",
    "oris",
    "tag heuer",
    "citizen",
    "bulova",
    "rado",
    "breitling",
]


def normalize_brand(value: str | None) -> str | None:
    if not value:
        return None

    value = value.strip().lower()

    if value in BRANDS:
        return value

    return None


def infer_brand(record: dict) -> str | None:
    """
    Infer watch brand from:
    1) forced_brand_hint
    2) analysis text
    """

    forced = normalize_brand(record.get("forced_brand_hint"))
    if forced:
        return forced

    text = build_analysis_text(record)

    for brand in BRANDS:
        if brand in text:
            return brand

    return None