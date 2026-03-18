from analyzer.text_utils import build_analysis_text


WATCH_TYPE_KEYWORDS = {
    "chronograph": ["cronografo", "cronógrafo", "chronograph", "chrono"],
    "diver": ["diver", "buceo", "seastar", "seamaster", "hydroconquest", "prospex"],
    "dress": ["visodate", "le locle", "de ville", "flagship", "carson", "tradition", "classic"],
    "sport": ["sportura", "arctura", "t-race", "prx", "conquest"],
}


def infer_watch_type_from_reference(reference: str | None, brand: str | None) -> str | None:
    if not reference or not brand:
        return None

    ref = reference.lower()

    if brand == "tissot":
        if ref.startswith("t048417"):
            return "chronograph"
        if ref.startswith("t122410"):
            return "dress"
        if ref.startswith("t120"):
            return "diver"
        if ref.startswith("t137"):
            return "sport"

    return None


def infer_watch_type(record: dict) -> str | None:
    """
    Infer watch type from reference first, then keywords.
    """

    text = build_analysis_text(record)
    brand = record.get("brand")
    reference = record.get("reference")

    by_reference = infer_watch_type_from_reference(reference, brand)
    if by_reference:
        return by_reference

    for watch_type, keywords in WATCH_TYPE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                return watch_type

    return None
