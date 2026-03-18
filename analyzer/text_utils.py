def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(str(value).strip().lower().split())


def build_analysis_text(record: dict) -> str:
    """
    Build one normalized analysis text blob from the most relevant fields.
    """

    parts = [
        record.get("brand"),
        record.get("title"),
        record.get("description"),
        record.get("raw_text"),
        record.get("raw_text_clean"),
        record.get("analysis_text"),
        record.get("discovery_context"),
        record.get("condition_text"),
        record.get("location"),
    ]

    cleaned = [clean_text(x) for x in parts if clean_text(x)]
    return " | ".join(cleaned)