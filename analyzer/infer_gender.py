# analyzer/infer_gender.py

from analyzer.text_utils import build_analysis_text


def infer_gender(record: dict) -> str | None:
    """
    Infer watch gender from record text.
    """

    text = build_analysis_text(record)

    if "señora" in text or "senora" in text or "lady" in text:
        return "female"

    if "caballero" in text or "hombre" in text or "gent" in text:
        return "male"

    if "unisex" in text:
        return "unisex"

    return None