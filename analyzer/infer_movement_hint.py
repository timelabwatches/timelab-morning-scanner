from analyzer.text_utils import build_analysis_text


MOVEMENT_KEYWORDS = {
    "quartz": [
        "quartz",
        "cuarzo",
        "battery",
        "pile",
        "eco-drive",
        "eco drive",
    ],
    "automatic": [
        "automatic",
        "automático",
        "automatico",
        "self winding",
        "self-winding",
        "powermatic",
        "co-axial",
        "co axial",
    ],
    "manual": [
        "manual",
        "hand-wound",
        "hand wound",
        "handwound",
        "carica manuale",
        "remontage manuel",
        "handaufzug",
    ],
}


def infer_movement_hint(record: dict) -> str | None:
    """
    Infer movement type from text.
    """

    text = build_analysis_text(record)

    for movement, keywords in MOVEMENT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                return movement

    return None