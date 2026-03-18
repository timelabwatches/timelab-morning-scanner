# analyzer/infer_condition.py

from analyzer.text_utils import build_analysis_text


CONDITION_MAP = [
    ("nos", "excellent"),
    ("new old stock", "excellent"),
    ("new with box and papers", "excellent"),
    ("new with box", "excellent"),
    ("new without box", "very_good"),
    ("nuevo con caja y documentación", "excellent"),
    ("nuevo con caja", "excellent"),
    ("nuevo sin caja", "very_good"),
    ("mint", "excellent"),
    ("excellent", "excellent"),
    ("very good", "very_good"),
    ("muy buen estado", "very_good"),
    ("très bon état", "very_good"),
    ("good", "good"),
    ("buen estado", "good"),
    ("working", "good"),
    ("running", "good"),
    ("used", "good"),
    ("usato", "good"),
    ("occasion", "good"),
    ("acceptable", "fair"),
    ("accettabile", "fair"),
    ("fair", "fair"),
    ("poor", "poor"),
    ("for parts", "poor"),
    ("parts only", "poor"),
    ("not working", "poor"),
    ("broken", "poor"),
]


def infer_condition(record: dict) -> str:
    """
    Infer listing condition from aggregated text.
    """

    text = build_analysis_text(record)

    for keyword, condition in CONDITION_MAP:
        if keyword in text:
            return condition

    return "unknown"