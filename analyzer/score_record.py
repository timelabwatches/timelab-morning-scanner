# analyzer/score_record.py

def score_record(record: dict) -> int:
    """
    Lightweight quality/confidence score for analyzed record.
    This mirrors the role of the analyst repo: score = confidence/quality,
    not the final economic decision.
    """

    score = 0

    if record.get("brand"):
        score += 15

    if record.get("reference"):
        score += 20

    if record.get("model"):
        score += 15

    if record.get("watch_type"):
        score += 10

    if record.get("movement_hint"):
        score += 10

    if record.get("condition") and record.get("condition") != "unknown":
        score += 10

    if record.get("reference_kb_hit"):
        score += 25

    if record.get("price") is not None:
        score += 5

    return min(score, 100)