# evaluator/classify_candidate.py

def classify_candidate(record: dict) -> str:
    """
    Classify listing as candidate quality level.
    """

    score = record.get("score", 0)

    if score >= 75:
        return "strong_candidate"

    if score >= 45:
        return "review"

    return "weak"