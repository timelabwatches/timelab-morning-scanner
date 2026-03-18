import re

from analyzer.text_utils import build_analysis_text


REFERENCE_PATTERNS = [
    r"\b[a-z]\.\d{1,4}\.\d{1,4}\.\d{1,4}\b",   # longines style: l.3.632.1
    r"\b[a-z]\d{5,8}[a-z]?\b",                 # t122410a / t048417 / srpd55
    r"\b\d{4}-\d{4}[a-z]?\b",                  # 7009-876a
    r"\b[a-z]{1,3}\d{2,4}[a-z]?\b",            # generic compact refs
]


def normalize_reference(value: str | None) -> str | None:
    """
    Normalize a reference while preserving readable dots for Longines-like refs.
    """

    if not value:
        return None

    ref = value.strip().lower()
    ref = ref.replace("/", "")
    ref = ref.replace("_", "")
    ref = re.sub(r"\s+", "", ref)

    return ref


def extract_reference_candidates(text: str) -> list[str]:
    """
    Extract possible references from text, ordered by specificity.
    """

    if not text:
        return []

    text = text.lower()
    candidates = []

    for pattern in REFERENCE_PATTERNS:
        for match in re.findall(pattern, text):
            normalized = normalize_reference(match)
            if normalized:
                candidates.append(normalized)

    seen = set()
    result = []

    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)

    return result


def score_reference_candidate(candidate: str, brand: str | None, text: str) -> int:
    """
    Score a reference candidate.
    """

    score = 0

    if "." in candidate:
        score += 30

    if re.search(r"\d", candidate):
        score += 20

    if len(candidate) >= 6:
        score += 15

    if brand == "tissot" and candidate.startswith("t"):
        score += 20

    if brand == "longines" and candidate.startswith("l."):
        score += 20

    if brand == "seiko":
        if re.match(r"^\d{4}-\d{4}[a-z]?$", candidate):
            score += 25
        elif re.match(r"^[a-z]{2,4}\d{3,5}$", candidate):
            score += 20

    if candidate in text:
        score += 10

    return score


def infer_reference(record: dict) -> str | None:
    """
    Infer watch reference from record text using regex + scoring.
    """

    text = build_analysis_text(record)
    brand = record.get("brand")

    candidates = extract_reference_candidates(text)

    if not candidates:
        return None

    ranked = sorted(
        candidates,
        key=lambda x: score_reference_candidate(x, brand, text),
        reverse=True,
    )

    return ranked[0]