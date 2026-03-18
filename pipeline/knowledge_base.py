import json
import re


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def extract_references(text: str) -> list[str]:
    t = (text or "").upper().replace("-", " ")
    refs = set()

    patterns = [
        r"\bT\d{3}\.?\d{3}\.?\d{2}\.?\d{3}\.?\d{2}\b",
        r"\bT\d{3}\.?\d{3}\b",
        r"\bSRP[A-Z]?\d{2,3}\b",
        r"\bSRPK\d{2,3}\b",
        r"\bSNXS\d{2,3}\b",
        r"\bSKX\d{3}\b",
        r"\bSPB\d{3}[A-Z]?\b",
        r"\bSARB\d{3}\b",
        r"\bL\d\.\d{3}\.\d\b",
        r"\bIW\d{5,6}\b",
        r"\bH\d{6,8}\b",
        r"\bWAF\d{4}[A-Z]?\b",
        r"\bWAY\d{4}[A-Z]?\b",
        r"\bWAZ\d{4}[A-Z]?\b",
        r"\bCAW\d{4}[A-Z]?\b",
        r"\bCV\d{4}[A-Z]?\b",
        r"\b\d{3}\.\d{4}\b",
        r"\b\d{3}\.\d{3}\.\d{2}\.\d{3}\.\d{2}\b",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, t):
            refs.add(match.group(0).replace(" ", ""))

    return sorted(refs)


def load_model_master(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        models = raw.get("models", [])
    else:
        models = raw

    return [row for row in models if isinstance(row, dict)]


def _score_model(text: str, refs: list[str], model: dict) -> dict:
    t = norm(text)

    brand = norm(model.get("brand", ""))
    family = norm(model.get("family", ""))
    model_name = norm(model.get("model", ""))
    target_id = str(model.get("target_id", "")).strip()

    exact_refs = [str(x).upper().replace(" ", "") for x in model.get("reference_exact", [])]
    prefix_refs = [str(x).upper().replace(" ", "") for x in model.get("reference_prefix", [])]
    must_have = [norm(x) for x in model.get("must_have_keywords", [])]
    negative = [norm(x) for x in model.get("negative_keywords", [])]

    keyword_score = 0
    matched_keywords = []

    for token in [brand, family, model_name]:
        if token and token in t:
            keyword_score += 10
            matched_keywords.append(token)

    for token in must_have:
        if token and token in t:
            keyword_score += 6
            matched_keywords.append(token)

    reference_score = 0
    matched_references = []

    for ref in refs:
        if ref in exact_refs:
            reference_score = max(reference_score, 70)
            matched_references.append(ref)
        elif any(ref.startswith(prefix) for prefix in prefix_refs):
            reference_score = max(reference_score, 45)
            matched_references.append(ref)

    negative_penalty = 0
    for token in negative:
        if token and token in t:
            negative_penalty += 20

    final_score = max(0, min(100, keyword_score + reference_score - negative_penalty))

    if final_score >= 75:
        band = "high"
    elif final_score >= 52:
        band = "medium"
    elif final_score >= 30:
        band = "low"
    else:
        band = "very_low"

    return {
        "target_id": target_id,
        "brand": model.get("brand", ""),
        "family": model.get("family", ""),
        "model": model.get("model", ""),
        "match_score": final_score,
        "match_confidence_band": band,
        "matched_keywords": matched_keywords,
        "matched_references": matched_references,
        "negative_penalty": negative_penalty,
    }


def resolve_listing_identity(text: str, models: list[dict]) -> dict:
    refs = extract_references(text)

    best = None
    second = None

    for model in models:
        scored = _score_model(text, refs, model)

        if best is None or scored["match_score"] > best["match_score"]:
            second = best
            best = scored
        elif second is None or scored["match_score"] > second["match_score"]:
            second = scored

    if best is None:
        return {
            "target_id": None,
            "brand": "",
            "family": "",
            "model": "",
            "match_score": 0,
            "match_confidence_band": "very_low",
            "matched_keywords": [],
            "matched_references": refs,
            "negative_penalty": 0,
            "ambiguity": True,
        }

    ambiguity = False
    if second is not None and second["match_score"] > 0:
        ambiguity = second["match_score"] >= best["match_score"] * 0.85

    best["ambiguity"] = ambiguity

    if not best["matched_references"]:
        best["matched_references"] = refs

    return best