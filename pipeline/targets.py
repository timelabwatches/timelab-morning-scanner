import json
from dataclasses import dataclass
from typing import Optional


def norm(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


@dataclass
class TargetModel:
    key: str
    keywords: list[str]
    refs: list[str]
    tier: str
    fake_risk: str
    catwiki_p50: float
    catwiki_p75: float
    buy_max: float
    query: str
    liquidity: str = "medium"
    p75_triggers_any: list[str] | None = None
    ebay_category_id: Optional[str] = None
    must_include: list[str] | None = None
    must_exclude: list[str] | None = None
    condition_boost_terms: list[str] | None = None
    condition_bad_terms: list[str] | None = None


def load_target_bundle(path: str) -> tuple[list[TargetModel], dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    meta: dict = {}
    if isinstance(raw, dict):
        meta = {
            "version": raw.get("version", ""),
            "notes": raw.get("notes", ""),
        }

    data = raw.get("targets", []) if isinstance(raw, dict) else raw
    if not isinstance(data, list):
        raise ValueError("target_list.json has invalid format")

    global_triggers = []
    if isinstance(raw, dict):
        globals_block = raw.get("globals", {})
        if isinstance(globals_block, dict):
            global_triggers = globals_block.get("p75_triggers_any", []) or []

    targets: list[TargetModel] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        brand = norm(item.get("brand", ""))
        model_keywords = item.get("model_keywords", []) or []

        if not brand or not model_keywords:
            continue

        keywords = [norm(x) for x in model_keywords if norm(x)]
        if brand not in keywords:
            keywords.insert(0, brand)
        else:
            keywords = [brand] + [x for x in keywords if x != brand]

        estimate = item.get("catawiki_estimate") or {}
        p50 = float(estimate.get("p50", 0.0) or 0.0)
        p75 = float(estimate.get("p75", p50) or 0.0)

        if p50 <= 0 and p75 > 0:
            p50 = p75
        if p75 <= 0 and p50 > 0:
            p75 = p50

        query = str(item.get("query", "")).strip()
        if not query:
            query = " ".join(keywords[:4]).strip()

        triggers = item.get("p75_triggers_any")
        if not triggers:
            triggers = global_triggers

        targets.append(
            TargetModel(
                key=str(item.get("id", f"{brand}_{keywords[1] if len(keywords) > 1 else 'model'}")),
                keywords=keywords,
                refs=item.get("refs", []) or [],
                tier=str(item.get("tier", "B")),
                fake_risk=str(item.get("risk", "medium")).lower(),
                catwiki_p50=p50,
                catwiki_p75=p75,
                buy_max=float(item.get("max_buy_eur", 0.0) or 0.0),
                query=query,
                liquidity=str(item.get("liquidity", "medium")),
                p75_triggers_any=[str(x) for x in (triggers or []) if str(x).strip()],
                ebay_category_id=item.get("ebay_category_id") or None,
                must_include=item.get("must_include", []) or [],
                must_exclude=item.get("must_exclude", []) or [],
                condition_boost_terms=item.get("condition_boost_terms", []) or [],
                condition_bad_terms=item.get("condition_bad_terms", []) or [],
            )
        )

    if not targets:
        raise ValueError("No valid targets found in target_list.json")

    return targets, meta