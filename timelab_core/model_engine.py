import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple

from timelab_core.scoring import norm

MODEL_MASTER_PATH = Path("data/model_master.json")
MARKET_COMP_DB_PATH = Path("data/market_comp_db.json")
TARGET_STATS_PATH = Path("data/target_stats.json")
FEEDBACK_PATH = Path("data/feedback_log.jsonl")


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_model_master(path: Path = MODEL_MASTER_PATH) -> List[Dict]:
    doc = load_json(path, {"models": []})
    models = doc.get("models", []) if isinstance(doc, dict) else []
    return [m for m in models if isinstance(m, dict)]


def load_target_stats(path: Path = TARGET_STATS_PATH) -> Dict[str, Dict]:
    doc = load_json(path, {"target_stats": []})
    out = {}
    for row in doc.get("target_stats", []):
        tid = str(row.get("target_id", "")).strip()
        if tid:
            out[tid] = row
    return out


def extract_references(text: str) -> List[str]:
    t = (text or "").upper().replace("-", " ")
    refs = set()
    patterns = [
        r"\bT\d{3}\.?\d{3}\b",
        r"\bSRP[A-Z]?\d{2,3}\b",
        r"\bSRPK\d{2,3}\b",
        r"\bSNXS\d{2,3}\b",
        r"\b7N32\b",
        r"\b4R36\b",
        r"\bETA\s*G10\b",
    ]
    for pat in patterns:
        for m in re.finditer(pat, t):
            refs.add(m.group(0).replace(" ", ""))
    return sorted(refs)


def _percentile(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    arr = sorted(vals)
    idx = int(round((len(arr) - 1) * p))
    return float(arr[max(0, min(len(arr) - 1, idx))])


def _iqr_filter(vals: List[float]) -> Tuple[List[float], List[float]]:
    if len(vals) < 4:
        return vals[:], []
    arr = sorted(vals)
    q1 = _percentile(arr, 0.25)
    q3 = _percentile(arr, 0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    filt = [v for v in arr if lo <= v <= hi]
    out = [v for v in arr if v < lo or v > hi]
    return filt, out


def rebuild_target_stats(market_path: Path = MARKET_COMP_DB_PATH, out_path: Path = TARGET_STATS_PATH) -> Dict:
    market = load_json(market_path, {"comparables": []})
    comps = [c for c in market.get("comparables", []) if isinstance(c, dict)]
    by_target = defaultdict(list)
    now = datetime.utcnow().date()

    for c in comps:
        tid = str(c.get("target_id", "")).strip()
        price = float(c.get("hammer_or_sale_price", 0) or 0)
        if not tid or price <= 0:
            continue
        date_raw = str(c.get("date_closed", ""))
        try:
            d = datetime.strptime(date_raw, "%Y-%m-%d").date()
        except Exception:
            d = now
        by_target[tid].append((price, d))

    rows = []
    for tid, vals in by_target.items():
        prices = [v for v, _ in vals]
        filtered, outliers = _iqr_filter(prices)
        if not filtered:
            filtered = prices
        p25 = _percentile(filtered, 0.25)
        p50 = _percentile(filtered, 0.50)
        p75 = _percentile(filtered, 0.75)
        p90 = _percentile(filtered, 0.90)

        p90d = [v for v, d in vals if d >= now - timedelta(days=90)]
        p180d = [v for v, d in vals if d >= now - timedelta(days=180)]
        roll90 = _percentile(p90d, 0.50) if p90d else p50
        roll180 = _percentile(p180d, 0.50) if p180d else p50
        trend = round(roll90 - roll180, 2)
        sample = len(filtered)
        conf = max(10, min(95, int(sample * 6 + (10 if sample >= 10 else 0))))

        rows.append({
            "target_id": tid,
            "sample_size": sample,
            "p25": round(p25, 2),
            "p50": round(p50, 2),
            "p75": round(p75, 2),
            "p90": round(p90, 2),
            "rolling_90d_p50": round(roll90, 2),
            "rolling_180d_p50": round(roll180, 2),
            "last_90d_trend": trend,
            "outliers": [round(x, 2) for x in outliers],
            "confidence_score": conf,
            "condition_adjustment_rules": {"excellent": 1.04, "good": 1.0, "worn": 0.9},
            "full_set_adjustment_rules": {"full_set": 1.03, "box_only": 1.01, "papers_only": 1.01},
        })

    doc = {"target_stats": sorted(rows, key=lambda x: x["target_id"]), "generated_at": datetime.utcnow().isoformat()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc


def resolve_listing_identity(text: str, visual_flags: Optional[Dict] = None, models: Optional[List[Dict]] = None) -> Dict:
    t = norm(text)
    refs = extract_references(text)
    models = models if models is not None else load_model_master()

    best = None
    for m in models:
        brand = norm(m.get("brand", ""))
        family = norm(m.get("family", ""))
        model = norm(m.get("model", ""))
        exact_refs = [str(x).upper().replace(" ", "") for x in m.get("reference_exact", [])]
        prefixes = [str(x).upper().replace(" ", "") for x in m.get("reference_prefix", [])]
        must_have = [norm(x) for x in m.get("must_have_keywords", [])]
        negatives = [norm(x) for x in m.get("negative_keywords", [])]

        ref_score = 0
        for r in refs:
            if r in exact_refs:
                ref_score = max(ref_score, 70)
            if any(r.startswith(pf) for pf in prefixes):
                ref_score = max(ref_score, 45)

        keyword_score = 0
        for tok in [brand, family, model] + must_have:
            if tok and tok in t:
                keyword_score += 6
        keyword_score = min(25, keyword_score)

        spec_tokens = ["chronograph", "chrono", "powermatic", "automatic", "quartz", "4r36", "eta g10", "200m", "300m", "snxs", "turtle"]
        spec_score = min(20, sum(4 for tok in spec_tokens if tok in t))

        neg_penalty = 0
        for n in negatives:
            if n and n in t:
                neg_penalty += 22
        # hard rule examples requested
        if "T120417" in refs and m.get("target_id") == "TISSOT_SEASTAR_POWERMATIC80":
            neg_penalty += 80
        if "SRPK87" in refs and m.get("target_id") == "SEIKO_PROSPEX_TURTLE_AUTO":
            neg_penalty += 80
        if "7N32" in refs and "chrono" in norm(m.get("watch_type", "")):
            neg_penalty += 60

        image_score = 0
        visual_flags = visual_flags or {}
        if visual_flags.get("chrono_real") and "chrono" in norm(m.get("watch_type", "")):
            image_score += 12
        if visual_flags.get("diver_real") and "diver" in norm(m.get("watch_type", "")):
            image_score += 10
        if visual_flags.get("conflict"):
            neg_penalty += 18

        final = ref_score + keyword_score + spec_score + image_score - neg_penalty
        row = {
            "model_id": m.get("model_id"),
            "target_id": m.get("target_id"),
            "reference_score": ref_score,
            "keyword_score": keyword_score,
            "spec_score": spec_score,
            "negative_penalty": neg_penalty,
            "image_score": image_score,
            "final_match_score": max(0, min(100, final)),
            "match_confidence_band": "high" if final >= 75 else "medium" if final >= 52 else "low",
            "matched_references": refs,
            "model_ambiguity": final < 60,
            "movement_type": m.get("movement_type", "unknown"),
        }
        if best is None or row["final_match_score"] > best["final_match_score"]:
            best = row

    if not best:
        return {
            "model_id": "UNKNOWN",
            "target_id": None,
            "reference_score": 0,
            "keyword_score": 0,
            "spec_score": 0,
            "negative_penalty": 0,
            "image_score": 0,
            "final_match_score": 0,
            "match_confidence_band": "low",
            "matched_references": refs,
            "model_ambiguity": True,
            "movement_type": "unknown",
        }
    return best


def gate_decision(match_band: str, sample_size: int, valuation_confidence: int, expected_net: float, roi: float, fake_allowed: bool, ambiguity: bool) -> str:
    if ambiguity or match_band == "low":
        return "SKIP"

    if match_band == "high" and valuation_confidence >= 65 and expected_net >= 20 and roi >= 0.08 and fake_allowed:
        return "BUY"

    if expected_net >= 20 and roi >= 0.08 and fake_allowed:
        return "REVIEW"

    if sample_size < 4:
        return "REVIEW"

    return "REVIEW"


def register_feedback(entry: Dict, path: Path = FEEDBACK_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")