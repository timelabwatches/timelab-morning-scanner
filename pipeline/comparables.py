import json
from datetime import datetime, timedelta


def load_comparables(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        rows = raw.get("comparables", [])
    else:
        rows = raw

    return [row for row in rows if isinstance(row, dict)]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * p))
    index = max(0, min(index, len(ordered) - 1))
    return float(ordered[index])


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _parse_date(value: str):
    if not value:
        return None

    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except Exception:
            continue

    return None


def _iqr_filter(values: list[float]) -> tuple[list[float], list[float]]:
    if len(values) < 4:
        return values[:], []

    ordered = sorted(values)
    q1 = _percentile(ordered, 0.25)
    q3 = _percentile(ordered, 0.75)
    iqr = q3 - q1

    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr

    filtered = [v for v in ordered if low <= v <= high]
    outliers = [v for v in ordered if v < low or v > high]
    return filtered, outliers


def get_target_stats(target_id: str, comparables: list[dict]) -> dict:
    target_id = str(target_id or "").strip()
    if not target_id:
        return {
            "target_id": "",
            "sample_size": 0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "rolling_90d_p50": 0.0,
            "rolling_180d_p50": 0.0,
            "last_90d_trend": 0.0,
            "confidence_score": 0,
            "outliers": [],
        }

    now = datetime.utcnow().date()
    matched = []

    for row in comparables:
        row_target = str(row.get("target_id", "")).strip()
        if row_target != target_id:
            continue

        price = _safe_float(row.get("hammer_or_sale_price", 0))
        if price <= 0:
            continue

        closed_date = _parse_date(str(row.get("date_closed", "")).strip())
        matched.append(
            {
                "price": price,
                "date_closed": closed_date,
            }
        )

    prices = [row["price"] for row in matched]
    filtered, outliers = _iqr_filter(prices)
    if not filtered:
        filtered = prices[:]

    p25 = _percentile(filtered, 0.25) if filtered else 0.0
    p50 = _percentile(filtered, 0.50) if filtered else 0.0
    p75 = _percentile(filtered, 0.75) if filtered else 0.0
    p90 = _percentile(filtered, 0.90) if filtered else 0.0

    prices_90d = [
        row["price"]
        for row in matched
        if row["date_closed"] and row["date_closed"] >= now - timedelta(days=90)
    ]
    prices_180d = [
        row["price"]
        for row in matched
        if row["date_closed"] and row["date_closed"] >= now - timedelta(days=180)
    ]

    rolling_90d_p50 = _percentile(prices_90d, 0.50) if prices_90d else p50
    rolling_180d_p50 = _percentile(prices_180d, 0.50) if prices_180d else p50
    last_90d_trend = round(rolling_90d_p50 - rolling_180d_p50, 2)

    sample_size = len(filtered)

    confidence_score = 0
    if sample_size >= 10:
        confidence_score = 90
    elif sample_size >= 7:
        confidence_score = 75
    elif sample_size >= 4:
        confidence_score = 60
    elif sample_size >= 2:
        confidence_score = 40
    elif sample_size == 1:
        confidence_score = 20

    return {
        "target_id": target_id,
        "sample_size": sample_size,
        "p25": round(p25, 2),
        "p50": round(p50, 2),
        "p75": round(p75, 2),
        "p90": round(p90, 2),
        "rolling_90d_p50": round(rolling_90d_p50, 2),
        "rolling_180d_p50": round(rolling_180d_p50, 2),
        "last_90d_trend": last_90d_trend,
        "confidence_score": confidence_score,
        "outliers": [round(x, 2) for x in outliers],
    }