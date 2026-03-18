# decision/decision_engine.py

def safe_get_expected_case(record: dict) -> dict:
    """
    Return expected economics case safely.
    """

    economics = record.get("economics") or {}
    expected_case = economics.get("expected_case") or {}

    return expected_case


def make_decision(record: dict) -> dict:
    """
    Final decision layer for TIMELAB.

    This is different from score/candidate_class:
    - score = quality / confidence of the expediente
    - decision = economic/actionable outcome

    Output decisions:
    - strong_buy
    - buy
    - review
    - pass
    """

    reference_kb_hit = bool(record.get("reference_kb_hit"))
    auction_estimate = record.get("auction_estimate") or {}
    economics = record.get("economics") or {}

    price_estimate_available = bool(auction_estimate.get("price_estimate_available"))
    price_confidence = auction_estimate.get("price_confidence") or "low"

    economics_available = bool(economics.get("economics_available"))
    expected_case = safe_get_expected_case(record)

    net_profit = expected_case.get("net_profit")
    roi = expected_case.get("roi")

    score = record.get("score", 0)
    is_ladies = bool(record.get("is_ladies"))

    if is_ladies:
        return {
            "decision": "pass",
            "decision_reason": "ladies_watch_penalty",
        }

    if not price_estimate_available or not economics_available:
        if score >= 75:
            return {
                "decision": "review",
                "decision_reason": "good_listing_but_no_pricing_data",
            }

        return {
            "decision": "pass",
            "decision_reason": "no_pricing_data",
        }

    if net_profit is None or roi is None:
        return {
            "decision": "review",
            "decision_reason": "economics_incomplete",
        }

    if net_profit <= 0 or roi <= 0:
        return {
            "decision": "pass",
            "decision_reason": "negative_expected_value",
        }

    if (
        reference_kb_hit
        and price_confidence in ["medium", "high"]
        and net_profit >= 150
        and roi >= 0.80
    ):
        return {
            "decision": "strong_buy",
            "decision_reason": "high_confidence_high_profit",
        }

    if reference_kb_hit and net_profit >= 150 and roi >= 1.00:
        return {
            "decision": "strong_buy",
            "decision_reason": "very_high_edge_reference_match",
        }

    if reference_kb_hit and net_profit >= 100 and roi >= 0.40:
        return {
            "decision": "buy",
            "decision_reason": "positive_reference_backed_edge",
        }

    if net_profit >= 60 and roi >= 0.20:
        return {
            "decision": "review",
            "decision_reason": "positive_but_needs_manual_review",
        }

    return {
        "decision": "pass",
        "decision_reason": "insufficient_edge",
    }


def apply_decision_engine(record: dict) -> dict:
    """
    Apply final decision layer to record.
    """

    result = make_decision(record)

    record["decision"] = result["decision"]
    record["decision_reason"] = result["decision_reason"]

    return record