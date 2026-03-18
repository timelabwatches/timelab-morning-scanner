# bridge/analyst_pipeline.py

from analyzer.apply_reference_kb import apply_reference_kb
from analyzer.infer_brand import infer_brand
from analyzer.infer_condition import infer_condition
from analyzer.infer_flags import infer_flags
from analyzer.infer_gender import infer_gender
from analyzer.infer_model import infer_model
from analyzer.infer_movement_hint import infer_movement_hint
from analyzer.infer_reference import infer_reference
from analyzer.infer_watch_type import infer_watch_type
from analyzer.score_record import score_record
from decision.decision_engine import apply_decision_engine
from economics.profit_engine import apply_profit_engine
from evaluator.classify_candidate import classify_candidate


def analyze_record(record: dict) -> dict:
    """
    Simplified TIMELAB analyst pipeline for eBay candidates.
    Mirrors the structure of timelab-watch-analyst:
    brand -> reference -> model -> condition/gender/type -> movement ->
    flags -> reference KB -> economics -> score -> candidate class -> decision
    """

    brand = infer_brand(record)
    reference = infer_reference({**record, "brand": brand})

    partial_for_model = {
        **record,
        "brand": brand,
        "reference": reference,
    }

    model = infer_model(partial_for_model, brand)

    partial_for_type = {
        **record,
        "brand": brand,
        "reference": reference,
        "model": model,
    }

    condition = infer_condition(record)
    gender = infer_gender(record)
    watch_type = infer_watch_type(partial_for_type)

    partial = {
        **record,
        "brand": brand,
        "model": model,
        "reference": reference,
        "condition": condition,
        "gender": gender,
        "watch_type": watch_type,
        "analysis_status": "analyzed",
    }

    movement_hint = infer_movement_hint(partial)

    analyzed = {
        **partial,
        "movement_hint": movement_hint,
    }

    analyzed.update(infer_flags(analyzed))

    analyzed = apply_reference_kb(analyzed)

    # Auction estimate may already be provided upstream.
    # If not, keep a safe empty structure so economics/decision do not break.
    if "auction_estimate" not in analyzed:
        analyzed["auction_estimate"] = {
            "price_estimate_available": False,
            "expected_hammer": None,
            "conservative_hammer": None,
            "optimistic_hammer": None,
            "price_confidence": "low",
        }

    analyzed = apply_profit_engine(analyzed)

    analyzed["score"] = score_record(analyzed)
    analyzed["candidate_class"] = classify_candidate(analyzed)

    analyzed = apply_decision_engine(analyzed)

    return analyzed