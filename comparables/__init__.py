"""TIMELAB comparables — Catawiki hammer-price estimator.

Public runtime API:
    from comparables import estimate_hammer_catawiki
    from comparables import infer_model_family, infer_mechanism_from_refs
    from comparables.shadow import shadow_compare, apply_comparables_engine

Build script (offline, run manually each quarter):
    python -m comparables.build_comparables_db --excel ... --catawiki ... --out data/comparables_db_v2.json
"""

from .comparables_engine import (
    ComparablesDB,
    estimate_hammer_catawiki,
    DEFAULT_DB_PATH,
)
from .enrichers import (
    infer_model_family,
    infer_mechanism_from_refs,
    infer_mechanism_from_family,
)
from .shadow import (
    shadow_compare,
    apply_comparables_engine,
    apply_comparables_engine_cc,
)

__all__ = [
    "ComparablesDB",
    "estimate_hammer_catawiki",
    "DEFAULT_DB_PATH",
    "infer_model_family",
    "infer_mechanism_from_refs",
    "infer_mechanism_from_family",
    "shadow_compare",
    "apply_comparables_engine",
    "apply_comparables_engine_cc",
]
