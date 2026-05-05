#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIMELAB — comparables_engine.py

Hierarchical Catawiki hammer estimator. Reads comparables_db_v2.json and
provides a single function the bot calls for every candidate listing.

Lookup order (first level with a hit wins):
    L1  ref bucket
    L2  brand × family × mech × chrono
    L3  brand × mech × chrono
    L4  brand × mech
    L5  brand

After the bucket is picked, two orthogonal multipliers may apply:
  - tier_multiplier_by_brand[brand][auction_tier]   (always, when known)
  - chrono_multiplier_by_brand_mech[B:M]            ONLY at L4/L5 (where the
    bucket itself doesn't separate chrono).
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_HAIRCUT = 1.00
CONFIDENCE_MIN_N = {1: 4, 2: 6, 3: 8, 4: 12, 5: 20}


def _resolve_default_db_path() -> Path:
    """Try common locations relative to the package and to CWD."""
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "data" / "comparables_db_v2.json",   # repo_root/data/
        here / "data" / "comparables_db_v2.json",          # comparables/data/
        here / "comparables_db_v2.json",                   # alongside engine
        Path("data/comparables_db_v2.json"),               # CWD/data/
        Path("comparables_db_v2.json"),                    # CWD
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]   # noisy fallback (will FileNotFoundError on read)


DEFAULT_DB_PATH = _resolve_default_db_path()


class ComparablesDB:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.db = json.loads(self.path.read_text(encoding="utf-8"))
        self.buckets = self.db.get("buckets", {})
        self._by_ref, self._by_l2, self._by_l3, self._by_l4, self._by_l5 = {}, {}, {}, {}, {}
        for k, v in self.buckets.items():
            L = v["level"]
            if L == 1:
                self._by_ref[v["key"]] = (k, v)
            elif L == 2:
                self._by_l2[(v["brand"], v["family"], v["mechanism"], v["is_chrono"])] = (k, v)
            elif L == 3:
                self._by_l3[(v["brand"], v["mechanism"], v["is_chrono"])] = (k, v)
            elif L == 4:
                self._by_l4[(v["brand"], v["mechanism"])] = (k, v)
            elif L == 5:
                self._by_l5[v["brand"]] = (k, v)
        self.tier_mults = self.db.get("tier_multipliers_by_brand", {})
        self.chrono_mults = self.db.get("chrono_multipliers_by_brand_mech", {})

    def _lookup_ref(self, refs):
        if not refs:
            return None
        for r in refs:
            hit = self._by_ref.get(r)
            if hit:
                return hit
        return None

    def _tier_multiplier(self, brand, auction_tier):
        if not auction_tier or auction_tier in ("?", "standard", None):
            return 1.0, None
        mult = self.tier_mults.get(brand, {}).get(auction_tier)
        if mult is None:
            return 1.0, None
        return float(mult), f"tier[{brand}/{auction_tier}]={mult}"

    def _chrono_multiplier(self, brand, mech, is_chrono, level_used):
        if level_used < 4 or mech == "?":
            return 1.0, None
        rec = self.chrono_mults.get(f"{brand}:{mech}")
        if not rec:
            return 1.0, None
        if is_chrono:
            mult = float(rec["multiplier"])
            return mult, f"chrono[{brand}:{mech}]=×{mult}"
        mult = 1.0 / float(rec["multiplier"]) ** 0.5
        return round(mult, 3), f"non_chrono_correction[{brand}:{mech}]=×{round(mult, 3)}"

    @staticmethod
    def _confidence(level: int, n: int) -> str:
        thr = CONFIDENCE_MIN_N.get(level, 999)
        if n >= thr * 2:
            return "high"
        if n >= thr:
            return "medium"
        return "low"

    def estimate(self,
                 brand: str,
                 mech: str | None = None,
                 is_chrono: bool = False,
                 refs: list | None = None,
                 model_family: str | None = None,
                 auction_tier: str | None = None,
                 haircut: float = DEFAULT_HAIRCUT) -> dict:
        if not brand:
            return self._empty_result("no_brand")

        hit = None
        level_used = None
        for level, fn in [
            (1, lambda: self._lookup_ref(refs)),
            (2, lambda: (
                self._by_l2.get((brand, model_family, mech or "?", bool(is_chrono)))
                if model_family and (mech and mech != "?") else None
            )),
            (3, lambda: (
                self._by_l3.get((brand, mech or "?", bool(is_chrono)))
                if mech and mech != "?" else None
            )),
            (4, lambda: (
                self._by_l4.get((brand, mech or "?"))
                if mech and mech != "?" else None
            )),
            (5, lambda: self._by_l5.get(brand)),
        ]:
            res = fn()
            if res:
                hit = res
                level_used = level
                break

        if hit is None:
            return self._empty_result(f"no_bucket_for[{brand}]")

        bucket_key, bucket = hit
        raw_p50 = float(bucket["p50"])
        raw_p25 = float(bucket["p25"])
        raw_p75 = float(bucket["p75"])
        n = int(bucket["n"])

        applied = {}
        tier_m, tier_label = self._tier_multiplier(brand, auction_tier)
        if tier_label:
            applied["tier"] = {"factor": tier_m, "label": tier_label}
        chrono_m, chrono_label = self._chrono_multiplier(brand, mech or "?", is_chrono, level_used)
        if chrono_label:
            applied["chrono"] = {"factor": chrono_m, "label": chrono_label}

        combined = tier_m * chrono_m
        expected = raw_p50 * combined * haircut
        conservative = raw_p25 * combined * haircut
        optimistic = raw_p75 * combined * haircut

        return {
            "expected_hammer":     round(expected, 2),
            "conservative":        round(conservative, 2),
            "optimistic":          round(optimistic, 2),
            "raw_p50":             round(raw_p50, 2),
            "p25":                 round(raw_p25, 2),
            "p75":                 round(raw_p75, 2),
            "n":                   n,
            "level_used":          level_used,
            "source_bucket":       bucket_key,
            "multipliers_applied": applied,
            "combined_multiplier": round(combined, 4),
            "confidence":          self._confidence(level_used, n),
            "reason":              f"L{level_used}_match",
        }

    @staticmethod
    def _empty_result(reason: str) -> dict:
        return {
            "expected_hammer": None, "conservative": None, "optimistic": None,
            "raw_p50": None, "p25": None, "p75": None, "n": 0,
            "level_used": None, "source_bucket": None,
            "multipliers_applied": {}, "combined_multiplier": 1.0,
            "confidence": "low", "reason": reason,
        }


_GLOBAL_DB = None

def estimate_hammer_catawiki(brand, mech=None, is_chrono=False, refs=None,
                             model_family=None, auction_tier=None,
                             db_path=None):
    global _GLOBAL_DB
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    if _GLOBAL_DB is None or str(_GLOBAL_DB.path) != str(db_path):
        _GLOBAL_DB = ComparablesDB(db_path)
    return _GLOBAL_DB.estimate(brand, mech=mech, is_chrono=is_chrono,
                               refs=refs, model_family=model_family,
                               auction_tier=auction_tier)


if __name__ == "__main__":
    import sys
    db_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/comparables_db_v2.json"
    db = ComparablesDB(db_path)

    cases = [
        ("Tissot Chrono XL Cuarzo Chrono (L2 specific)",
            dict(brand="Tissot", mech="Cuarzo", is_chrono=True, model_family="chrono xl")),
        ("Tissot generic Cuarzo Chrono (no family → L3)",
            dict(brand="Tissot", mech="Cuarzo", is_chrono=True, model_family=None)),
        ("Seiko 5 Auto (L2 specific)",
            dict(brand="Seiko", mech="Automático", is_chrono=False, model_family="seiko 5")),
        ("Seiko Auto generic (no family → L3)",
            dict(brand="Seiko", mech="Automático", is_chrono=False, model_family=None)),
        ("Seiko Cuarzo non-chrono VINTAGE tier (L4 + tier×0.59)",
            dict(brand="Seiko", mech="Cuarzo", is_chrono=False, auction_tier="vintage")),
        ("Seiko Cuarzo Chrono (L3, no tier)",
            dict(brand="Seiko", mech="Cuarzo", is_chrono=True)),
        ("Tissot Cuarzo non-chrono VINTAGE tier",
            dict(brand="Tissot", mech="Cuarzo", is_chrono=False, auction_tier="vintage")),
        ("Longines unknown mech (→ L5 brand)",
            dict(brand="Longines", mech="?", is_chrono=False)),
        ("Hamilton Khaki Auto (Hamilton has only L5)",
            dict(brand="Hamilton", mech="Automático", is_chrono=False, model_family="khaki field")),
        ("Brand not in DB",
            dict(brand="NoSuchBrand", mech="Cuarzo", is_chrono=False)),
    ]

    print(f"DB: {db_path}  ({db.db['n_comps_total']} comps, {len(db.buckets)} buckets)\n")
    print(f"{'Case':<54s}  {'L':>3s}  {'p25':>4s} {'p50':>4s} {'p75':>4s}  "
          f"{'×mult':>6s}  {'expected':>9s}  {'conf':>6s}  bucket")
    print("-" * 132)
    for label, kw in cases:
        r = db.estimate(**kw)
        L = r['level_used'] if r['level_used'] else "-"
        p50 = f"{r['raw_p50']:.0f}" if r['raw_p50'] is not None else "—"
        p25 = f"{r['p25']:.0f}" if r['p25'] is not None else "—"
        p75 = f"{r['p75']:.0f}" if r['p75'] is not None else "—"
        exp = f"{r['expected_hammer']:.0f}€" if r['expected_hammer'] is not None else "—"
        mult = f"×{r['combined_multiplier']}" if r['expected_hammer'] is not None else "—"
        conf = r['confidence']
        bk = r['source_bucket'] or r['reason']
        print(f"{label:<54s}  L{L}  {p25:>4s} {p50:>4s} {p75:>4s}  {mult:>6s}  {exp:>9s}  {conf:>6s}  {bk}")
