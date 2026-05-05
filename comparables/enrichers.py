#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIMELAB — enrichers.py

Two enrichers used by the comparables DB builder and the lookup engine:

1) infer_model_family(brand, text) → str | None
   Per-brand alias table mapping fragments of an object name to a canonical
   family token. Longest-match wins. Output is lowercase.

2) infer_mechanism_from_refs(refs, brand_hint=None) → str | None
   Looks up known calibre/reference patterns (7S26 → Automático, 7T92 →
   Cuarzo, T120.417 → Cuarzo, etc.). Falls back to the bot's existing
   reference_knowledge_base.json when available.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# ---------- MODEL FAMILY ----------

# Per brand: list of (canonical, [aliases]). Longest alias is matched first.
# Aliases are matched as substrings (case-insensitive, normalized whitespace).
_MODEL_FAMILY_TABLE: dict[str, list[tuple[str, list[str]]]] = {
    "Tissot": [
        ("prc 200",       ["prc 200", "prc200", "prs 200", "prs200"]),
        ("prs 516",       ["prs 516", "prs516"]),
        ("pr 100",        ["pr 100", "pr100", "pr-100"]),
        ("prx",           ["prx"]),
        ("seastar",       ["seastar", "sea star", "sea-star"]),
        ("le locle",      ["le locle", "lelocle"]),
        ("visodate",      ["visodate"]),
        ("t-race",        ["t-race", "t race", "trace"]),
        ("carson",        ["carson"]),
        ("couturier",     ["couturier"]),
        ("tradition",     ["tradition"]),
        ("gentleman",     ["gentleman"]),
        ("t-lord",        ["t-lord", "t lord"]),
        ("chrono xl",     ["chrono xl", "chronograph xl"]),
        ("v8",            ["v8 chronograph", " v8 ", " v8"]),
        ("f1",            ["f1 chronograph", " f1 ", " f1"]),
        ("everytime",     ["everytime"]),
        ("heritage",      ["heritage"]),
        ("ballade",       ["ballade"]),
        ("classique",     ["classique"]),
        ("t-classic",     ["t-classic", "t classic"]),
        ("dream",         ["dream"]),
        ("stylist",       ["stylist"]),
    ],
    "Seiko": [
        ("prospex",          ["prospex"]),
        ("presage",          ["presage"]),
        ("astron",           ["astron"]),
        ("seiko 5 sports",   ["seiko 5 sports", "5 sports"]),
        ("seiko 5",          ["seiko 5", "seiko5"]),
        ("supersport chrono",["supersport chronograph", "supersport"]),
        ("startimer",        ["startimer"]),
        ("velatura",         ["velatura"]),
        ("sportura",         ["sportura"]),
        ("recraft",          ["recraft"]),
        ("premier",          ["premier"]),
        ("kinetic diver",    ["kinetic diver", "kinetic scuba"]),
        ("chronograph 100m", ["chronograph 100m", "chrono 100m"]),
        ("chronograph alarm",["chronograph alarm"]),
        ("racing chrono",    ["racing chronograph"]),
        # Vintage / refs without family — caliber-anchored buckets
        ("vintage diver",    ["king diver"]),
    ],
    "Longines": [
        ("hydroconquest",  ["hydroconquest", "hydro conquest"]),
        ("conquest",       ["conquest"]),
        ("flagship",       ["flagship"]),
        ("dolcevita",      ["dolcevita", "dolce vita"]),
        ("master",         ["master collection", "mastercollection"]),
        ("evidenza",       ["evidenza"]),
        ("admiral",        ["admiral"]),
        ("comet",          ["comet"]),
        ("heritage",       ["heritage"]),
        ("spirit",         ["spirit"]),
        ("calatrava",      ["calatrava"]),
    ],
    "Hamilton": [
        ("khaki field",      ["khaki field"]),
        ("khaki navy",       ["khaki navy", "khaki scuba"]),
        ("khaki diver",      ["khaki diver"]),
        ("khaki pilot",      ["khaki pilot"]),
        ("khaki regatta",    ["khaki regatta"]),
        ("khaki",            ["khaki"]),
        ("jazzmaster thinline", ["jazzmaster thinline"]),
        ("jazzmaster viewmatic",["jazzmaster viewmatic"]),
        ("jazzmaster",       ["jazzmaster", "jazz master"]),
        ("ventura",          ["ventura"]),
        ("classima",         ["classima"]),
        ("american classic", ["american classic"]),
    ],
    "Omega": [
        ("seamaster",     ["seamaster"]),
        ("speedmaster",   ["speedmaster"]),
        ("constellation", ["constellation"]),
        ("de ville",      ["de ville", "deville"]),
        ("geneve",        ["geneve", "genève"]),
        ("dynamic",       ["dynamic"]),
        ("electronic f300", ["electronic f300", "f300", "tuning fork"]),
    ],
    "TAG Heuer": [
        ("aquaracer", ["aquaracer", "aqua racer"]),
        ("carrera",   ["carrera"]),
        ("formula 1", ["formula 1", "formula1"]),
        ("monaco",    ["monaco"]),
        ("kirium",    ["kirium"]),
        ("link",      ["link"]),
    ],
    "Zenith": [
        ("el primero", ["el primero", "elprimero"]),
        ("calatrava",  ["calatrava"]),
        ("stellina",   ["stellina"]),
        ("defy",       ["defy"]),
        ("pilot",      ["pilot"]),
    ],
    "Junghans": [
        ("max bill",   ["max bill", "maxbill"]),
        ("trilastic",  ["trilastic"]),
        ("meister",    ["meister"]),
        ("milano",     ["milano"]),
    ],
    "Certina": [
        ("ds action", ["ds action", "ds-action"]),
        ("ds podium", ["ds podium", "ds-podium"]),
        ("ds first",  ["ds first", "ds-first", "dsfirst"]),
        ("ds spel",   ["ds spel", "ds-spel"]),
        ("ds-1",      ["ds-1", "ds 1"]),
        ("ds-2",      ["ds-2", "ds 2"]),
        ("ds",        ["ds"]),
        ("tank",      ["tank"]),
    ],
    "Oris": [
        ("big crown", ["big crown"]),
        ("aquis",     ["aquis"]),
        ("artelier",  ["artelier"]),
        ("diver",     ["diver"]),
        ("sport",     ["sport"]),
    ],
    "Mortima": [
        ("superdatomatic diver", ["superdatomatic diver"]),
        ("superdatomatic",       ["superdatomatic", "super datomatic"]),
    ],
    "Cauny": [
        ("centenario", ["centenario"]),
        ("prima",      ["prima"]),
        ("calendario", ["calendario"]),
    ],
    "Baume & Mercier": [
        ("classima", ["classima"]),
        ("riviera",  ["riviera"]),
        ("hampton",  ["hampton"]),
        ("capeland", ["capeland"]),
        ("clifton",  ["clifton"]),
        ("geneve",   ["geneve", "genève"]),
        ("landeron", ["landeron"]),  # vintage chrono caliber, family-like signal
    ],
    "Citizen": [
        ("eco-drive promaster", ["promaster eco-drive", "eco-drive promaster"]),
        ("promaster",  ["promaster"]),
        ("eco-drive",  ["eco-drive", "eco drive"]),
        ("marinaut",   ["marinaut"]),
    ],
    "Rado": [
        ("diastar",  ["diastar", "diastar original"]),
        ("jubile",   ["jubile", "jubilé"]),
        ("ceramica", ["ceramica"]),
    ],
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def infer_model_family(brand: str, text: str) -> str | None:
    """Return canonical family token (lowercase) or None.
    Matches the longest alias first to avoid 'pr' eating 'prx'."""
    if not brand or not text:
        return None
    table = _MODEL_FAMILY_TABLE.get(brand)
    if not table:
        return None
    t = " " + _norm(text) + " "
    # Build (alias, canonical) list and sort by alias length desc
    pairs = []
    for canonical, aliases in table:
        for a in aliases:
            pairs.append((a.lower(), canonical))
    pairs.sort(key=lambda x: -len(x[0]))
    for alias, canonical in pairs:
        if (" " + alias + " ") in t or (" " + alias) in t.rstrip() or (alias + " ") in t.lstrip():
            return canonical
    return None


# ---------- REF → MECHANISM ----------

# Manual table of caliber/reference prefixes whose movement is unambiguous.
# Keys are the prefix or full ref (lowercase, no separators).
# We'll match by `ref starts with key` after normalizing.
_REF_TO_MECH = {
    # ── Seiko quartz chronograph movements ──
    "7t92": "Cuarzo", "7t62": "Cuarzo", "7t32": "Cuarzo", "7t82": "Cuarzo",
    "7t42": "Cuarzo", "8t63": "Cuarzo", "4t57": "Cuarzo",
    # ── Seiko quartz (non-chrono) ──
    "7n42": "Cuarzo", "7n43": "Cuarzo", "5y23": "Cuarzo", "v145": "Cuarzo",
    "v158": "Cuarzo", "v172": "Cuarzo",
    # ── Seiko automatic ──
    "7s26": "Automático", "7s36": "Automático", "7s35": "Automático",
    "4r35": "Automático", "4r36": "Automático", "4r37": "Automático",
    "6r15": "Automático", "6r35": "Automático", "6r27": "Automático",
    "6308": "Automático", "6306": "Automático", "6309": "Automático",
    "6105": "Automático", "6106": "Automático", "6119": "Automático",
    "6138": "Automático", "6139": "Automático",  # vintage chrono auto
    "5606": "Automático", "5126": "Automático", "5719": "Automático",
    "7005": "Automático", "7006": "Automático", "7019": "Automático",
    "7625": "Automático", "7626": "Automático", "8205": "Automático",
    "5y23": "Cuarzo",
    # ── Tissot quartz refs ──
    "t048": "Cuarzo", "t115": "Cuarzo", "t116": "Cuarzo",
    "t120417": "Cuarzo",  # chrono quartz family
    "t127410": "Cuarzo",  # Gentleman quartz steel
    "t095417": "Cuarzo",
    "t039": "Cuarzo",
    "t106": "Cuarzo",
    # ── Tissot automatic refs ──
    "t086": "Automático",  # Heritage automatic
    "t41": "Automático",   # Le Locle automatic family (T41.x)
    "t006": "Automático",  # Le Locle automatic
    "t137407": "Automático",  # PRX automatic
    "t085": "Automático",  # Powermatic 80 family
    # ── Hamilton ──
    "h70": "Automático",  # Khaki Field auto family
    "h64": "Cuarzo",      # most quartz Hamiltons
    "h82": "Cuarzo",      # Khaki Navy quartz chrono usually
    # ── Omega ──
    "f300": "Electrónico",
    "esa9162": "Electrónico", "esa 9162": "Electrónico",
    # ── Longines ──
    "l890": "Automático", "l688": "Automático", "l633": "Automático",
}


def _normalize_ref(ref: str) -> str:
    return re.sub(r"[\s\-\.]", "", (ref or "").lower())


# Cache for reference_knowledge_base.json content
_KB_CACHE: dict | None = None


def _load_kb(kb_path: Path | str | None = None) -> dict:
    """Lazy-load the bot's reference KB once. Tries common repo locations."""
    global _KB_CACHE
    if _KB_CACHE is not None:
        return _KB_CACHE
    if kb_path is None:
        here = Path(__file__).resolve().parent
        candidates = [
            here.parent / "knowledge_base" / "reference_knowledge_base.json",
            here / "knowledge_base" / "reference_knowledge_base.json",
            Path("knowledge_base/reference_knowledge_base.json"),
        ]
        for c in candidates:
            if c.exists():
                kb_path = c
                break
    if kb_path is None or not Path(kb_path).exists():
        _KB_CACHE = {}
        return _KB_CACHE
    try:
        _KB_CACHE = json.loads(Path(kb_path).read_text(encoding="utf-8"))
    except Exception:
        _KB_CACHE = {}
    return _KB_CACHE


_KB_MECH_TO_ES = {
    "automatic": "Automático",
    "quartz":    "Cuarzo",
    "kinetic":   "Cuarzo",     # solar/kinetic quartz from market behaviour
    "manual":    "Cuerda",
    "hand-wound":"Cuerda",
}


def infer_mechanism_from_refs(refs: list[str], brand_hint: str | None = None,
                              kb_path: Path | str | None = None) -> str | None:
    """
    Try multiple lookups:
      1) manual _REF_TO_MECH (exact normalized match, then prefix)
      2) bot's reference_knowledge_base.json
    Returns 'Cuarzo' / 'Automático' / 'Cuerda' / 'Electrónico' / None.
    """
    if not refs:
        return None
    # Normalize refs and check manual table
    normalized = [_normalize_ref(r) for r in refs]
    for n in normalized:
        if n in _REF_TO_MECH:
            return _REF_TO_MECH[n]
        # prefix match: try shrinking from the end
        for L in range(len(n), 2, -1):
            prefix = n[:L]
            if prefix in _REF_TO_MECH:
                return _REF_TO_MECH[prefix]

    # KB lookup
    kb = _load_kb(kb_path)
    if kb:
        for n in normalized:
            entry = kb.get(n)
            if entry:
                mh = (entry.get("movement_hint") or "").lower().strip()
                if mh in _KB_MECH_TO_ES:
                    return _KB_MECH_TO_ES[mh]
                # KB might have watch_type chronograph but still no movement
                # — leave as None to allow downstream backoff
    return None


# ---------- FAMILY → MECHANISM ----------
#
# Two layers of defaults, used as last-resort fallback when:
#   (1) the analyzer's text-based movement detection failed, AND
#   (2) the ref→mech lookup also failed.
#
# Layer A: data-driven defaults learned from your real cierres
#   - auto-generated by build_comparables_db.py, stored in comparables_db_v2.json
#     under key "family_mech_defaults"
#   - only published when n>=3 cierres AND 100% share the same mechanism
#   - re-learned every time the DB is rebuilt → self-improving over time
#
# Layer B: industry-canonical defaults (this list)
#   - hardcoded for cases where the watch industry consensus is overwhelming
#     but you don't have enough cierres yet
#   - each entry justified inline; only added when a wrong default would
#     barely move the price (because the alternative mech is rare or close
#     in market value)
#
# Lookup order: layer A first (your data), layer B second (industry).
# Returns None if nothing matches → caller falls back to L5 brand bucket.

# Layer B — industry canonical defaults. Add entries here ONLY when:
#   - you can name the dominant caliber/movement family
#   - the alternative mechanism is rare AND wouldn't catastrophically misprice
_FAMILY_MECH_INDUSTRY_DEFAULTS: dict[tuple[str, str], str] = {
    # Seiko — modern Prospex line is overwhelmingly automatic (4R/6R calibers).
    # Solar variants exist (PADI Turtle Solar) but are minority.
    ("Seiko", "prospex"):           "Automático",
    ("Seiko", "seiko 5 sports"):    "Automático",  # by definition automatic

    # Hamilton — modern Khaki Field is universally automatic (H70 calibre).
    # Vintage hand-wound Khakis exist but are <10% of secondary market.
    ("Hamilton", "khaki field"):    "Automático",
    ("Hamilton", "khaki navy"):     "Automático",
    ("Hamilton", "jazzmaster"):     "Automático",
    ("Hamilton", "jazzmaster viewmatic"):  "Automático",
    ("Hamilton", "jazzmaster thinline"):   "Automático",
    ("Hamilton", "ventura"):        "Cuarzo",       # post-1988 Ventura is quartz

    # Longines — modern automatic-only sport/dress lines.
    ("Longines", "spirit"):         "Automático",
    ("Longines", "hydroconquest"):  "Automático",
    ("Longines", "master"):         "Automático",   # Master Collection = automatic only

    # Tissot — Le Locle line is automatic-only since 2000s.
    ("Tissot", "le locle"):         "Automático",

    # Zenith — El Primero IS the automatic chronograph caliber.
    # Defy and Pilot lines are automatic-only in modern catalogue.
    ("Zenith", "el primero"):       "Automático",
    ("Zenith", "defy"):             "Automático",
    ("Zenith", "pilot"):            "Automático",
}


# Cache for the data-driven defaults loaded from comparables_db_v2.json
_FAMILY_MECH_DATA_CACHE: dict | None = None


def _load_family_mech_data_defaults(db_path: Path | str | None = None) -> dict:
    """Lazy-load `family_mech_defaults` from the comparables DB JSON.
    Returns a dict mapping 'Brand:family' → {mechanism, n, source}."""
    global _FAMILY_MECH_DATA_CACHE
    if _FAMILY_MECH_DATA_CACHE is not None:
        return _FAMILY_MECH_DATA_CACHE
    if db_path is None:
        here = Path(__file__).resolve().parent
        candidates = [
            here.parent / "data" / "comparables_db_v2.json",
            here / "data" / "comparables_db_v2.json",
            here / "comparables_db_v2.json",
            Path("data/comparables_db_v2.json"),
            Path("comparables_db_v2.json"),
        ]
        for c in candidates:
            if c.exists():
                db_path = c
                break
    if db_path is None or not Path(db_path).exists():
        _FAMILY_MECH_DATA_CACHE = {}
        return _FAMILY_MECH_DATA_CACHE
    try:
        db = json.loads(Path(db_path).read_text(encoding="utf-8"))
        _FAMILY_MECH_DATA_CACHE = db.get("family_mech_defaults", {}) or {}
    except Exception:
        _FAMILY_MECH_DATA_CACHE = {}
    return _FAMILY_MECH_DATA_CACHE


def infer_mechanism_from_family(brand: str, family: str | None,
                                db_path: Path | str | None = None) -> tuple[str | None, str]:
    """Returns (mechanism, source). source ∈ {'data_driven', 'industry', 'none'}.
    Caller can use `source` for logging/audit.
    """
    if not brand or not family:
        return None, "none"

    # Layer A — data-driven (your real cierres)
    data_defaults = _load_family_mech_data_defaults(db_path)
    key = f"{brand}:{family}"
    rec = data_defaults.get(key)
    if rec and rec.get("mechanism"):
        return rec["mechanism"], "data_driven"

    # Layer B — industry canonical
    industry = _FAMILY_MECH_INDUSTRY_DEFAULTS.get((brand, family))
    if industry:
        return industry, "industry"

    return None, "none"


# Quick self-test when run directly
if __name__ == "__main__":
    cases = [
        ("Tissot", "Tissot - PRX - 2010-2020", ["T137410"]),
        ("Tissot", "Tissot - Seastar Automatic - 2010-2020", []),
        ("Seiko",  "Seiko - 7S26 Automatic - 1970-1979", ["7S26"]),
        ("Seiko",  "Seiko - Chronograph 100M - 7T92 - 2000-2010", ["7T92"]),
        ("Hamilton", "Hamilton - Khaki Field H70455", ["H70"]),
        ("Omega", "Omega - Electronic f300 Hz Chronometer", []),
    ]
    for brand, text, refs in cases:
        fam = infer_model_family(brand, text)
        mech_refs = infer_mechanism_from_refs(refs, brand_hint=brand)
        mech_fam, src = infer_mechanism_from_family(brand, fam)
        print(f"{brand:10s} | text='{text}'")
        print(f"   family={fam!r}")
        print(f"   mech via refs   = {mech_refs!r}")
        print(f"   mech via family = {mech_fam!r}  (source={src})")
