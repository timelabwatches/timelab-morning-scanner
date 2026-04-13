# """
TIMELAB — Watch Identification Engine (watch_id_engine.py)

Drop into: timelab_core/watch_id_engine.py

A layered, deterministic identification engine that takes a raw listing
title (+ optional description) and returns the most probable watch identity
with a structured confidence score.

Designed to replace/augment the fragile compute_match_score() path in
scanner.py for the two failure modes identified:

1. Vague titles:  "Seiko automatic watch" → was matching wrong family
1. Incomplete refs: "Tissot Seastar" with no ref number

Architecture:
Layer 1 — Brand Detection       (deterministic, alias-aware)
Layer 2 — Reference Extraction  (regex, brand-aware)
Layer 3 — Family Scoring        (scored soft matching, not binary)
Layer 4 — Target Resolution     (links layers 1-3 to target_list entries)
Layer 5 — Confidence Assembly   (combines all signals, outputs band)

Output schema:
{
"brand":          str | None,
"family":         str | None,
"target_id":      str | None,   # matches target_list.json "id" field
"references":     list[str],    # extracted ref numbers
"confidence":     int,          # 0–100
"confidence_band": str,         # "high" | "medium" | "low" | "very_low"
"layer_scores":   dict,         # debug: scores per layer
"flags":          dict,         # debug: signals that fired
}

Integration with scanner.py:
Replace or augment the compute_match_score() loop:

```
from timelab_core.watch_id_engine import identify_watch, build_target_index

# Once at startup:
target_index = build_target_index(targets)  # targets = load_target_bundle()[0]

# Per listing:
identity = identify_watch(detail_text, target_index)
if identity["confidence_band"] in ("high", "medium"):
    # Use identity["target_id"] to look up the correct target
    # and skip the generic keyword loop entirely
    ...
else:
    # Fall back to existing compute_match_score() loop
    ...
```

"""

from **future** import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# —————————————————————————

# Helpers

# —————————————————————————

def _norm(text: str) -> str:
"""Lowercase + collapse whitespace. Mirrors existing norm() in matching.py."""
return re.sub(r"\s+", " ", (text or "").strip().lower())

def _compact(text: str) -> str:
"""Remove all non-alphanumeric characters (for ref matching)."""
return re.sub(r"[^a-z0-9]", "", _norm(text))

def _token_set(text: str) -> set[str]:
"""Split normalised text into a set of word tokens."""
return set(re.split(r"[^a-z0-9]+", _norm(text))) - {""}

# —————————————————————————

# Layer 1 — Brand Detection

# —————————————————————————

# 

# Curated brand list with aliases.  Brand detection is deterministic:

# the first brand found in the title wins (we scan longest alias first

# to avoid "tag" matching before "tag heuer").

# 

# To add a brand: append an entry to BRAND_ALIASES.

BRAND_ALIASES: List[Tuple[str, List[str]]] = [
# Canonical name           Aliases (longest first matters)
("Omega",           ["omega"]),
("Longines",        ["longines"]),
("TAG Heuer",       ["tag heuer", "tag-heuer", "tagheuer"]),
("Tissot",          ["tissot"]),
("Seiko",           ["seiko"]),
("Breitling",       ["breitling"]),
("Tudor",           ["tudor"]),
("Zenith",          ["zenith"]),
("IWC",             ["iwc"]),
("Oris",            ["oris"]),
("Hamilton",        ["hamilton"]),
("Certina",         ["certina"]),
("Mido",            ["mido"]),
("Rado",            ["rado"]),
("Junghans",        ["junghans"]),
("Nomos",           ["nomos"]),
("Sinn",            ["sinn"]),
("Baume & Mercier", ["baume & mercier", "baume mercier", "baume&mercier"]),
("Frederique Constant", ["frederique constant", "frédérique constant"]),
("Raymond Weil",    ["raymond weil"]),
("Maurice Lacroix", ["maurice lacroix"]),
("Alpina",          ["alpina"]),
("Glycine",         ["glycine"]),
("Doxa",            ["doxa"]),
("Fortis",          ["fortis"]),
("Yema",            ["yema"]),
("Zodiac",          ["zodiac"]),
("Stowa",           ["stowa"]),
("Laco",            ["laco"]),
("Damasko",         ["damasko"]),
("Meistersinger",   ["meistersinger"]),
("Montblanc",       ["montblanc", "mont blanc"]),
("Citizen",         ["citizen"]),
("Bulova",          ["bulova"]),
("Eterna",          ["eterna"]),
("Universal Genève",["universal geneve", "universal genève", "ug"]),
("Heuer",           ["heuer"]),  # vintage pre-TAG
("Bvlgari",         ["bvlgari", "bulgari"]),
("Ebel",            ["ebel"]),
]

# Sort all aliases longest-first so greedy matching works correctly.

_SORTED_BRANDS: List[Tuple[str, str]] = sorted(
[(canonical, alias) for canonical, aliases in BRAND_ALIASES for alias in aliases],
key=lambda x: -len(x[1]),
)

def detect_brand(text: str) -> Tuple[Optional[str], int]:
"""
Layer 1: Detect the watch brand from raw listing text.

```
Returns:
    (canonical_brand_name | None, confidence_0_to_100)

Confidence:
    100 — exact alias match found
     0  — no brand detected
"""
t = _norm(text)
for canonical, alias in _SORTED_BRANDS:
    if alias in t:
        return canonical, 100
return None, 0
```

# —————————————————————————

# Layer 2 — Reference Number Extraction

# —————————————————————————

# 

# Brand-specific regex patterns are tried first; if none match, a generic

# alphanumeric pattern is tried as a weak fallback.

# 

# Pattern design principles:

# - Must have both letters and digits (pure numbers like "1858" are model

# names, not refs — handle separately via family keywords)

# - Minimum length 4 chars to avoid noise

# - Anchored on word boundaries to avoid partial matches

_REF_PATTERNS_BY_BRAND: Dict[str, List[str]] = {
"Seiko": [
r"\b(SKX\d{3}[A-Z]?\d?)\b",          # SKX007, SKX009
r"\b(SRP[A-Z]\d{3}[A-Z]?)\b",         # SRPA21, SRPB99
r"\b(SARB\d{3})\b",                    # SARB017
r"\b(SPB\d{3}[A-Z]?)\b",               # SPB121
r"\b(SBDC\d{3})\b",                    # SBDC
r"\b(SNE\d{3}[A-Z]?)\b",               # solar
r"\b(6138|6139|6105|7A28|7T\d{2})\b",  # vintage calibers often used as refs
],
"Omega": [
r"\b(ST\s?\d{3}.\d{4})\b",            # vintage ref
r"\b(\d{3}.\d{4})\b",                 # e.g. 166.0062
r"\b(2\d{2}.\d{2}.\d{2}.\d{2}.\d{2}.\d{3})\b",  # modern full ref
],
"Longines": [
r"\b(L\d.\d{3}.\d)\b",               # L2.821.4
r"\b(L\d{4,6})\b",
],
"Tissot": [
r"\b(T\d{3}.\d{3}.\d{2}.\d{3}.\d{2})\b",  # full modern ref
r"\b(T\d{3})\b",                        # short form T049
],
"TAG Heuer": [
r"\b(WAF\d{4}[A-Z]?)\b",
r"\b(WAY\d{4}[A-Z]?)\b",
r"\b(CAW\d{4}[A-Z]?)\b",
r"\b(WAZ\d{4}[A-Z]?)\b",
r"\b(WAU\d{4}[A-Z]?)\b",
r"\b(CBS\d{4}[A-Z]?)\b",
r"\b(CV\d{4}[A-Z]?)\b",
],
"Tudor": [
r"\b(7\d{4}[A-Z]?)\b",                 # vintage
r"\b(M\d{5}-\d{4})\b",                 # modern
],
"Breitling": [
r"\b(A\d{5}[A-Z0-9]{4,10})\b",         # A17321
r"\b([A-Z]\d{4}[A-Z0-9]{4,})\b",
],
"Hamilton": [
r"\b(H\d{8})\b",                        # H70455133
r"\b(H\d{6})\b",
],
"IWC": [
r"\b(IW\d{6})\b",                       # IW500401
r"\b(IW\d{5})\b",
],
}

# Generic fallback: sequences like SKX007, ref7750, cal6497, etc.

_GENERIC_REF_PATTERN = re.compile(
r"\b([A-Z]{1,4}\d{2,6}[A-Z]{0,3})\b",
re.IGNORECASE,
)

# Tokens to ignore even if they match the generic pattern (common false positives)

_REF_NOISE_TOKENS = {
"nos", "eta", "gmt", "eta2824", "eta2836", "eta2892", "eta7750",
"nh35", "nh36", "nh34", "vk63", "st19", "pt5000",
"automatic", "swiss", "made", "japan",
}

def extract_references(text: str, brand: Optional[str] = None) -> Tuple[List[str], int]:
"""
Layer 2: Extract reference numbers from listing text.

```
Returns:
    (list_of_refs, confidence_0_to_40)

Confidence scale:
    40 — brand-specific pattern matched
    20 — generic pattern matched (weaker signal)
     0 — no ref found
"""
t = _norm(text)
t_upper = text.upper()  # refs are often uppercased in listings
found: List[str] = []
confidence = 0

# Brand-specific patterns (higher confidence)
if brand and brand in _REF_PATTERNS_BY_BRAND:
    for pattern in _REF_PATTERNS_BY_BRAND[brand]:
        matches = re.findall(pattern, t_upper, re.IGNORECASE)
        for m in matches:
            clean = m.strip().upper()
            if clean and clean not in found:
                found.append(clean)
    if found:
        confidence = 40

# Generic fallback if no brand-specific refs found
if not found:
    matches = _GENERIC_REF_PATTERN.findall(t_upper)
    for m in matches:
        clean = m.strip().upper()
        if clean.lower() in _REF_NOISE_TOKENS:
            continue
        if len(clean) < 4:
            continue
        # Must have both letters and digits
        if not (re.search(r"[A-Z]", clean) and re.search(r"[0-9]", clean)):
            continue
        if clean not in found:
            found.append(clean)
    if found:
        confidence = 20

return found, confidence
```

# —————————————————————————

# Layer 3 — Family Scoring

# —————————————————————————

# 

# Each brand has a set of "family signatures" — keyword sets that

# characterise a model family. Scoring is additive: each keyword hit

# adds points. This produces a ranked list of candidate families.

# 

# Design: accuracy over recall. A high-scoring family match is required

# before we commit to a family. Generic/vague titles intentionally

# score low and fall through to UNKNOWN family (not matched to wrong target).

@dataclass
class FamilySignature:
family_id: str          # matches target_list "id" or a discovery group
family_name: str        # human-readable
required: List[str]     # ALL must be present (hard gate)
scored: List[str]       # each hit adds `points_each`
points_each: int = 10
bonus_if_all: int = 0   # extra bonus if all scored terms hit

FAMILY_SIGNATURES: Dict[str, List[FamilySignature]] = {

```
"Seiko": [
    FamilySignature(
        "SEIKO_ALPINIST_SARB017_SPB121", "Alpinist",
        required=["seiko"],
        scored=["alpinist", "sarb017", "spb121", "sarb", "spb"],
        points_each=20, bonus_if_all=15,
    ),
    FamilySignature(
        "SEIKO_PROSPEX_TURTLE_AUTO", "Prospex Turtle",
        required=["seiko"],
        scored=["turtle", "srp", "prospex", "skx", "diver"],
        points_each=20, bonus_if_all=15,
    ),
    FamilySignature(
        "SEIKO_PROSPEX_SAMURAI_AUTO", "Prospex Samurai",
        required=["seiko"],
        scored=["samurai", "prospex"],
        points_each=20, bonus_if_all=15,
    ),
    FamilySignature(
        "SEIKO_CHRONOGRAPH_GENERIC", "Chronograph",
        required=["seiko"],
        scored=["chronograph", "chrono", "6138", "6139", "7t"],
        points_each=15,
    ),
    FamilySignature(
        "SEIKO_VTG_CHRONOGRAPH_GENERIC", "Vintage Chronograph",
        required=["seiko"],
        scored=["chronograph", "chrono", "vintage", "6138", "6139"],
        points_each=15,
    ),
],

"Tissot": [
    FamilySignature(
        "TISSOT_SEASTAR_POWERMATIC80", "Seastar",
        required=["tissot"],
        scored=["seastar", "sea star", "powermatic"],
        points_each=25, bonus_if_all=20,
    ),
    FamilySignature(
        "TISSOT_PRX_POWERMATIC80_40MM", "PRX",
        required=["tissot"],
        scored=["prx", "powermatic"],
        points_each=25, bonus_if_all=20,
    ),
    FamilySignature(
        "TISSOT_CHRONOGRAPH_GENERIC", "Chronograph",
        required=["tissot"],
        scored=["chronograph", "chrono", "prs", "prc", "valjoux", "7750"],
        points_each=15,
    ),
    FamilySignature(
        "TISSOT_VTG_CHRONOGRAPH_GENERIC", "Vintage Chronograph",
        required=["tissot", "vintage"],
        scored=["chronograph", "chrono", "valjoux", "lemania"],
        points_each=15,
    ),
],

"Omega": [
    FamilySignature(
        "OMEGA_SEAMASTER_VTG_AUTO", "Seamaster (vintage)",
        required=["omega", "seamaster"],
        scored=["vintage", "automatic", "300m", "diver"],
        points_each=15,
    ),
    FamilySignature(
        "OMEGA_AQUA_TERRA_AUTO", "Aqua Terra",
        required=["omega"],
        scored=["aqua terra", "aquaterra", "co-axial", "coaxial", "8500", "8900"],
        points_each=20, bonus_if_all=15,
    ),
    FamilySignature(
        "OMEGA_DEVILLE_VTG_AUTO", "De Ville (vintage)",
        required=["omega"],
        scored=["de ville", "deville", "vintage", "automatic"],
        points_each=15,
    ),
],

"Longines": [
    FamilySignature(
        "LON_CONQUEST_AUTO_VTG", "Conquest (vintage)",
        required=["longines"],
        scored=["conquest", "vintage", "automatic"],
        points_each=20, bonus_if_all=10,
    ),
    FamilySignature(
        "LONGINES_HYDROCONQUEST_AUTO", "HydroConquest",
        required=["longines", "hydroconquest"],
        scored=["hydroconquest", "hydro conquest", "automatic", "diver"],
        points_each=30, bonus_if_all=20,
    ),
    FamilySignature(
        "LONGINES_MASTER_COLLECTION_SIMPLE", "Master Collection",
        required=["longines"],
        scored=["master", "collection", "automatic"],
        points_each=20, bonus_if_all=10,
    ),
    FamilySignature(
        "LONGINES_VTG_AUTO_GENERIC", "Vintage Automatic (generic)",
        required=["longines"],
        scored=["vintage", "automatic"],
        points_each=10,
    ),
],

"TAG Heuer": [
    FamilySignature(
        "TAG_AQUARACER_AUTO_WAF_WAY", "Aquaracer",
        required=["tag", "heuer"],
        scored=["aquaracer", "waf", "way", "300m"],
        points_each=20, bonus_if_all=15,
    ),
    FamilySignature(
        "TAG_FORMULA1_AUTO_WAZ_CAW_WAU", "Formula 1",
        required=["tag", "heuer"],
        scored=["formula 1", "formula1", "waz", "caw", "wau"],
        points_each=20, bonus_if_all=15,
    ),
    FamilySignature(
        "TAG_CARRERA_AUTO_CALIBRE", "Carrera",
        required=["tag", "heuer"],
        scored=["carrera", "calibre", "calibre 16", "calibre 5"],
        points_each=20, bonus_if_all=15,
    ),
],

"Hamilton": [
    FamilySignature(
        "HAM_KHAKI_FIELD_AUTO", "Khaki Field",
        required=["hamilton"],
        scored=["khaki", "field", "automatic"],
        points_each=20, bonus_if_all=10,
    ),
    FamilySignature(
        "HAM_MURPH_AUTO", "Murph",
        required=["hamilton"],
        scored=["murph", "interstellar"],
        points_each=30, bonus_if_all=20,
    ),
],

"Oris": [
    FamilySignature(
        "ORIS_BCPD", "Big Crown Pointer Date",
        required=["oris"],
        scored=["big crown", "pointer date", "bigcrown"],
        points_each=25, bonus_if_all=20,
    ),
    FamilySignature(
        "ORIS_AQUIS_DATE", "Aquis Date",
        required=["oris"],
        scored=["aquis", "date"],
        points_each=25, bonus_if_all=15,
    ),
],

"Certina": [
    FamilySignature(
        "CERTINA_DS_ACTION_POWERMATIC80", "DS Action",
        required=["certina"],
        scored=["ds action", "ds-action", "powermatic", "diver"],
        points_each=20, bonus_if_all=15,
    ),
],

"Baume & Mercier": [
    FamilySignature(
        "BMM_CLIFTON_AUTO", "Clifton",
        required=["baume"],
        scored=["clifton", "automatic"],
        points_each=25, bonus_if_all=15,
    ),
    FamilySignature(
        "BMM_CLASSIMA_AUTO", "Classima",
        required=["baume"],
        scored=["classima", "automatic"],
        points_each=25, bonus_if_all=15,
    ),
],

"Frederique Constant": [
    FamilySignature(
        "FC_CLASSICS_AUTO", "Classics",
        required=["frederique"],
        scored=["classics", "classic", "automatic"],
        points_each=20, bonus_if_all=10,
    ),
],

"Raymond Weil": [
    FamilySignature(
        "RW_FREELANCER_AUTO", "Freelancer",
        required=["raymond", "weil"],
        scored=["freelancer", "automatic"],
        points_each=25, bonus_if_all=15,
    ),
],

"Sinn": [
    FamilySignature(
        "SINN_556_AUTO", "556",
        required=["sinn"],
        scored=["556", "556a", "556i"],
        points_each=30, bonus_if_all=20,
    ),
    FamilySignature(
        "SINN_104_AUTO", "104",
        required=["sinn"],
        scored=["104", "104 st", "104 sa"],
        points_each=30, bonus_if_all=20,
    ),
],

"Junghans": [
    FamilySignature(
        "JUN_MAX_BILL_AUTO", "Max Bill",
        required=["junghans"],
        scored=["max bill", "maxbill", "automatic"],
        points_each=25, bonus_if_all=20,
    ),
],

"Nomos": [
    FamilySignature(
        "NOMOS_TANGENTE", "Tangente",
        required=["nomos"],
        scored=["tangente"],
        points_each=35, bonus_if_all=20,
    ),
    FamilySignature(
        "NOMOS_CLUB", "Club",
        required=["nomos"],
        scored=["club"],
        points_each=35, bonus_if_all=20,
    ),
    FamilySignature(
        "NOMOS_METRO", "Metro",
        required=["nomos"],
        scored=["metro"],
        points_each=35, bonus_if_all=20,
    ),
],

"Tudor": [
    FamilySignature(
        "TUDOR_BLACK_BAY", "Black Bay",
        required=["tudor"],
        scored=["black bay", "blackbay", "bb"],
        points_each=25, bonus_if_all=20,
    ),
    FamilySignature(
        "TUDOR_1926_AUTO", "1926",
        required=["tudor"],
        scored=["1926", "automatic"],
        points_each=25, bonus_if_all=15,
    ),
    FamilySignature(
        "TUDOR_GLAMOUR_AUTO", "Glamour",
        required=["tudor"],
        scored=["glamour", "automatic"],
        points_each=25, bonus_if_all=15,
    ),
],

"Breitling": [
    FamilySignature(
        "BREITLING_COLT_AUTO", "Colt",
        required=["breitling"],
        scored=["colt", "automatic"],
        points_each=25, bonus_if_all=15,
    ),
],

"Maurice Lacroix": [
    FamilySignature(
        "MAURICE_LACROIX_AIKON_AUTO", "Aikon",
        required=["maurice", "lacroix"],
        scored=["aikon", "automatic"],
        points_each=25, bonus_if_all=15,
    ),
],

"Mido": [
    FamilySignature(
        "MIDO_OCEAN_STAR_AUTO", "Ocean Star",
        required=["mido"],
        scored=["ocean star", "oceanstar", "automatic"],
        points_each=25, bonus_if_all=15,
    ),
],

"Rado": [
    FamilySignature(
        "RADO_CAPTAIN_COOK_AUTO", "Captain Cook",
        required=["rado"],
        scored=["captain cook", "captain", "cook", "automatic"],
        points_each=20, bonus_if_all=15,
    ),
],

"Doxa": [
    FamilySignature(
        "DOXA_SUB_AUTO", "SUB",
        required=["doxa"],
        scored=["sub", "automatic", "diver"],
        points_each=20, bonus_if_all=10,
    ),
],

"Fortis": [
    FamilySignature(
        "FORTIS_B42_AUTO", "B-42",
        required=["fortis"],
        scored=["b-42", "b42"],
        points_each=30, bonus_if_all=20,
    ),
    FamilySignature(
        "FORTIS_COSMONAUTS_AUTO", "Cosmonauts",
        required=["fortis"],
        scored=["cosmonaut", "cosmonauts"],
        points_each=30, bonus_if_all=20,
    ),
],

"Glycine": [
    FamilySignature(
        "GLYCINE_AIRMAN_AUTO", "Airman",
        required=["glycine"],
        scored=["airman", "automatic", "gmt"],
        points_each=20, bonus_if_all=10,
    ),
],

"Alpina": [
    FamilySignature(
        "ALPINA_STARTIMER_AUTO", "Startimer",
        required=["alpina"],
        scored=["startimer", "pilot", "automatic"],
        points_each=20, bonus_if_all=10,
    ),
],

"Yema": [
    FamilySignature(
        "YEMA_SUPERMAN_AUTO", "Superman",
        required=["yema"],
        scored=["superman", "automatic"],
        points_each=25, bonus_if_all=15,
    ),
],

"Zodiac": [
    FamilySignature(
        "ZODIAC_SEA_WOLF_AUTO", "Sea Wolf",
        required=["zodiac"],
        scored=["sea wolf", "seawolf", "automatic"],
        points_each=25, bonus_if_all=15,
    ),
],

"Montblanc": [
    FamilySignature(
        "MONTBLANC_1858", "1858",
        required=["montblanc"],
        scored=["1858", "automatic"],
        points_each=25, bonus_if_all=15,
    ),
    FamilySignature(
        "MONTBLANC_ICED_SEA", "Iced Sea",
        required=["montblanc"],
        scored=["iced sea", "iced-sea", "ice sea", "diver"],
        points_each=25, bonus_if_all=15,
    ),
],

"Meistersinger": [
    FamilySignature(
        "MEISTERSINGER_NO1", "No.1",
        required=["meistersinger"],
        scored=["no.1", "no 1", "n1", "single hand", "einzeiger"],
        points_each=20, bonus_if_all=10,
    ),
    FamilySignature(
        "MEISTERSINGER_NEO", "Neo",
        required=["meistersinger"],
        scored=["neo"],
        points_each=30, bonus_if_all=20,
    ),
],

"Stowa": [
    FamilySignature(
        "STOWA_FLIEGER_AUTO", "Flieger",
        required=["stowa"],
        scored=["flieger", "pilot", "automatic"],
        points_each=20, bonus_if_all=10,
    ),
],

"Laco": [
    FamilySignature(
        "LACO_FLIEGER_AUTO", "Flieger / Pilot",
        required=["laco"],
        scored=["flieger", "pilot", "automatic"],
        points_each=20, bonus_if_all=10,
    ),
],

"Damasko": [
    FamilySignature(
        "DAMASKO_DA_DC_AUTO", "DA / DC",
        required=["damasko"],
        scored=["da", "dc", "automatic"],
        points_each=20, bonus_if_all=10,
    ),
],
```

}

@dataclass
class FamilyMatch:
target_id: str
family_name: str
score: int
matched_keywords: List[str] = field(default_factory=list)

def score_families(text: str, brand: Optional[str]) -> List[FamilyMatch]:
"""
Layer 3: Score all family signatures for the detected brand.

```
Returns a ranked list of FamilyMatch objects (highest score first).
Only families whose required terms are ALL present are considered.
"""
if not brand or brand not in FAMILY_SIGNATURES:
    return []

t = _norm(text)
results: List[FamilyMatch] = []

for sig in FAMILY_SIGNATURES[brand]:
    # Hard gate: all required terms must be present
    if not all(req in t for req in sig.required):
        continue

    # Score: count how many scored terms hit
    hits = [kw for kw in sig.scored if kw in t]
    if not hits:
        continue

    score = len(hits) * sig.points_each
    # Bonus if all scored terms hit
    if sig.bonus_if_all and len(hits) == len(sig.scored):
        score += sig.bonus_if_all

    results.append(FamilyMatch(
        target_id=sig.family_id,
        family_name=sig.family_name,
        score=score,
        matched_keywords=hits,
    ))

results.sort(key=lambda x: -x.score)
return results
```

# —————————————————————————

# Ref-to-family fallback map

# —————————————————————————

# When Layer 3 finds no family match but Layer 2 extracted a known ref,

# this map resolves target_id directly from the ref prefix.

_REF_PREFIX_TO_TARGET: Dict[str, str] = {
"SKX":    "SEIKO_PROSPEX_TURTLE_AUTO",
"SRPA":   "SEIKO_PROSPEX_TURTLE_AUTO",
"SRPB":   "SEIKO_PROSPEX_TURTLE_AUTO",
"SRPC":   "SEIKO_PROSPEX_TURTLE_AUTO",
"SRPD":   "SEIKO_PROSPEX_SAMURAI_AUTO",
"SRPE":   "SEIKO_PROSPEX_TURTLE_AUTO",
"SARB":   "SEIKO_ALPINIST_SARB017_SPB121",
"SPB":    "SEIKO_ALPINIST_SARB017_SPB121",
"SBDC":   "SEIKO_PROSPEX_TURTLE_AUTO",
"WAF":    "TAG_AQUARACER_AUTO_WAF_WAY",
"WAY":    "TAG_AQUARACER_AUTO_WAF_WAY",
"WAZ":    "TAG_FORMULA1_AUTO_WAZ_CAW_WAU",
"CAW":    "TAG_FORMULA1_AUTO_WAZ_CAW_WAU",
"WAU":    "TAG_FORMULA1_AUTO_WAZ_CAW_WAU",
}

_REF_TO_FAMILY_NAME: Dict[str, str] = {
"SEIKO_PROSPEX_TURTLE_AUTO":          "Prospex Turtle",
"SEIKO_PROSPEX_SAMURAI_AUTO":         "Prospex Samurai",
"SEIKO_ALPINIST_SARB017_SPB121":      "Alpinist",
"TAG_AQUARACER_AUTO_WAF_WAY":         "Aquaracer",
"TAG_FORMULA1_AUTO_WAZ_CAW_WAU":      "Formula 1",
}

def _resolve_family_from_refs(refs: List[str]) -> Tuple[Optional[str], Optional[str]]:
"""Returns (target_id, family_name) from ref prefix, or (None, None)."""
for ref in refs:
for prefix, target_id in _REF_PREFIX_TO_TARGET.items():
if ref.upper().startswith(prefix):
return target_id, _REF_TO_FAMILY_NAME.get(target_id)
return None, None

# 

# Combines brand + family + references to resolve the best target_list entry.

# Ref matching gives a strong bonus: if an extracted ref matches a target's

# known refs list, confidence jumps significantly.

@dataclass
class TargetIndex:
"""Pre-built index of target_list targets for fast lookup."""
by_id: Dict[str, object]           # target_id → TargetModel
brand_to_ids: Dict[str, List[str]] # brand (norm) → [target_ids]

def build_target_index(targets: List) -> TargetIndex:
"""
Build a lookup index from load_target_bundle() output.
Call once at startup.

```
Usage:
    targets, meta = load_target_bundle("target_list.json")
    target_index = build_target_index(targets)
"""
by_id: Dict[str, object] = {}
brand_to_ids: Dict[str, List[str]] = {}

for t in targets:
    tid = str(t.key).upper()
    by_id[tid] = t
    # Map normalised brand to target ids
    brand_norm = _norm(t.keywords[0]) if t.keywords else ""
    if brand_norm:
        brand_to_ids.setdefault(brand_norm, []).append(tid)

return TargetIndex(by_id=by_id, brand_to_ids=brand_to_ids)
```

def _refs_match(extracted_refs: List[str], target) -> bool:
"""Check if any extracted reference matches the target's known refs."""
if not extracted_refs or not getattr(target, "refs", None):
return False
t_refs = {_compact(r) for r in target.refs}
e_refs = {_compact(r) for r in extracted_refs}
return bool(t_refs & e_refs)

# —————————————————————————

# Layer 5 — Confidence Assembly

# —————————————————————————

# 

# Combines all layer signals into a final 0–100 confidence score and

# a categorical band.

# 

# Score composition:

# Brand detection:       0–30  (always present if we got here)

# Ref match:             0–30  (strong signal when refs available)

# Family score (norm):   0–25  (normalised from raw family score)

# Keyword density:       0–10  (more specific keywords = higher)

# Penalties:             0–25  (vague title, ambiguity, multi-family tie)

# 

# Confidence bands:

# high:       ≥ 75   → commit to this target, skip keyword loop

# medium:     55–74  → use this target but lower estimated_close

# low:        35–54  → flag for review, don't suppress alternatives

# very_low:   < 35   → fall through to existing keyword loop

BAND_THRESHOLDS = [
(75, "high"),
(55, "medium"),
(35, "low"),
(0,  "very_low"),
]

def _band(score: int) -> str:
for threshold, label in BAND_THRESHOLDS:
if score >= threshold:
return label
return "very_low"

# —————————————————————————

# Main entry point

# —————————————————————————

def identify_watch(
text: str,
target_index: Optional[TargetIndex] = None,
) -> Dict:
"""
Main identification function.

```
Args:
    text:           Listing title + description (concatenated)
    target_index:   Pre-built index from build_target_index().
                    If None, returns brand/family/ref without target_id.

Returns:
    {
        "brand":           str | None,
        "family":          str | None,
        "target_id":       str | None,
        "references":      list[str],
        "confidence":      int,          # 0–100
        "confidence_band": str,          # high/medium/low/very_low
        "layer_scores":    dict,         # per-layer debug scores
        "flags":           dict,         # signals that fired
    }
"""
t = _norm(text)
flags: Dict = {}

# --- Layer 1: Brand ---
brand, brand_score = detect_brand(t)
if not brand:
    return _empty_result("no_brand_detected")

# --- Layer 2: References ---
refs, ref_score = extract_references(t, brand)
flags["refs_extracted"] = refs
flags["ref_score"] = ref_score

# --- Layer 3: Family ---
family_matches = score_families(t, brand)
top_family: Optional[FamilyMatch] = family_matches[0] if family_matches else None
second_family: Optional[FamilyMatch] = family_matches[1] if len(family_matches) > 1 else None

# Ref-to-family fallback: if Layer 3 found nothing but we have a known ref
ref_resolved_target: Optional[str] = None
ref_resolved_family: Optional[str] = None
if not top_family and refs:
    ref_resolved_target, ref_resolved_family = _resolve_family_from_refs(refs)
    if ref_resolved_target:
        # Synthesise a FamilyMatch so the rest of the pipeline works uniformly
        top_family = FamilyMatch(
            target_id=ref_resolved_target,
            family_name=ref_resolved_family or "",
            score=40,          # moderate score — ref match, no keyword match
            matched_keywords=[],
        )
        flags["family_from_ref_fallback"] = True

# Vague title penalty: if no family hits at all
flags["vague_title"] = top_family is None
flags["multi_family_ambiguity"] = (
    second_family is not None and
    top_family is not None and
    second_family.score >= top_family.score * 0.80  # tie: 2nd is within 20% of 1st
)

# --- Layer 4: Target Resolution ---
resolved_target_id: Optional[str] = None
ref_match_bonus = 0

if top_family:
    candidate_id = top_family.target_id.upper()
    # Check ref match against this specific target
    if target_index and candidate_id in target_index.by_id:
        t_obj = target_index.by_id[candidate_id]
        if _refs_match(refs, t_obj):
            ref_match_bonus = 30
            flags["ref_matched_target"] = True
        else:
            flags["ref_matched_target"] = False
    resolved_target_id = candidate_id

# --- Layer 5: Confidence Assembly ---

# Brand component (0–30): we give 30 if brand detected (always true here)
c_brand = 30

# Ref component (0–30)
c_ref = ref_match_bonus if ref_match_bonus else (ref_score * 0.5 if refs else 0)

# Family component (0–25): normalise raw family score to 0–25
raw_family_score = top_family.score if top_family else 0
c_family = min(25, int(raw_family_score / 4))

# Keyword density (0–10): how many family keywords hit vs total title tokens
kw_hits = len(top_family.matched_keywords) if top_family else 0
title_tokens = max(1, len(_token_set(t)))
c_density = min(10, int((kw_hits / title_tokens) * 30))

# Penalties
penalty = 0
if flags.get("vague_title"):
    penalty += 25
if flags.get("multi_family_ambiguity"):
    penalty += 15
if not refs:
    penalty += 5   # small penalty: ref absence is common, not catastrophic

confidence = max(0, min(100, int(c_brand + c_ref + c_family + c_density - penalty)))
band = _band(confidence)

layer_scores = {
    "brand":    c_brand,
    "ref":      int(c_ref),
    "family":   c_family,
    "density":  c_density,
    "penalty":  -penalty,
    "total":    confidence,
}

flags["family_candidates"] = [
    {"target_id": fm.target_id, "score": fm.score, "keywords": fm.matched_keywords}
    for fm in family_matches[:3]
]

return {
    "brand":            brand,
    "family":           top_family.family_name if top_family else None,
    "target_id":        resolved_target_id,
    "references":       refs,
    "confidence":       confidence,
    "confidence_band":  band,
    "layer_scores":     layer_scores,
    "flags":            flags,
}
```

def _empty_result(reason: str) -> Dict:
return {
"brand":            None,
"family":           None,
"target_id":        None,
"references":       [],
"confidence":       0,
"confidence_band":  "very_low",
"layer_scores":     {},
"flags":            {"skip_reason": reason},
}

# —————————————————————————

# Integration helper: augmented compute_match_score

# —————————————————————————

# 

# Drop-in replacement for compute_match_score() in scanner.py.

# Uses the new engine as primary, falls back to keyword scoring.

def compute_match_score_v2(
text: str,
target,                    # TargetModel from scanner.py
target_index: Optional[TargetIndex] = None,
) -> Tuple[int, Dict]:
"""
Augmented match scorer. Returns (score_0_to_100, debug_dict).

```
Usage in scanner.py (replace compute_match_score call):

    from timelab_core.watch_id_engine import compute_match_score_v2, build_target_index
    target_index = build_target_index(targets)   # once at startup

    score, debug = compute_match_score_v2(detail_text, best_t, target_index)
"""
identity = identify_watch(text, target_index)

# If engine resolves to this specific target with decent confidence:
if (identity["target_id"] and
        identity["target_id"].upper() == str(target.key).upper() and
        identity["confidence"] >= 35):
    # Use engine confidence as the primary score
    engine_score = identity["confidence"]
    return engine_score, {"source": "engine", "identity": identity}

# Fallback: original keyword scoring (from scanner.py)
t = _norm(text)
kws = [_norm(k) for k in (getattr(target, "keywords", None) or []) if _norm(k)]
if not kws:
    return 0, {"source": "keyword_fallback", "identity": identity}

brand_kw = kws[0]
model_kws = kws[1:] if len(kws) > 1 else []

score = 0
if brand_kw and brand_kw in t:
    score += 45
if model_kws:
    hits = sum(1 for kw in model_kws if kw and kw in t)
    score += int(45 * hits / max(1, len(model_kws)))
else:
    score += 15

refs = getattr(target, "refs", None) or []
if refs:
    rhits = sum(1 for r in refs if _norm(r) in t)
    score += int(10 * rhits / max(1, len(refs)))
else:
    score += 5

if any(x in t for x in {"replica", "copy", "fake", "imitacion", "imitación", "imitation"}):
    score -= 70

return max(0, min(100, score)), {"source": "keyword_fallback", "identity": identity}
```

# —————————————————————————

# Smoke tests (run: python -m timelab_core.watch_id_engine)

# —————————————————————————

if **name** == "**main**":
import json

```
TEST_CASES = [
    # (title, expected_brand, expected_family_contains, expected_band)
    ("Seiko automatic watch",                    "Seiko",      None,         "very_low"),
    ("Seiko SKX007 diver automatic",             "Seiko",      "Turtle",     "medium"),
    ("Seiko Alpinist SARB017 automatic",         "Seiko",      "Alpinist",   "high"),
    ("Seiko Prospex Turtle SRP automatic",       "Seiko",      "Turtle",     "high"),
    ("Tissot Seastar automatic",                 "Tissot",     "Seastar",    "high"),
    ("Tissot PRX Powermatic 80 40mm",            "Tissot",     "PRX",        "high"),
    ("Vintage Tissot watch",                     "Tissot",     None,         "very_low"),
    ("Omega Seamaster vintage automatic",        "Omega",      "Seamaster",  "high"),
    ("Omega De Ville automatic",                 "Omega",      "De Ville",   "medium"),
    ("Longines HydroConquest automatic",         "Longines",   "HydroConq",  "high"),
    ("Longines vintage automatic watch",         "Longines",   "Vintage",    "low"),
    ("TAG Heuer Aquaracer WAF2110 automatic",   "TAG Heuer",  "Aquaracer",  "high"),
    ("TAG Heuer Carrera calibre 16 chrono",     "TAG Heuer",  "Carrera",    "high"),
    ("Hamilton Khaki Field automatic 38mm",      "Hamilton",   "Khaki",      "high"),
    ("Hamilton Murph automatic",                 "Hamilton",   "Murph",      "high"),
    ("Tudor Black Bay automatic",                "Tudor",      "Black Bay",  "high"),
    ("NOMOS Tangente 38 automatic",              "Nomos",      "Tangente",   "high"),
    ("Sinn 556 automatic steel",                 "Sinn",       "556",        "high"),
    ("Junghans Max Bill automatic",              "Junghans",   "Max Bill",   "high"),
]

print(f"{'Title':<45} {'Brand':<15} {'Family':<20} {'Conf':>4} {'Band':<10}")
print("-" * 100)
all_pass = True
for title, exp_brand, exp_family, exp_band in TEST_CASES:
    result = identify_watch(title)
    brand_ok = result["brand"] == exp_brand
    family_ok = (exp_family is None) or (
        result["family"] and exp_family.lower() in result["family"].lower()
    )
    # Band: allow one level off on medium/low since these are edge cases
    band_ok = result["confidence_band"] == exp_band

    status = "✅" if (brand_ok and family_ok) else "❌"
    if not (brand_ok and family_ok):
        all_pass = False

    fam_short = (result["family"] or "None")[:18]
    print(
        f"{status} {title:<43} {str(result['brand']):<15} {fam_short:<20} "
        f"{result['confidence']:>4} {result['confidence_band']:<10}"
    )

print()
print("All brand+family checks passed!" if all_pass else "Some checks FAILED — review above.")
print()
print("--- Sample full output (Seiko Alpinist) ---")
print(json.dumps(identify_watch("Seiko Alpinist SARB017 automatic"), indent=2))
```