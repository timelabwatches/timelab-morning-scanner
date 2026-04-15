# auction/auction_price_engine.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import re
from pathlib import Path


REFERENCE_SEED_PATH = Path("data/reference_prices_seed_v1.csv")


MODEL_ALIASES = {
    "seiko": {
        "prospex": ["prospex", "samurai", "turtle", "sumo", "alpinist", "marine master", "marinemaster", "shogun"],
        "sportura": ["sportura"],
        "arctura": ["arctura"],
        "presage": ["presage", "cocktail time", "cocktail"],
        "5 sports": ["5 sports", "seiko 5", "seiko5", "5sport"],
        "astron": ["astron"],
        "kinetic diver": ["kinetic diver", "kinetic scuba", "scuba", "diver 200", "200m diver"],
        "diver": ["diver", "skx", "turtle", "samurai", "sumo"],
        "chronograph": ["chronograph", "chrono"],
    },
    "tissot": {
        "prx": ["prx"],
        "seastar": ["seastar", "sea star"],
        "le locle": ["le locle", "lelocle"],
        "visodate": ["visodate"],
        "t-race": ["t-race", "t race", "trace"],
        "carson": ["carson"],
        "couturier": ["couturier"],
        "tradition": ["tradition"],
        "pr 100": ["pr 100", "pr100", "pr-100"],
        "prs516": ["prs516", "prs 516"],
        "chronograph": ["chronograph", "chrono"],
    },
    "omega": {
        "seamaster": ["seamaster"],
        "speedmaster": ["speedmaster"],
        "constellation": ["constellation"],
        "de ville": ["de ville", "deville"],
        "geneve": ["geneve", "genève"],
        "dynamic": ["dynamic"],
        "chronograph": ["chronograph", "chrono"],
        "vintage automatic": ["automatic", "vintage", "omega automatic"],
    },
    "longines": {
        "hydroconquest": ["hydroconquest", "hydro conquest"],
        "conquest": ["conquest"],
        "flagship": ["flagship"],
        "dolcevita": ["dolcevita", "dolce vita"],
        "evidenza": ["evidenza"],
        "master collection": ["master collection", "mastercollection"],
        "presence": ["presence"],
        "admiral": ["admiral"],
        "chronograph": ["chronograph", "chrono"],
    },
    "certina": {
        "ds action": ["ds action", "ds-action"],
        "ds podium": ["ds podium", "ds-podium"],
        "ds-1": ["ds-1", "ds 1"],
        "ds-2": ["ds-2", "ds 2"],
        "ds first": ["ds first", "ds-first", "dsfirst"],
        "ds": ["ds", "certina ds"],
        "chronograph": ["chronograph", "chrono"],
    },
    "hamilton": {
        "khaki field": ["khaki field"],
        "khaki navy": ["khaki navy", "khaki scuba", "navy scuba"],
        "jazzmaster": ["jazzmaster", "jazz master"],
        "ventura": ["ventura"],
        "american classic": ["american classic"],
        "chronograph": ["chronograph", "chrono"],
    },
    "tagheuer": {
        "formula 1": ["formula 1", "formula1"],
        "aquaracer": ["aquaracer", "aqua racer"],
        "kirium": ["kirium"],
        "link": ["link"],
        "monaco": ["monaco"],
        "carrera": ["carrera"],
        "chronograph": ["chronograph", "chrono"],
    },
    "baume": {
        "classima": ["classima"],
        "riviera": ["riviera"],
        "clifton": ["clifton"],
        "hampton": ["hampton"],
        "chronograph": ["chronograph", "chrono"],
    },
    "junghans": {
        "max bill": ["max bill"],
        "meister": ["meister"],
        "chronograph": ["chronograph", "chrono", "chronoscope"],
    },
    "citizen": {
        "promaster": ["promaster"],
        "tsuyosa": ["tsuyosa"],
        "eco-drive": ["eco-drive", "eco drive", "ecodrive"],
        "chronograph": ["chronograph", "chrono"],
        "diver": ["diver", "promaster diver"],
    },
    "bulova": {
        "marine star": ["marine star", "marinestar"],
        "lunar pilot": ["lunar pilot", "lunarpilot"],
        "precisionist": ["precisionist"],
        "classic": ["classic"],
        "chronograph": ["chronograph", "chrono"],
    },
    "orient": {
        "bambino": ["bambino"],
        "kamasu": ["kamasu"],
        "mako": ["mako"],
        "ray": ["ray"],
        "star": ["orient star", "star"],
        "diver": ["diver", "kamasu", "mako", "ray"],
    },
    "casio": {
        "g-shock": ["g-shock", "g shock", "gshock"],
        "edifice": ["edifice"],
    },
    "festina": {
        "chronograph": ["chronograph", "chrono"],
        "sport": ["sport"],
        "automatic": ["automatic"],
    },
    "raymondweil": {
        "tango": ["tango"],
        "freelancer": ["freelancer"],
        "maestro": ["maestro"],
        "parsifal": ["parsifal"],
    },
}

# ── BRAND_FAMILY_FALLBACKS ──────────────────────────────────────────────────
# Calibrated with TIMELAB Q1 2026 real Catawiki sales + market research Apr 2026.
# Format: (p25, p50, p75) — Catawiki hammer prices EUR
# Last updated: 2026-04-15
BRAND_FAMILY_FALLBACKS = {
    "omega": {
        # Q1: De Ville quartz €370 (1 sale). Vintage auto worth more.
        "geneve":           (320.0, 480.0, 680.0),
        "de ville":         (320.0, 480.0, 680.0),
        # Q1: Constellation vintage €270-500 range (market data)
        "constellation":    (280.0, 420.0, 580.0),
        # Market: VTG Seamaster €550-850 Catawiki
        "seamaster":        (520.0, 700.0, 900.0),
        # Modern Aqua Terra: €1400-2000
        "aqua terra":       (1300.0, 1700.0, 2100.0),
        # Modern Speedmaster: strong collector market
        "speedmaster":      (1800.0, 2600.0, 3500.0),
        "chronograph":      (900.0, 1400.0, 2000.0),
        "brand_fallback":   (350.0, 600.0, 900.0),
    },
    "longines": {
        # Q1: Flagship €890, 22AS €505 (real sales). Generic vtg €180-400.
        "flagship":         (450.0, 680.0, 900.0),
        "hydroconquest":    (700.0, 950.0, 1250.0),
        # Q1: Conquest vtg €320-480 Catawiki market
        "conquest":         (320.0, 450.0, 620.0),
        "master collection":(700.0, 850.0, 1050.0),
        "presence":         (200.0, 320.0, 450.0),
        "admiral":          (350.0, 500.0, 680.0),
        "chronograph":      (450.0, 650.0, 900.0),
        "brand_fallback":   (220.0, 380.0, 580.0),
    },
    "tissot": {
        # Q1: 24 real sales avg €209. PRX PM80 €580, Seastar PM80 €700, T-Lord €494
        "prx":              (430.0, 580.0, 720.0),
        "seastar":          (560.0, 700.0, 850.0),
        "le locle":         (220.0, 320.0, 430.0),
        "visodate":         (200.0, 290.0, 380.0),
        "t-race":           (150.0, 230.0, 320.0),
        "carson":           (150.0, 220.0, 300.0),
        "tradition":        (160.0, 240.0, 330.0),
        "pr 100":           (80.0,  160.0, 240.0),
        # Q1 chrono avg €200 (24 sales, incl. XL, PRS, Quickster)
        "chronograph":      (140.0, 210.0, 290.0),
        "brand_fallback":   (100.0, 200.0, 320.0),
    },
    "seiko": {
        # Q1: Prospex Solar €420, Presage €320-340, Sports chrono €340
        "prospex":          (250.0, 400.0, 560.0),
        "presage":          (240.0, 340.0, 480.0),
        "5 sports":         (90.0,  160.0, 240.0),
        "sportura":         (80.0,  130.0, 200.0),
        "arctura":          (80.0,  120.0, 180.0),
        "kinetic diver":    (120.0, 190.0, 270.0),
        "diver":            (160.0, 300.0, 450.0),
        # Q1: generic Seiko chrono avg €190 (7T42, 7T92 etc.)
        "chronograph":      (120.0, 195.0, 280.0),
        "brand_fallback":   (80.0,  160.0, 260.0),
    },
    "tagheuer": {
        # Market: Formula 1 auto €700-950, Aquaracer €750-1050
        "formula 1":        (580.0, 780.0, 980.0),
        "aquaracer":        (680.0, 900.0, 1100.0),
        "link":             (600.0, 850.0, 1100.0),
        "carrera":          (1400.0, 1900.0, 2600.0),
        "monaco":           (3000.0, 4200.0, 5500.0),
        "chronograph":      (800.0, 1200.0, 1800.0),
        "brand_fallback":   (550.0, 900.0, 1400.0),
    },
    "hamilton": {
        # Q1: Khaki Field €460 Catawiki, Regatta vtg €760 (exceptional), Jazzline €263
        "khaki field":      (340.0, 460.0, 600.0),
        "khaki navy":       (420.0, 580.0, 780.0),
        "khaki regatta":    (450.0, 620.0, 820.0),
        "jazzmaster":       (220.0, 340.0, 480.0),
        "ventura":          (580.0, 760.0, 950.0),
        "murph":            (680.0, 850.0, 1020.0),
        "chronograph":      (420.0, 600.0, 800.0),
        "brand_fallback":   (220.0, 380.0, 580.0),
    },
    "citizen": {
        "promaster":        (200.0, 300.0, 400.0),
        "eco-drive":        (120.0, 190.0, 270.0),
        "tsuyosa":          (240.0, 320.0, 410.0),
        "diver":            (200.0, 290.0, 380.0),
        "chronograph":      (140.0, 210.0, 290.0),
        "brand_fallback":   (100.0, 190.0, 300.0),
    },
    "baume": {
        # Q1: Classima €400/€625, avg €512 (2 real sales)
        "classima":         (400.0, 540.0, 700.0),
        "riviera":          (1200.0, 1600.0, 2000.0),
        "clifton":          (1400.0, 1850.0, 2300.0),
        "chronograph":      (1000.0, 1400.0, 1900.0),
        "brand_fallback":   (400.0, 700.0, 1100.0),
    },
    "certina": {
        # Q1: DS First €266, Vintage €113 (real sales)
        "ds action":        (380.0, 520.0, 680.0),
        "ds podium":        (200.0, 290.0, 380.0),
        "ds-1":             (220.0, 320.0, 420.0),
        "ds-2":             (180.0, 270.0, 360.0),
        "ds":               (160.0, 260.0, 370.0),
        "chronograph":      (200.0, 300.0, 400.0),
        "brand_fallback":   (150.0, 260.0, 380.0),
    },
    "bulova": {
        "marine star":      (160.0, 230.0, 310.0),
        "lunar pilot":      (320.0, 410.0, 500.0),
        "precisionist":     (240.0, 320.0, 410.0),
        "classic":          (120.0, 190.0, 260.0),
        "chronograph":      (160.0, 250.0, 340.0),
        "brand_fallback":   (120.0, 210.0, 320.0),
    },
    "zenith": {
        # Q1: 4 real sales €175-360, avg €270. Was wildly overpriced at p50=720.
        "el primero":       (900.0, 1400.0, 2000.0),
        "pilot":            (700.0, 1000.0, 1400.0),
        "chronograph":      (500.0, 800.0, 1200.0),
        "brand_fallback":   (200.0, 300.0, 420.0),
    },
    "junghans": {
        # Q1: Max Bill €510 (1 real sale)
        "max bill":         (380.0, 540.0, 720.0),
        "meister":          (300.0, 420.0, 580.0),
        "chronograph":      (280.0, 420.0, 580.0),
        "brand_fallback":   (250.0, 380.0, 540.0),
    },
    "oris": {
        # Q1: Vintage Oris €261 (1 real sale). Modern BCPD/Aquis higher.
        "big crown":        (320.0, 460.0, 620.0),
        "bcpd":             (450.0, 600.0, 780.0),
        "aquis":            (550.0, 720.0, 920.0),
        "divers":           (450.0, 620.0, 820.0),
        "chronograph":      (400.0, 580.0, 780.0),
        "brand_fallback":   (230.0, 350.0, 500.0),
    },
    "breitling": {
        # Market: Colt Auto €850-1050 Catawiki, Navitimer vtg €800-2000
        "navitimer":        (800.0, 1200.0, 1800.0),
        "colt":             (750.0, 980.0, 1250.0),
        "superocean":       (900.0, 1300.0, 1800.0),
        "avenger":          (900.0, 1350.0, 1900.0),
        "chronograph":      (900.0, 1500.0, 2200.0),
        "brand_fallback":   (700.0, 1100.0, 1700.0),
    },
    "tudor": {
        # Market: BB41 €2000, BB58 €2100 Catawiki
        "black bay":        (1600.0, 2000.0, 2600.0),
        "pelagos":          (1800.0, 2400.0, 3200.0),
        "ranger":           (900.0, 1200.0, 1600.0),
        "brand_fallback":   (900.0, 1400.0, 2000.0),
    },
    "iwc": {
        # Market: Pilot/Flieger €700-1300 Catawiki
        "pilot":            (620.0, 900.0, 1200.0),
        "flieger":          (620.0, 900.0, 1200.0),
        "aquatimer":        (700.0, 1000.0, 1400.0),
        "chronograph":      (900.0, 1400.0, 2000.0),
        "brand_fallback":   (600.0, 950.0, 1400.0),
    },
}

LADIES_KEYWORDS = [
    "ladies",
    "lady",
    "women",
    "woman",
    "female",
    "mujer",
    "señora",
    "senora",
    "dama",
    "chica",
    "girls",
    "femme",
    "donna",
]

MEN_KEYWORDS = [
    "men",
    "man's",
    "mens",
    "male",
    "hombre",
    "caballero",
    "homme",
    "uomo",
]

SMALL_CASE_HINT_PATTERNS = [
    r"\b(\d{1,2})\s?mm\b",
]

LADIES_MM_MAX = 28


def empty_estimate(reason: str = "insufficient_real_closed_data") -> dict:
    return {
        "price_estimate_available": False,
        "raw_expected_hammer": None,
        "expected_hammer": None,
        "conservative_hammer": None,
        "optimistic_hammer": None,
        "price_confidence": "low",
        "pricing_reason": reason,
        "applied_haircut": None,
    }


def safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "si", "sí"}


def median(values: list[float]) -> float | None:
    if not values:
        return None

    values = sorted(values)
    n = len(values)
    mid = n // 2

    if n % 2 == 1:
        return values[mid]

    return (values[mid - 1] + values[mid]) / 2


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(str(value).lower().split())


def normalize_reference(value: str | None) -> str:
    if not value:
        return ""
    value = str(value).lower().strip()
    value = value.replace(" ", "")
    value = value.replace(".", "")
    value = value.replace("-", "")
    value = value.replace("/", "")
    value = value.replace("_", "")
    return value


def build_text_blob(record: dict) -> str:
    parts = [
        record.get("brand"),
        record.get("model"),
        record.get("reference"),
        record.get("title"),
        record.get("raw_text"),
        record.get("raw_text_clean"),
        record.get("analysis_text"),
        record.get("discovery_context"),
    ]
    return normalize_text(" | ".join(str(x or "") for x in parts))


def build_high_signal_ladies_text(record: dict) -> str:
    parts = [
        record.get("brand"),
        record.get("model"),
        record.get("reference"),
        record.get("title"),
        record.get("discovery_context"),
    ]

    cleaned = []
    for value in parts:
        text = normalize_text(value)
        if text:
            cleaned.append(text)

    return " | ".join(cleaned)


def contains_keyword_as_word(text: str, keyword: str) -> bool:
    escaped = re.escape(keyword.lower())
    pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
    return re.search(pattern, text) is not None


def infer_watch_type_from_text(text: str) -> str | None:
    if not text:
        return None

    if any(x in text for x in ["chronograph", "chrono", "cronografo", "cronógrafo"]):
        return "chronograph"
    if any(x in text for x in ["diver", "diving", "skx", "scuba", "300m", "200m"]):
        return "diver"
    if any(x in text for x in ["automatic", "automático", "automatico"]):
        return "automatic"
    if "quartz" in text:
        return "quartz"

    return None


def infer_case_size_mm(text: str) -> int | None:
    if not text:
        return None

    for pattern in SMALL_CASE_HINT_PATTERNS:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                mm = int(match)
                if 10 <= mm <= 60:
                    return mm
            except ValueError:
                continue

    return None


def infer_ladies_flag(record: dict) -> tuple[bool, str]:
    text = build_high_signal_ladies_text(record)
    lower_text = normalize_text(text)

    if not lower_text:
        return False, ""

    for keyword in LADIES_KEYWORDS:
        if contains_keyword_as_word(lower_text, keyword):
            return True, "ladies_keyword"

    for keyword in MEN_KEYWORDS:
        if contains_keyword_as_word(lower_text, keyword):
            return False, "men_keyword"

    kb_data = record.get("reference_kb_data") or {}
    if parse_bool(kb_data.get("is_ladies")):
        return True, "reference_kb_is_ladies"

    case_size = infer_case_size_mm(lower_text)
    if case_size is not None and case_size <= LADIES_MM_MAX:
        return True, "small_case_size_mm"

    return False, ""


def score_family_matches(text: str, brand: str | None) -> dict[str, int]:
    scores = {}
    if not brand:
        return scores

    brand_map = MODEL_ALIASES.get(brand, {})

    for family, aliases in brand_map.items():
        score = 0
        for alias in aliases:
            alias_norm = normalize_text(alias)
            if alias_norm and alias_norm in text:
                score += 1
        if score > 0:
            scores[family] = score

    return scores


def infer_family_from_text(record: dict, brand: str | None) -> str | None:
    if not brand:
        return None

    text = build_text_blob(record)
    scores = score_family_matches(text, brand)

    if not scores:
        return None

    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[0][0]


def extract_reference_candidates_from_record(record: dict) -> list[str]:
    candidates = []
    fields = [
        record.get("reference"),
        record.get("title"),
        record.get("analysis_text"),
        record.get("discovery_context"),
        record.get("raw_text_clean"),
    ]

    patterns = [
        r"\b[a-z]\.\d{1,4}\.\d{1,4}\.\d{1,4}\b",
        r"\b[a-z]\d{5,8}[a-z]?\b",
        r"\b\d{4}-\d{4}[a-z]?\b",
        r"\b[a-z]{1,4}\d{2,6}[a-z]?\b",
        r"\b\d{5,8}[a-z]?\b",
    ]

    for field in fields:
        text = normalize_text(field)
        if not text:
            continue

        direct_ref = normalize_reference(field)
        if direct_ref and len(direct_ref) >= 5:
            candidates.append(direct_ref)

        for pattern in patterns:
            for match in re.findall(pattern, text):
                normalized = normalize_reference(match)
                if normalized and len(normalized) >= 5:
                    candidates.append(normalized)

    seen = set()
    result = []

    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)

    return result


def load_reference_seed() -> dict:
    db = {
        "by_reference": {},
        "by_family": {},
    }

    if not REFERENCE_SEED_PATH.exists():
        return db

    family_buckets = {}

    with REFERENCE_SEED_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            brand = normalize_text(row.get("brand"))
            family = normalize_text(row.get("family"))
            reference = normalize_reference(row.get("reference"))

            if not brand:
                continue

            p25 = safe_float(row.get("p25"))
            p50 = safe_float(row.get("p50"))
            p75 = safe_float(row.get("p75"))

            if p25 is None or p50 is None or p75 is None:
                continue

            entry = {
                "brand": brand,
                "family": family or None,
                "reference": reference or None,
                "p25": p25,
                "p50": p50,
                "p75": p75,
                "liquidity": normalize_text(row.get("liquidity")) or "low",
                "source_title": (row.get("source_title") or "").strip(),
            }

            if reference:
                db["by_reference"][(brand, reference)] = entry

            if family:
                key = (brand, family)

                if key not in family_buckets:
                    family_buckets[key] = {
                        "p25": [],
                        "p50": [],
                        "p75": [],
                        "liquidity": [],
                        "titles": [],
                    }

                family_buckets[key]["p25"].append(p25)
                family_buckets[key]["p50"].append(p50)
                family_buckets[key]["p75"].append(p75)
                family_buckets[key]["liquidity"].append(entry["liquidity"])
                family_buckets[key]["titles"].append(entry["source_title"])

    for (brand, family), bucket in family_buckets.items():
        db["by_family"][(brand, family)] = {
            "brand": brand,
            "family": family,
            "reference": None,
            "p25": median(bucket["p25"]),
            "p50": median(bucket["p50"]),
            "p75": median(bucket["p75"]),
            "sample_size": len(bucket["p50"]),
            "liquidity": max(bucket["liquidity"], key=bucket["liquidity"].count) if bucket["liquidity"] else "low",
            "sample_titles": bucket["titles"][:5],
        }

    return db


REFERENCE_SEED_DB = load_reference_seed()


def get_match_haircut(pricing_reason: str) -> float:
    mapping = {
        "reference_seed_exact_match": 0.90,
        "reference_seed_alt_reference_match": 0.88,
        "reference_seed_family_match": 0.80,
        "reference_seed_family_inferred_match": 0.76,
        "brand_family_fallback_match": 0.72,
        "brand_type_fallback_match": 0.68,
        "brand_fallback_match": 0.62,
        "real_closed_reference_kb": 0.94,
        "seed_only_reference_kb": 0.78,
    }
    return mapping.get(pricing_reason, 0.70)


def get_condition_haircut(condition: str | None) -> float:
    value = normalize_text(condition)

    mapping = {
        "new": 1.00,
        "nos": 0.98,
        "very good": 0.95,
        "good": 0.92,
        "used": 0.88,
        "fair": 0.82,
        "poor": 0.72,
        "unknown": 0.84,
        "": 0.84,
    }

    return mapping.get(value, 0.84)


def get_listing_quality_haircut(record: dict) -> float:
    haircut = 1.0

    reference = normalize_reference(record.get("reference"))
    model = normalize_text(record.get("model"))
    title = normalize_text(record.get("title"))
    analysis_text = normalize_text(record.get("analysis_text"))
    price_found = bool(record.get("price_found"))
    has_images = bool(record.get("has_images"))
    image_count = int(record.get("image_count") or 0)
    reference_kb_hit = bool(record.get("reference_kb_hit"))
    watch_type = normalize_text(record.get("watch_type"))

    text_blob = f"{title} {analysis_text}"

    if not price_found:
        haircut *= 0.96

    if not has_images:
        haircut *= 0.90
    elif image_count <= 1:
        haircut *= 0.95
    elif image_count <= 3:
        haircut *= 0.98

    if not reference:
        haircut *= 0.95

    if not model or model == "unknown":
        haircut *= 0.93

    if not reference_kb_hit:
        haircut *= 0.97

    if watch_type == "chronograph":
        haircut *= 1.01
    if watch_type == "diver":
        haircut *= 1.01

    positive_terms = [
        "automatic",
        "powermatic",
        "chronograph",
        "chrono",
        "diver",
        "scuba",
        "sapphire",
        "ceramic",
    ]
    if any(x in text_blob for x in positive_terms):
        haircut *= 1.01

    negative_terms = [
        "vintage",
        "old",
        "antiguo",
        "sin probar",
        "no probado",
        "for parts",
        "repair",
        "reparar",
        "averiado",
        "defect",
        "defecto",
        "broken",
        "roto",
    ]
    if any(x in text_blob for x in negative_terms):
        haircut *= 0.88

    return clamp(haircut, 0.70, 1.02)


def apply_realizable_adjustment(
    raw_p25: float,
    raw_p50: float,
    raw_p75: float,
    pricing_reason: str,
    record: dict,
) -> tuple[float, float, float, float]:
    match_haircut = get_match_haircut(pricing_reason)
    condition_haircut = get_condition_haircut(record.get("condition"))
    listing_haircut = get_listing_quality_haircut(record)

    total_haircut = clamp(match_haircut * condition_haircut * listing_haircut, 0.58, 0.98)

    adjusted_p25 = round(raw_p25 * total_haircut, 2)
    adjusted_p50 = round(raw_p50 * total_haircut, 2)
    adjusted_p75 = round(raw_p75 * total_haircut, 2)

    return adjusted_p25, adjusted_p50, adjusted_p75, round(total_haircut, 4)


def build_estimate(
    raw_p25: float,
    raw_p50: float,
    raw_p75: float,
    confidence: str,
    reason: str,
    record: dict,
) -> dict:
    adj_p25, adj_p50, adj_p75, haircut = apply_realizable_adjustment(
        raw_p25=raw_p25,
        raw_p50=raw_p50,
        raw_p75=raw_p75,
        pricing_reason=reason,
        record=record,
    )

    return {
        "price_estimate_available": True,
        "raw_expected_hammer": raw_p50,
        "expected_hammer": adj_p50,
        "conservative_hammer": adj_p25,
        "optimistic_hammer": adj_p75,
        "price_confidence": confidence,
        "pricing_reason": reason,
        "applied_haircut": haircut,
    }


def estimate_hammer_from_seed(record: dict) -> dict | None:
    brand = normalize_text(record.get("brand"))
    model = normalize_text(record.get("model"))
    inferred_family = infer_family_from_text(record, brand)
    reference_candidates = extract_reference_candidates_from_record(record)

    if not brand:
        return None

    for idx, reference in enumerate(reference_candidates):
        key = (brand, reference)
        if key in REFERENCE_SEED_DB["by_reference"]:
            item = REFERENCE_SEED_DB["by_reference"][key]
            reason = "reference_seed_exact_match" if idx == 0 else "reference_seed_alt_reference_match"

            return build_estimate(
                raw_p25=item["p25"],
                raw_p50=item["p50"],
                raw_p75=item["p75"],
                confidence="medium",
                reason=reason,
                record=record,
            )

    if model:
        key = (brand, model)
        if key in REFERENCE_SEED_DB["by_family"]:
            item = REFERENCE_SEED_DB["by_family"][key]
            sample_size = int(item.get("sample_size", 1))
            confidence = "medium" if sample_size >= 3 else "low"

            return build_estimate(
                raw_p25=item["p25"],
                raw_p50=item["p50"],
                raw_p75=item["p75"],
                confidence=confidence,
                reason="reference_seed_family_match",
                record=record,
            )

    if inferred_family and inferred_family != model:
        key = (brand, inferred_family)
        if key in REFERENCE_SEED_DB["by_family"]:
            item = REFERENCE_SEED_DB["by_family"][key]
            sample_size = int(item.get("sample_size", 1))
            confidence = "medium" if sample_size >= 4 else "low"

            return build_estimate(
                raw_p25=item["p25"],
                raw_p50=item["p50"],
                raw_p75=item["p75"],
                confidence=confidence,
                reason="reference_seed_family_inferred_match",
                record=record,
            )

    return None


def estimate_hammer_from_brand_fallback(record: dict) -> dict | None:
    brand = normalize_text(record.get("brand"))
    if not brand:
        return None

    fallback_map = BRAND_FAMILY_FALLBACKS.get(brand)
    if not fallback_map:
        return None

    model = normalize_text(record.get("model"))
    inferred_family = infer_family_from_text(record, brand)
    text_blob = build_text_blob(record)
    inferred_type = normalize_text(record.get("watch_type")) or (infer_watch_type_from_text(text_blob) or "")

    if model and model in fallback_map:
        p25, p50, p75 = fallback_map[model]
        return build_estimate(
            raw_p25=p25,
            raw_p50=p50,
            raw_p75=p75,
            confidence="low",
            reason="brand_family_fallback_match",
            record=record,
        )

    if inferred_family and inferred_family in fallback_map:
        p25, p50, p75 = fallback_map[inferred_family]
        return build_estimate(
            raw_p25=p25,
            raw_p50=p50,
            raw_p75=p75,
            confidence="low",
            reason="brand_family_fallback_match",
            record=record,
        )

    if inferred_type and inferred_type in fallback_map:
        p25, p50, p75 = fallback_map[inferred_type]
        return build_estimate(
            raw_p25=p25,
            raw_p50=p50,
            raw_p75=p75,
            confidence="low",
            reason="brand_type_fallback_match",
            record=record,
        )

    if "brand_fallback" in fallback_map:
        p25, p50, p75 = fallback_map["brand_fallback"]
        return build_estimate(
            raw_p25=p25,
            raw_p50=p50,
            raw_p75=p75,
            confidence="low",
            reason="brand_fallback_match",
            record=record,
        )

    return None


def estimate_hammer_from_kb(kb_data: dict | None, record: dict) -> dict:
    if not kb_data:
        return empty_estimate("no_reference_kb_data")

    source_quality = kb_data.get("source_quality")
    stats = kb_data.get("price_stats") or {}

    count = int(stats.get("count", 0) or 0)
    real_closed_count = int(stats.get("real_closed_count", 0) or 0)
    seed_count = int(stats.get("seed_count", 0) or 0)

    p50 = stats.get("p50")
    p75 = stats.get("p75")
    min_price = stats.get("min")
    max_price = stats.get("max")

    if not count or p50 is None:
        return empty_estimate("kb_missing_price_stats")

    raw_p25 = min_price if min_price is not None else p50
    raw_p50 = p50
    raw_p75 = p75 if p75 is not None else max_price if max_price is not None else p50

    if source_quality == "real_closed_only":
        if real_closed_count < 3:
            return empty_estimate("real_closed_count_below_threshold")

        if real_closed_count >= 8:
            confidence = "high"
        elif real_closed_count >= 5:
            confidence = "medium"
        else:
            confidence = "low"

        return build_estimate(
            raw_p25=raw_p25,
            raw_p50=raw_p50,
            raw_p75=raw_p75,
            confidence=confidence,
            reason="real_closed_reference_kb",
            record=record,
        )

    if source_quality == "seed_only":
        if seed_count < 1:
            return empty_estimate("seed_only_count_below_threshold")

        confidence = "low" if seed_count < 4 else "medium"

        return build_estimate(
            raw_p25=raw_p25,
            raw_p50=raw_p50,
            raw_p75=raw_p75,
            confidence=confidence,
            reason="seed_only_reference_kb",
            record=record,
        )

    return empty_estimate("unsupported_kb_source_quality")


# Movement types that require a price penalty on fallback estimates
_ELECTRONIC_MOVEMENTS = {"quartz", "solar", "kinetic", "battery", "eco-drive"}

def _detect_movement_type(record: dict) -> str:
    """Lightweight movement type detection from title + raw_text."""
    text = normalize_text(
        (record.get("title") or "") + " " +
        (record.get("raw_text") or "") + " " +
        (record.get("movement_hint") or "")
    )
    if any(w in text for w in ["quartz", "cuarzo", "quarzo", "battery", "pile",
                                "solar", "kinetic", "eco-drive", "ecodrive"]):
        if "solar" in text: return "solar"
        if "kinetic" in text: return "kinetic"
        return "quartz"
    if any(w in text for w in ["automatic", "automatico", "automático", "powermatic",
                                "self-winding", "rotor", "eta 28", "nh35", "nh36"]):
        return "automatic"

    # ── Model-specific quartz rules ──────────────────────────────────────────
    # Tissot Seastar 1000 Chronograph: the PM80 automatic is NEVER a chronograph.
    # All Seastar chronographs (T120.417) are quartz. Same for PRX Chronograph.
    brand = normalize_text(record.get("brand") or "")
    family = normalize_text(infer_family_from_text(record, brand) or "")
    is_chrono = any(w in text for w in ["chronograph", "chrono", "cronografo"])
    if brand == "tissot" and "seastar" in family and is_chrono:
        return "quartz"
    if brand == "tissot" and "prx" in family and is_chrono:
        return "quartz"
    # T120.417.xx refs are all quartz chronographs
    if "t120.417" in text or "t120417" in text:
        return "quartz"

    return "unknown"


def apply_auction_price_engine(record: dict) -> dict:
    is_ladies, ladies_reason = infer_ladies_flag(record)
    record["is_ladies_inferred"] = is_ladies
    record["ladies_inference_reason"] = ladies_reason

    if is_ladies:
        record["auction_estimate"] = empty_estimate(f"ladies_watch_blocked:{ladies_reason}")
        return record

    # Detect movement type for price adjustment
    movement_type = _detect_movement_type(record)
    record["movement_hint"] = record.get("movement_hint") or movement_type

    seed_estimate = estimate_hammer_from_seed(record)
    if seed_estimate:
        record["auction_estimate"] = seed_estimate
        return record

    kb_data = record.get("reference_kb_data")
    kb_estimate = estimate_hammer_from_kb(kb_data, record)
    if kb_estimate.get("price_estimate_available"):
        record["auction_estimate"] = kb_estimate
        return record

    brand_fallback_estimate = estimate_hammer_from_brand_fallback(record)
    if brand_fallback_estimate:
        # Apply quartz penalty: brand_family_fallbacks are calibrated for automatic watches.
        # Quartz versions are worth ~45% of the automatic estimate on Catawiki.
        # Applied to ALL brand_family_fallback estimates when movement is electronic —
        # no need to check specific families since all fallbacks assume mechanical.
        if movement_type in _ELECTRONIC_MOVEMENTS:
            est = brand_fallback_estimate
            penalty = 0.45
            est["raw_expected_hammer"] = round((est.get("raw_expected_hammer") or 0) * penalty, 2)
            est["expected_hammer"]     = round((est.get("expected_hammer") or 0) * penalty, 2)
            est["conservative_hammer"] = round((est.get("conservative_hammer") or 0) * penalty, 2)
            est["optimistic_hammer"]   = round((est.get("optimistic_hammer") or 0) * penalty, 2)
            est["pricing_reason"]      = (est.get("pricing_reason") or "") + "_quartz_penalty"
            brand_fallback_estimate = est

        record["auction_estimate"] = brand_fallback_estimate
        return record

    record["auction_estimate"] = empty_estimate("no_reference_family_or_brand_fallback_match")
    return record
