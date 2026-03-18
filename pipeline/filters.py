# pipeline/filters.py

import re
from typing import Optional


EU_ISO2 = {
    "ES", "FR", "DE", "IT", "PT", "BE", "NL", "LU", "IE", "AT", "FI", "SE",
    "DK", "PL", "CZ", "SK", "SI", "HR", "HU", "RO", "BG", "GR", "CY", "MT",
    "LV", "LT", "EE",
}

GLOBAL_ACCESSORY_TERMS = {
    "strap", "band", "bracelet", "armband", "cinturino", "bracciale",
    "correa", "pulsera", "brazalete",
    "buckle", "hebilla", "fibbie", "fibbia",
    "clasp", "deployant", "déployant", "faltschließe", "faltschliesse",
    "cierre", "cierres",
    "links", "link", "eslabon", "eslabón", "eslabones", "endlink", "end link",
    "spring bar", "springbar", "pasadores", "tool", "herramienta",
    "instruction", "instructions", "manual", "manuale", "booklet", "libretto",
    "gebrauchsanleitung", "catalog", "catalogue", "catalogo",
    "box only", "only box", "solo caja", "caja sola", "caja solo",
    "watch box", "uhrenbox", "scatola", "scatola orologio", "boite", "boîte",
}

WATCH_INDICATORS = {
    "watch", "wristwatch", "reloj", "orologio", "montre", "uhr",
    "automatic", "automatik", "automatique",
    "chronograph", "chrono", "gmt",
    "date", "diver", "sub", "seamaster", "aquaracer", "hydroconquest",
    "orologio uomo", "orologio donna",
}

ACCESSORY_CONTEXT_TERMS = {
    "for", "para", "per", "für", "fur", "pour",
    "models", "modelos", "modele", "modelli",
    "fits", "compatible", "compatibile", "kompatibel",
    "does not fit", "no se adapta", "non si adatta", "passt nicht",
}

INCOMPLETE_HARD_TERMS = {
    "sin mecanismo", "falta movimiento", "falta el movimiento", "caja vacía", "caja vacia",
    "sin maquinaria", "sin máquina", "sin maquina", "solo caja", "caja sola",
    "senza meccanismo", "senza meccanica", "senza macchina", "manca il movimento",
    "movimento mancante", "solo cassa", "cassa vuota",
    "boîtier seul", "boitier seul", "sans mecanisme", "sans mécanisme", "sans mouvement",
    "ohne uhrwerk", "ohne werk", "gehäuse ohne", "gehaeuse ohne",
    "box only", "only box", "with box only", "solo caja", "caja sola", "caja solo",
    "watch box", "uhrenbox", "scatola", "scatola orologio",
    "for parts", "parts only", "movement only", "only movement", "solo movimiento",
    "solo calibro", "solo calibre", "part only", "only part",
    "spares", "ricambi", "pièces", "pieces", "piece", "pieza", "piezas", "ersatzteile",
    "watch case", "case only", "dial only", "only dial", "boitier", "boîtier",
    "solo esfera", "only hands", "set of hands", "crown only", "stem only",
    "center wheel", "wheel part", "balance staff", "main plate", "mainplate",
}

GLOBAL_HARD_BAD_TERMS = {
    "broken", "not working", "doesn't work", "does not work",
    "defect", "defective", "as is", "untested", "not tested",
    "no funciona", "averiado", "averiada", "sin funcionar",
    "non funziona", "guasto", "ne fonctionne pas", "defekt", "funktioniert nicht",
    "missing",
    "replica", "copy", "imitacion", "imitación", "imitation", "fake",
    "for parts", "parts only", "part only", "movement only", "only movement", "solo movimiento",
}

AUTO_CONTRADICTIONS = {
    "quartz", "cuarzo", "battery", "pile",
    "manual", "hand-wound", "hand wound", "handwound",
    "carica manuale", "a carica manuale", "remontage manuel", "handaufzug",
    "solar", "kinetic",
}

LOW_QUALITY_TERMS = {
    "read the description",
    "see description",
    "please read",
    "balance ok",
    "working ok",
}

PARTS_TERMS = {
    "movement", "movimiento", "movimento", "uhrwerk", "werk",
    "caliber", "calibre", "ebauche", "ébauche",
    "dial", "esfera", "quadrante", "zifferblatt",
    "case", "watch case", "caja", "cassa", "boitier", "boîtier", "gehäuse", "gehaeuse",
    "hands", "manecillas", "zeiger", "lancette",
    "crown", "corona", "stem", "tija", "winding stem",
    "crystal", "glass", "plexi", "bezel", "insert",
    "rotor", "balance", "mainplate", "main plate", "bridge", "bridges",
    "wheel", "center wheel", "escape wheel", "balance staff",
    "donor", "spare", "spares", "ricambi", "part", "parts", "piece", "pieces", "pieza", "piezas",
    "uhrenbox", "watch box", "scatola", "boite", "boîte",
}

STRONG_WHOLE_WATCH_TERMS = {
    "wristwatch", "reloj", "orologio", "montre", "uhr",
    "full set", "new with box", "new with box and papers",
    "nuevo con caja", "nuevo con caja y documentación",
}

LADIES_TERMS = {
    "ladies", "lady", "women", "woman", "female",
    "mujer", "señora", "senora", "dama", "chica",
    "femme", "donna", "girl", "girls",
}

MEN_TERMS = {
    "men", "man's", "mens", "male", "hombre", "caballero", "homme", "uomo", "gent",
}

PARTS_ONLY_PATTERNS = [
    r"\bwatch box\b",
    r"\buhrenbox\b",
    r"\bscatola\b",
    r"\bwatch case\b",
    r"\bcase only\b",
    r"\bdial only\b",
    r"\bonly dial\b",
    r"\bmovement only\b",
    r"\bonly movement\b",
    r"\bfor parts\b",
    r"\bparts only\b",
    r"\bpart only\b",
    r"\bsolo movimiento\b",
    r"\bsolo calibre\b",
    r"\bsolo calibro\b",
    r"\bsolo caja\b",
    r"\bcaja sola\b",
    r"\bbo[iî]tier seul\b",
    r"\bwithout movement\b",
    r"\bsans mouvement\b",
    r"\bohne uhrwerk\b",
    r"\bcenter wheel\b",
    r"\bwheel part\b",
]

CASE_SIZE_PATTERN = r"\b(\d{1,2})\s?mm\b"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def norm_tokens(items: list[str]) -> list[str]:
    return [norm(x) for x in (items or []) if norm(x)]


def has_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms if term)


def count_hits(text: str, terms: set[str]) -> int:
    return sum(1 for term in terms if term and term in text)


def is_eu_location(location_text: str) -> bool:
    if not location_text:
        return True

    match = re.search(r"\b([A-Z]{2})\b\s*$", location_text.strip())
    if match:
        return match.group(1).upper() in EU_ISO2

    lowered = norm(location_text)
    common_names = [
        "spain", "españa", "france", "francia", "germany", "alemania", "italy", "italia",
        "portugal", "belgium", "bélgica", "netherlands", "países bajos", "austria",
        "ireland", "finland", "sweden", "denmark", "poland", "czech", "slovakia",
        "slovenia", "croatia", "hungary", "romania", "bulgaria", "greece",
        "luxembourg", "latvia", "lithuania", "estonia", "cyprus", "malta",
    ]
    return any(name in lowered for name in common_names)


def looks_like_accessory(text: str) -> bool:
    t = norm(text)
    if not t:
        return False

    if not has_any(t, GLOBAL_ACCESSORY_TERMS):
        return False

    has_watch = has_any(t, WATCH_INDICATORS)
    has_context = has_any(t, ACCESSORY_CONTEXT_TERMS)

    if has_context:
        return True

    if not has_watch:
        return True

    strong_accessory = {
        "strap", "correa", "pulsera", "buckle", "hebilla", "clasp",
        "deployant", "links", "eslabon", "eslabón", "watch box", "uhrenbox", "scatola",
    }

    if has_any(t, strong_accessory) and not any(x in t for x in {"watch", "reloj", "orologio", "montre", "uhr"}):
        return True

    return False


def infer_case_size_mm(text: str) -> int | None:
    t = norm(text)
    if not t:
        return None

    matches = re.findall(CASE_SIZE_PATTERN, t)
    for match in matches:
        try:
            mm = int(match)
            if 10 <= mm <= 60:
                return mm
        except ValueError:
            continue

    return None


def looks_like_ladies_small_watch(text: str) -> bool:
    t = norm(text)
    if not t:
        return False

    if has_any(t, MEN_TERMS):
        return False

    ladies_signal = has_any(t, LADIES_TERMS)
    mm = infer_case_size_mm(t)

    if ladies_signal:
        return True

    # Bloqueamos tamaños muy pequeños si no hay señal masculina.
    if mm is not None and mm <= 28:
        return True

    return False


def looks_like_movement_or_parts(text: str) -> bool:
    t = norm(text)
    if not t:
        return False

    for pattern in PARTS_ONLY_PATTERNS:
        if re.search(pattern, t):
            return True

    if has_any(t, {"movement only", "only movement", "for parts", "parts only", "part only"}):
        return True

    parts_hits = count_hits(t, PARTS_TERMS)
    whole_watch_hits = count_hits(t, STRONG_WHOLE_WATCH_TERMS)

    if parts_hits >= 2 and whole_watch_hits == 0:
        return True

    movement_core = {"movement", "movimiento", "movimento", "uhrwerk", "werk", "caliber", "calibre"}
    if has_any(t, movement_core):
        full_watch_context = {
            "wristwatch", "reloj completo", "orologio completo", "montre complète", "watch complete",
            "with bracelet", "with strap", "con correa", "con brazalete",
        }
        if not has_any(t, full_watch_context):
            return True

    isolated_part_terms = {
        "dial", "esfera", "quadrante", "zifferblatt",
        "case", "watch case", "caja", "cassa", "boitier", "boîtier",
        "hands", "manecillas", "crown", "corona", "stem", "tija",
        "wheel", "center wheel", "watch box", "uhrenbox", "scatola",
    }
    if has_any(t, isolated_part_terms) and whole_watch_hits == 0:
        return True

    return False


def has_incomplete_hard_terms(text: str) -> bool:
    t = norm(text)
    return any(term in t for term in INCOMPLETE_HARD_TERMS)


def is_low_quality_listing(text: str) -> bool:
    t = norm(text)
    return any(term in t for term in LOW_QUALITY_TERMS)


def reject_reason(text: str, location_text: str = "", eu_only: bool = True) -> Optional[str]:
    t = norm(text)

    if not t:
        return "empty_text"

    if eu_only and not is_eu_location(location_text):
        return "non_eu"

    if looks_like_accessory(t):
        return "accessory"

    if looks_like_movement_or_parts(t):
        return "movement_or_parts"

    if has_incomplete_hard_terms(t):
        return "incomplete_or_manual_or_parts"

    if looks_like_ladies_small_watch(t):
        return "ladies_or_too_small"

    if has_any(t, GLOBAL_HARD_BAD_TERMS):
        return "hard_bad_terms"

    if is_low_quality_listing(t):
        return "low_quality_listing"

    return None


def title_passes_target_filters(text: str, must_include: list[str], must_exclude: list[str], keywords: list[str]) -> bool:
    t = norm(text)

    include_tokens = norm_tokens(must_include or [])
    if include_tokens and not all(token in t for token in include_tokens):
        return False

    exclude_tokens = norm_tokens(must_exclude or [])
    if exclude_tokens and any(token in t for token in exclude_tokens):
        return False

    keyword_set = set(norm_tokens(keywords or []))
    expects_automatic = any(
        token in keyword_set
        for token in {"automatic", "powermatic 80", "co-axial", "co axial", "calibre", "caliber"}
    )

    if expects_automatic and has_any(t, AUTO_CONTRADICTIONS):
        return False

    return True