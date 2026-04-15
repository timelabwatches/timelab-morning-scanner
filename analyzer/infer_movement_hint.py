import re
from analyzer.text_utils import build_analysis_text


MOVEMENT_KEYWORDS = {
    "quartz": [
        "quartz", "cuarzo", "quarzo", "battery", "pile",
        "eco-drive", "eco drive", "solar", "kinetic",
        "v739", "radio controlled", "atomic time",
    ],
    "automatic": [
        "automatic", "automático", "automatico", "automat",
        "self winding", "self-winding", "powermatic", "powermatic 80",
        "co-axial", "co axial", "eta 28", "eta 25",
        "miyota", "nh35", "nh36", "4r35", "4r36", "6r15",
        "7s26", "7s36", "srpc", "srpe",
    ],
    "manual": [
        "manual", "hand-wound", "hand wound", "handwound",
        "carica manuale", "remontage manuel", "handaufzug",
        "cuerda manual", "carga manual",
    ],
}

# Reference-based rules: model reference patterns → movement type
# Applied AFTER keyword scan, only when keywords not found
_REF_QUARTZ_PATTERNS = [
    r"\b7t[34289][0-9]\b",          # Seiko quartz: 7T42, 7T32, 7T82, 7T92
    r"\b5y2[0-9]\b",                # Seiko 5Y22, 5Y23 (solar)
    r"\bv739\b",                    # Seiko V739 solar
    r"\bt120\.41[0-9]\b",          # Tissot Seastar Chrono quartz T120.417
    r"\bt120417\b",
    r"\bpr50\b",                    # Tissot PR50 (quartz)
    r"\ba927\b",                    # Seiko A927 digital
    r"\bvk63\b",                    # Miyota VK63 meca-quartz (microbrands)
    r"\bvk64\b",
]

_REF_AUTOMATIC_PATTERNS = [
    r"\bsnk[a-z0-9]{2,8}\b",       # Seiko 5: SNK809, SNKL07...
    r"\bsnab[a-z0-9]{2,8}\b",      # Seiko 5
    r"\bsrp[a-z][0-9]",             # Seiko Prospex auto: SRPC, SRPE, SRPD...
    r"\b7[6][0-9]{2}[- ][0-9]",     # Seiko 76xx vintage automatic
    r"\b6[12569][0-9]{2}[- ][0-9]", # Seiko 61xx/62xx/6119/6138/6139 vintage auto
    r"\bra-[a-z]{2}[0-9]",          # Orient RA-xx automatic
]


def infer_movement_hint(record: dict) -> str | None:
    """
    Infer movement type from text and reference patterns.
    Returns: 'automatic' | 'manual' | 'quartz' | 'solar' | 'kinetic' | None
    """
    text = build_analysis_text(record).lower()

    # Step 1: keyword scan
    for movement, keywords in MOVEMENT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                if keyword == "solar" or keyword == "v739":
                    return "solar"
                if keyword == "kinetic":
                    return "kinetic"
                return movement

    # Step 2: reference-number based detection
    for pattern in _REF_QUARTZ_PATTERNS:
        if re.search(pattern, text):
            return "quartz"

    for pattern in _REF_AUTOMATIC_PATTERNS:
        if re.search(pattern, text):
            return "automatic"

    # Step 3: model-specific rules
    brand = (record.get("brand") or "").lower().strip()
    model = (record.get("model") or "").lower().strip()
    title = (record.get("title") or "").lower()

    # Tissot Seastar Chronograph = always quartz (PM80 ≠ chrono)
    if brand == "tissot" and "seastar" in model and "chronograph" in text:
        return "quartz"
    # Tissot V8 = quartz
    if brand == "tissot" and "v8" in title and "v8" not in model.replace("v8", ""):
        return "quartz"
    # Tissot PR50 = quartz
    if brand == "tissot" and "pr50" in text:
        return "quartz"

    return None
