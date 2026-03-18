# analyzer/infer_model.py

from analyzer.text_utils import build_analysis_text


MODEL_ALIASES = {
    "seiko": {
        "prospex": ["prospex", "samurai", "turtle", "sumo", "alpinist", "marine master", "marinemaster", "shogun"],
        "sportura": ["sportura"],
        "arctura": ["arctura"],
        "presage": ["presage", "cocktail time", "cocktail"],
        "5 sports": ["5 sports", "seiko 5", "seiko5", "5sport"],
        "astron": ["astron"],
        "kinetic diver": ["kinetic diver", "kinetic scuba", "scuba", "diver 200", "200m diver"],
    },
    "tissot": {
        "prx": ["prx"],
        "seastar": ["seastar", "sea star"],
        "le locle": ["le locle"],
        "visodate": ["visodate"],
        "t-race": ["t-race", "t race", "trace"],
        "carson": ["carson"],
        "couturier": ["couturier"],
        "tradition": ["tradition"],
        "pr 100": ["pr 100", "pr100"],
    },
    "omega": {
        "seamaster": ["seamaster"],
        "speedmaster": ["speedmaster"],
        "constellation": ["constellation"],
        "de ville": ["de ville", "deville"],
        "geneve": ["geneve", "genève"],
        "dynamic": ["dynamic"],
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
    },
    "certina": {
        "ds action": ["ds action", "ds-action"],
        "ds podium": ["ds podium", "ds-podium"],
        "ds-1": ["ds-1", "ds 1"],
        "ds-2": ["ds-2", "ds 2"],
        "ds first": ["ds first", "ds-first"],
    },
    "hamilton": {
        "khaki field": ["khaki field"],
        "khaki navy": ["khaki navy", "khaki scuba", "navy scuba"],
        "jazzmaster": ["jazzmaster", "jazz master"],
        "ventura": ["ventura"],
        "american classic": ["american classic"],
    },
    "tagheuer": {
        "formula 1": ["formula 1", "formula1"],
        "aquaracer": ["aquaracer", "aqua racer"],
        "kirium": ["kirium"],
        "link": ["link"],
        "monaco": ["monaco"],
        "carrera": ["carrera"],
    },
    "baume": {
        "classima": ["classima"],
        "riviera": ["riviera"],
        "clifton": ["clifton"],
        "hampton": ["hampton"],
    },
    "junghans": {
        "max bill": ["max bill"],
        "meister": ["meister"],
    },
    "citizen": {
        "promaster": ["promaster"],
        "tsuyosa": ["tsuyosa"],
        "eco-drive": ["eco-drive", "eco drive"],
    },
    "bulova": {
        "marine star": ["marine star", "marinestar"],
        "lunar pilot": ["lunar pilot", "lunarpilot"],
        "precisionist": ["precisionist"],
    },
}


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(str(value).lower().split())


def normalize_reference(value: str | None) -> str:
    if not value:
        return ""
    value = str(value).lower().strip()
    value = value.replace(".", "")
    value = value.replace("-", "")
    value = value.replace("/", "")
    value = value.replace("_", "")
    value = value.replace(" ", "")
    return value


def infer_model_from_reference(reference: str | None, brand: str | None) -> str | None:
    if not reference or not brand:
        return None

    ref = normalize_reference(reference)

    if brand == "tissot":
        if ref.startswith("t122410"):
            return "carson"
        if ref.startswith("t048417") or ref.startswith("t048"):
            return "t-race"
        if ref.startswith("t115417") or ref.startswith("t115"):
            return "t-race"
        if ref.startswith("t137"):
            return "prx"
        if ref.startswith("t120"):
            return "seastar"
        if ref.startswith("t006"):
            return "le locle"
        if ref.startswith("t019"):
            return "visodate"
        if ref.startswith("t063"):
            return "tradition"
        if ref.startswith("t101"):
            return "pr 100"

    if brand == "seiko":
        if ref.startswith("srpd") or ref.startswith("srpb") or ref.startswith("srp777") or ref.startswith("sbdc") or ref.startswith("sne"):
            return "prospex"
        if ref.startswith("snae"):
            return "sportura"
        if ref.startswith("snl"):
            return "arctura"
        if ref.startswith("ska"):
            return "kinetic diver"
        if ref.startswith("sarx") or ref.startswith("ssa") or ref.startswith("srpe"):
            return "presage"
        if ref.startswith("skx") or ref.startswith("srp") or ref.startswith("srpe") or ref.startswith("srph"):
            return "5 sports"

    if brand == "longines":
        if ref.startswith("l3"):
            return "conquest"
        if ref.startswith("l4"):
            return "flagship"
        if ref.startswith("l2"):
            return "master collection"
        if ref.startswith("l5"):
            return "dolcevita"

    if brand == "certina":
        if ref.startswith("c032"):
            return "ds action"
        if ref.startswith("c001") or ref.startswith("c034"):
            return "ds podium"
        if ref.startswith("c006"):
            return "ds-1"
        if ref.startswith("c024"):
            return "ds-2"

    if brand == "hamilton":
        if ref.startswith("h684") or ref.startswith("h704"):
            return "khaki field"
        if ref.startswith("h823"):
            return "khaki navy"
        if ref.startswith("h325"):
            return "jazzmaster"
        if ref.startswith("h244"):
            return "ventura"

    if brand == "tagheuer":
        if ref.startswith("wah") or ref.startswith("caz"):
            return "formula 1"
        if ref.startswith("waf") or ref.startswith("way") or ref.startswith("wan"):
            return "aquaracer"
        if ref.startswith("wl") or ref.startswith("wt"):
            return "link"
        if ref.startswith("cv"):
            return "carrera"

    if brand == "baume":
        if ref.startswith("m0a10") or ref.startswith("moa10"):
            return "classima"
        if ref.startswith("m0a08") or ref.startswith("moa08"):
            return "riviera"
        if ref.startswith("m0a10") or ref.startswith("moa10"):
            return "clifton"

    if brand == "junghans":
        if ref.startswith("027") or ref.startswith("041"):
            return "max bill"

    if brand == "citizen":
        if ref.startswith("bn") or ref.startswith("ny"):
            return "promaster"
        if ref.startswith("nj015"):
            return "tsuyosa"

    if brand == "bulova":
        if ref.startswith("96b") or ref.startswith("98b"):
            return "marine star"

    return None


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


def infer_model_from_keywords(text: str, brand: str | None) -> str | None:
    if not brand:
        return None

    scores = score_family_matches(text, brand)

    if not scores:
        return None

    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[0][0]


def infer_model(record: dict, brand: str | None) -> str | None:
    """
    Infer watch family using:
    1) reference
    2) keyword scoring on analysis text
    """

    text = normalize_text(build_analysis_text(record))
    reference = record.get("reference")

    model_from_ref = infer_model_from_reference(reference, brand)
    if model_from_ref:
        return model_from_ref

    return infer_model_from_keywords(text, brand)