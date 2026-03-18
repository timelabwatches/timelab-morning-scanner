# knowledge/lookup_reference_kb.py

import json
from pathlib import Path

from knowledge.normalize_reference import normalize_reference


KB_PATH = Path("knowledge_base/reference_knowledge_base.json")


def load_reference_kb() -> dict:
    """
    Load reference knowledge base if available.
    """

    if not KB_PATH.exists():
        return {}

    with KB_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def lookup_reference(reference: str | None) -> dict | None:
    """
    Lookup a reference in the knowledge base.
    """

    key = normalize_reference(reference)

    if not key:
        return None

    kb = load_reference_kb()

    return kb.get(key)