import re


def normalize_reference(value: str | None) -> str | None:
    """
    Normalize reference into a stable lookup key.
    """

    if not value:
        return None

    ref = value.strip().lower()
    ref = ref.replace(" ", "")
    ref = ref.replace("-", "")
    ref = ref.replace("_", "")
    ref = ref.replace("/", "")

    ref = re.sub(r"[^a-z0-9.]", "", ref)

    return ref or None