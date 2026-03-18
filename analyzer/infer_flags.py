# analyzer/infer_flags.py

def infer_flags(record: dict) -> dict:
    """
    Infer simple boolean flags from analyzed record.

    Important:
    - watch type and movement are independent
    - a watch can be chronograph + quartz
    - a watch can be chronograph + automatic
    """

    watch_type = record.get("watch_type")
    gender = record.get("gender")
    movement_hint = record.get("movement_hint")

    return {
        "is_chronograph": watch_type == "chronograph",
        "is_ladies": gender == "female",
        "is_quartz": movement_hint == "quartz",
        "is_automatic": movement_hint == "automatic",
    }