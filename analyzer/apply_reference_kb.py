from knowledge.lookup_reference_kb import lookup_reference


def apply_reference_kb(record: dict) -> dict:
    """
    If reference exists in knowledge base, use it to refine analysis fields.

    Priority:
    - keep existing field if KB has no value
    - override when KB has a confident structured value
    """

    reference = record.get("reference")

    kb_hit = lookup_reference(reference)

    if not kb_hit:
        record["reference_kb_hit"] = False
        record["reference_kb_data"] = None
        return record

    record["reference_kb_hit"] = True
    record["reference_kb_data"] = kb_hit

    if kb_hit.get("brand"):
        record["brand"] = kb_hit["brand"]

    if kb_hit.get("model_family"):
        record["model"] = kb_hit["model_family"]

    if kb_hit.get("watch_type"):
        record["watch_type"] = kb_hit["watch_type"]

    if kb_hit.get("movement_hint"):
        record["movement_hint"] = kb_hit["movement_hint"]

    if "is_chronograph" in kb_hit:
        record["is_chronograph"] = bool(kb_hit["is_chronograph"])

    if "is_ladies" in kb_hit:
        record["is_ladies"] = bool(kb_hit["is_ladies"])

    if record.get("movement_hint") == "quartz":
        record["is_quartz"] = True

    if record.get("movement_hint") == "automatic":
        record["is_automatic"] = True

    return record