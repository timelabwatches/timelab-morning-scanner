# state_ebay.py
# -*- coding: utf-8 -*-

import json
import os
import time
from typing import Any, Dict, Optional


def _now_ts() -> int:
    return int(time.time())


def load_state(path: str) -> Dict[str, Any]:
    """
    State schema:
    {
      "version": 1,
      "items": {
        "<item_id>": {"last_sent_ts": 1700000000, "last_price": 123.45}
      }
    }
    """
    if not os.path.exists(path):
        return {"version": 1, "items": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"version": 1, "items": {}}
        if "items" not in data or not isinstance(data["items"], dict):
            data["items"] = {}
        if "version" not in data:
            data["version"] = 1
        return data
    except Exception:
        return {"version": 1, "items": {}}


def save_state(path: str, state: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def is_in_cooldown(item_id: str, state: Dict[str, Any], cooldown_h: int) -> bool:
    if cooldown_h <= 0:
        return False
    items = state.get("items", {})
    rec = items.get(str(item_id))
    if not isinstance(rec, dict):
        return False
    last_ts = rec.get("last_sent_ts")
    if not isinstance(last_ts, int):
        return False
    return (_now_ts() - last_ts) < int(cooldown_h) * 3600


def should_repost_due_to_price_drop(
    item_id: str,
    current_price: Optional[float],
    state: Dict[str, Any],
    repost_pct: float,
) -> bool:
    """
    True if current_price <= last_price * (1 - repost_pct/100)
    Only applies if we have last_price stored.
    """
    if current_price is None:
        return False
    if repost_pct <= 0:
        return False
    items = state.get("items", {})
    rec = items.get(str(item_id))
    if not isinstance(rec, dict):
        return False
    last_price = rec.get("last_price")
    if not isinstance(last_price, (int, float)):
        return False

    try:
        last_price_f = float(last_price)
        cur_price_f = float(current_price)
        if last_price_f <= 0:
            return False
        threshold = last_price_f * (1.0 - repost_pct / 100.0)
        return cur_price_f <= threshold
    except Exception:
        return False


def mark_sent(item_id: str, price: Optional[float], state: Dict[str, Any]) -> None:
    items = state.setdefault("items", {})
    rec = items.get(str(item_id), {})
    if not isinstance(rec, dict):
        rec = {}
    rec["last_sent_ts"] = _now_ts()
    if price is not None:
        try:
            rec["last_price"] = float(price)
        except Exception:
            pass
    items[str(item_id)] = rec