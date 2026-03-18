import json
import os
import time


def load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_state(path: str, state: dict) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def is_repost_due_to_price_drop(
    item_id: str,
    price: float,
    state: dict,
    cooldown_hours: int,
    price_drop_repost: float,
) -> bool:
    record = state.get(item_id)
    if not isinstance(record, dict):
        return False

    last_ts = float(record.get("last_ts", 0.0))
    last_price = float(record.get("last_price", price))

    age_hours = (time.time() - last_ts) / 3600.0
    if age_hours >= cooldown_hours:
        return False

    if last_price <= 0:
        return False

    drop = (last_price - price) / last_price
    return drop >= price_drop_repost


def is_in_cooldown(
    item_id: str,
    price: float,
    state: dict,
    cooldown_hours: int,
    price_drop_repost: float,
) -> bool:
    record = state.get(item_id)
    if not isinstance(record, dict):
        return False

    last_ts = float(record.get("last_ts", 0.0))
    last_price = float(record.get("last_price", price))

    age_hours = (time.time() - last_ts) / 3600.0
    if age_hours >= cooldown_hours:
        return False

    if last_price > 0:
        drop = (last_price - price) / last_price
        if drop >= price_drop_repost:
            return False

    return True


def update_state(item_id: str, price: float, state: dict) -> None:
    state[item_id] = {
        "last_ts": time.time(),
        "last_price": price,
    }