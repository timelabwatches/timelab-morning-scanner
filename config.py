import os
from dataclasses import dataclass
from typing import Set


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        return float(raw.replace(",", "."))
    except Exception:
        return default


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except Exception:
        return default


def env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def env_set(name: str, default: str) -> Set[str]:
    raw = os.getenv(name, default)
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


@dataclass
class Settings:
    min_net_eur: float
    min_net_roi: float
    min_match_score: int
    allow_fake_risk: Set[str]

    buy_max_mult: float

    catwiki_commission: float
    payment_processing: float
    packaging_eur: float
    misc_eur: float
    ship_arbitrage_eur: float
    effective_tax_rate_on_profit: float

    http_timeout: int

    ebay_client_id: str
    ebay_client_secret: str
    ebay_marketplace_id: str
    ebay_limit: int
    ebay_throttle_s: float
    ebay_default_category_id: str
    detail_fetch_n: int
    ebay_detail_throttle_s: float
    ebay_allowed_category_ids: Set[str]

    telegram_bot_token: str
    telegram_chat_id: str
    tg_max_len: int

    cooldown_hours: int
    price_drop_repost: float

    target_list_path: str
    model_master_path: str
    comparables_path: str
    target_stats_path: str
    state_path: str


def load_settings() -> Settings:
    ebay_default_category_id = env_str("EBAY_DEFAULT_CATEGORY_ID", "31387")
    ebay_allowed_category_ids = {
        x.strip()
        for x in os.getenv("EBAY_ALLOWED_CATEGORY_IDS", ebay_default_category_id).split(",")
        if x.strip()
    }

    return Settings(
        min_net_eur=env_float("MIN_NET_EUR", 20.0),
        min_net_roi=env_float("MIN_NET_ROI", 0.05),
        min_match_score=env_int("MIN_MATCH_SCORE", 60),
        allow_fake_risk=env_set("ALLOW_FAKE_RISK", "low,medium"),
        buy_max_mult=env_float("BUY_MAX_MULT", 1.25),
        catwiki_commission=env_float("CATWIKI_COMMISSION", 0.125),
        payment_processing=env_float("PAYMENT_PROCESSING", 0.0),
        packaging_eur=env_float("PACKAGING_EUR", 5.0),
        misc_eur=env_float("MISC_EUR", 5.0),
        ship_arbitrage_eur=env_float("SHIP_ARBITRAGE_EUR", 35.0),
        effective_tax_rate_on_profit=env_float("EFFECTIVE_TAX_RATE_ON_PROFIT", 0.15),
        http_timeout=env_int("HTTP_TIMEOUT", 25),
        ebay_client_id=env_str("EBAY_CLIENT_ID", ""),
        ebay_client_secret=env_str("EBAY_CLIENT_SECRET", ""),
        ebay_marketplace_id=env_str("EBAY_MARKETPLACE_ID", "EBAY_ES"),
        ebay_limit=env_int("EBAY_LIMIT", 50),
        ebay_throttle_s=env_float("EBAY_THROTTLE_S", 0.35),
        ebay_default_category_id=ebay_default_category_id,
        detail_fetch_n=env_int("DETAIL_FETCH_N", 35),
        ebay_detail_throttle_s=env_float("EBAY_DETAIL_THROTTLE_S", 0.20),
        ebay_allowed_category_ids=ebay_allowed_category_ids,
        telegram_bot_token=env_str("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=env_str("TELEGRAM_CHAT_ID", ""),
        tg_max_len=env_int("TG_MAX_LEN", 3500),
        cooldown_hours=env_int("COOLDOWN_HOURS", 72),
        price_drop_repost=env_float("PRICE_DROP_REPOST", 0.10),
        target_list_path=env_str("TARGET_LIST_PATH", "target_list.json"),
        model_master_path=env_str("MODEL_MASTER_PATH", "data/model_master.json"),
        comparables_path=env_str("COMPARABLES_PATH", "data/market_comp_db.json"),
        target_stats_path=env_str("TARGET_STATS_PATH", "data/target_stats.json"),
        state_path=env_str("EBAY_STATE_PATH", "state/state_ebay.json"),
    )


def validate_settings(settings: Settings) -> None:
    errors = []

    if not settings.ebay_client_id:
        errors.append("Missing EBAY_CLIENT_ID")
    if not settings.ebay_client_secret:
        errors.append("Missing EBAY_CLIENT_SECRET")
    if not settings.telegram_bot_token:
        errors.append("Missing TELEGRAM_BOT_TOKEN")
    if not settings.telegram_chat_id:
        errors.append("Missing TELEGRAM_CHAT_ID")

    if settings.min_net_eur < 0:
        errors.append("MIN_NET_EUR must be >= 0")
    if settings.min_net_roi < 0:
        errors.append("MIN_NET_ROI must be >= 0")
    if settings.min_match_score < 0 or settings.min_match_score > 100:
        errors.append("MIN_MATCH_SCORE must be between 0 and 100")
    if settings.buy_max_mult <= 0:
        errors.append("BUY_MAX_MULT must be > 0")
    if settings.http_timeout <= 0:
        errors.append("HTTP_TIMEOUT must be > 0")
    if settings.ebay_limit <= 0:
        errors.append("EBAY_LIMIT must be > 0")
    if settings.detail_fetch_n < 0:
        errors.append("DETAIL_FETCH_N must be >= 0")
    if settings.cooldown_hours < 0:
        errors.append("COOLDOWN_HOURS must be >= 0")
    if settings.price_drop_repost < 0 or settings.price_drop_repost > 1:
        errors.append("PRICE_DROP_REPOST must be between 0 and 1")

    if not settings.ebay_default_category_id:
        errors.append("Missing EBAY_DEFAULT_CATEGORY_ID")
    if not settings.ebay_allowed_category_ids:
        errors.append("EBAY_ALLOWED_CATEGORY_IDS cannot be empty")
    if not settings.target_list_path:
        errors.append("Missing TARGET_LIST_PATH")
    if not settings.model_master_path:
        errors.append("Missing MODEL_MASTER_PATH")
    if not settings.comparables_path:
        errors.append("Missing COMPARABLES_PATH")
    if not settings.state_path:
        errors.append("Missing EBAY_STATE_PATH")

    if errors:
        raise RuntimeError("Invalid configuration:\n- " + "\n- ".join(errors))