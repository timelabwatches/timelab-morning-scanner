#!/usr/bin/env python3
"""TIMELAB — RealCash scanner. Reuses secondhand_core.py."""
from secondhand_core import SiteConfig, run_scan, env_str, env_int, env_float

CFG = SiteConfig(
    shop_id="realcash",
    shop_label="RealCash",
    base_url="https://realcash.es",
    catalog_urls=[
        "https://realcash.es/relojes/",
    ],
    # RealCash uses WooCommerce: product URLs are /producto/<slug>/ or /relojes/<slug>/
    # We use a permissive pattern and rely on canonical filtering downstream
    # RealCash WP custom permalinks: products live at root with a "reloj-" prefix.
    # E.g.: /reloj-automatico-seiko-5-sports-srpd51k1-negro/
    # Requiring "/reloj-" prefix excludes other categories (iPhones, etc.)
    product_link_pattern=r"/reloj(?:es)?-[a-z0-9-]{8,}/?$",
    product_url_prefix="https://realcash.es",
    paginate=True,
    max_pages=20,  # 1043 watches in catalog → 20 pages × 36 = 720 items max
    telegram_chat_env="TELEGRAM_CHAT_ID_REALCASH",
)

if __name__ == "__main__":
    run_scan(
        CFG,
        targets_path=env_str("TARGET_LIST_PATH", "target_list.json"),
        min_net_eur=env_float("MIN_NET_EUR", 30.0),
        min_roi=env_float("MIN_ROI", 0.10),
        max_items=env_int("MAX_ITEMS", 400),
        sleep_s=env_float("FETCH_SLEEP_S", 1.0),
        cooldown_hours=env_int("COOLDOWN_HOURS", 48),
    )
