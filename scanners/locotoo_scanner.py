#!/usr/bin/env python3
"""TIMELAB — Locotoo scanner. Reuses secondhand_core.py."""
from secondhand_core import SiteConfig, run_scan, env_str, env_int, env_float

CFG = SiteConfig(
    shop_id="locotoo",
    shop_label="Locotoo",
    base_url="https://locotoo.com",
    catalog_urls=[
        # Locotoo: products live at /shop/product/<category-slug>/<sku>
        # Need a real category URL — try common slugs:
        "https://locotoo.com/shop/category/relojes",
        "https://locotoo.com/shop/category/relojes-de-alta-gama",
        "https://locotoo.com/shop/category/joyeria-y-relojes",
    ],
    # Locotoo product URLs: /shop/product/<category-slug>/<sku>
    # E.g. /shop/product/reloj-pulsera-2434/e0498273hg-1919074
    product_link_pattern=r"/shop/product/[^/?#]+/[^/?#]+",
    product_url_prefix="https://locotoo.com",
    paginate=True,
    max_pages=10,
    telegram_chat_env="TELEGRAM_CHAT_ID_LOCOTOO",
)

if __name__ == "__main__":
    run_scan(
        CFG,
        targets_path=env_str("TARGET_LIST_PATH", "target_list.json"),
        min_net_eur=env_float("MIN_NET_EUR", 30.0),
        min_roi=env_float("MIN_ROI", 0.10),
        max_items=env_int("MAX_ITEMS", 200),
        sleep_s=env_float("FETCH_SLEEP_S", 1.0),
        cooldown_hours=env_int("COOLDOWN_HOURS", 48),
    )
