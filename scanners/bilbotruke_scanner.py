#!/usr/bin/env python3
"""TIMELAB — Bilbotruke scanner. Reuses secondhand_core.py."""
from secondhand_core import SiteConfig, run_scan, env_str, env_int, env_float

CFG = SiteConfig(
    shop_id="bilbotruke",
    shop_label="Bilbotruke",
    base_url="https://bilbotruke.net",
    catalog_urls=[
        "https://bilbotruke.net/183-relojes",
        "https://bilbotruke.net/126-relojes-de-lujo",
        "https://bilbotruke.net/324-otros-relojes-de-alta-gama",
        "https://bilbotruke.net/229-reloj-caballero",
        "https://bilbotruke.net/383-relojes-de-coleccion",
    ],
    # Bilbotruke product URLs follow PrestaShop pattern: /<id>-<slug>.html
    # Example: /36351-reloj-lanscotte-ceramic.html
    product_link_pattern=r"/\d{3,6}-[a-z0-9-]+\.html",
    product_url_prefix="https://bilbotruke.net",
    paginate=True,
    max_pages=8,
    telegram_chat_env="TELEGRAM_CHAT_ID_BILBOTRUKE",
)

if __name__ == "__main__":
    run_scan(
        CFG,
        targets_path=env_str("TARGET_LIST_PATH", "target_list.json"),
        min_net_eur=env_float("MIN_NET_EUR", 30.0),
        min_roi=env_float("MIN_ROI", 0.10),
        max_items=env_int("MAX_ITEMS", 250),
        sleep_s=env_float("FETCH_SLEEP_S", 1.0),
        cooldown_hours=env_int("COOLDOWN_HOURS", 48),
    )
