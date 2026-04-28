#!/usr/bin/env python3
"""TIMELAB — RealCash scanner. Reuses secondhand_core.py."""
from secondhand_core import SiteConfig, run_scan, env_str, env_int, env_float

CFG = SiteConfig(
    shop_id="realcash",
    shop_label="RealCash",
    base_url="https://realcash.es",
    catalog_urls=[
        # Primary: /relojes/ (WooCommerce category, was confirmed in web search)
        "https://realcash.es/relojes/",
        "https://realcash.es/relojes/page/2/",
        "https://realcash.es/relojes/page/3/",
        "https://realcash.es/relojes/page/4/",
        "https://realcash.es/relojes/page/5/",
        # Fallback: AJAX-aware search results (in case /relojes/ doesn't exist)
        "https://realcash.es/?s=Reloj&post_type=product&type_aws=true&aws_id=1&aws_filter=1",
    ],
    # RealCash WP permalinks: products at /reloj-XXXXX/ as RELATIVE links in HTML.
    # Pattern is domain-agnostic and anchored to URL end so /relojes/ (category)
    # and /reloj-XXX/ (product) are distinguished.
    product_link_pattern=r"/reloj(?:es)?-[a-z0-9-]{6,}/?$",
    product_url_prefix="https://realcash.es",
    paginate=False,  # pages 1-5 already explicit in catalog_urls; no auto-paginate
    max_pages=1,
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
