#!/usr/bin/env python3
"""
TIMELAB — Vinted scraping diagnostic probe.

Single-purpose: prove (or disprove) that we can hit Vinted's catalog API
from GitHub Actions / data-center IP without a paid proxy.

Strategy mirrors Wallapop probe:
  1. GET https://www.vinted.es/ to obtain session cookies (Cloudflare clears
     cf_clearance, vinted_fr_session, etc.)
  2. GET https://www.vinted.es/api/v2/catalog/items with the cookies
  3. Inspect response: status, body shape, item count
  4. Print enough detail to choose the next step:
       - "viable" : 200 + items list with usable fields
       - "stuck"  : 403 / 401 / captcha / empty
"""
import json
import os
import sys
import time
from typing import Any, Dict

import requests

VINTED_HOME = "https://www.vinted.es/"
VINTED_API  = "https://www.vinted.es/api/v2/catalog/items"

# Vinted's category id for "Watches" in Spain.
# Discovered from the website's URL pattern; verify by visiting
# https://www.vinted.es/catalog?catalog[]=304
WATCHES_CATEGORY = "304"

# Chrome 120 desktop fingerprint
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/120.0.0.0 Safari/537.36")

HOME_HEADERS = {
    "User-Agent": UA,
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT":             "1",
    "Connection":      "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Ch-Ua":          '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    "Sec-Ch-Ua-Mobile":   "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest":     "document",
    "Sec-Fetch-Mode":     "navigate",
    "Sec-Fetch-Site":     "none",
    "Sec-Fetch-User":     "?1",
}

API_HEADERS = {
    "User-Agent":      UA,
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer":         "https://www.vinted.es/catalog?catalog[]=" + WATCHES_CATEGORY,
    "Origin":          "https://www.vinted.es",
    "DNT":             "1",
    "Connection":      "keep-alive",
    "X-Requested-With": "XMLHttpRequest",   # Vinted's JS uses this
    "Sec-Ch-Ua":          '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    "Sec-Ch-Ua-Mobile":   "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest":     "empty",
    "Sec-Fetch-Mode":     "cors",
    "Sec-Fetch-Site":     "same-origin",
}


def log(msg: str) -> None:
    print(f"[VT-PROBE] {msg}", flush=True)


def warmup(sess: requests.Session) -> bool:
    """Visit the homepage, return True if we got cookies."""
    try:
        r = sess.get(VINTED_HOME, headers=HOME_HEADERS, timeout=20)
        cookie_names = sorted({c.name for c in sess.cookies})
        log(f"warmup status={r.status_code} content_length={len(r.content)} "
            f"cookies={len(sess.cookies)} names={cookie_names[:10]}")
        if r.status_code != 200:
            log(f"warmup non_200_body_excerpt={r.text[:300]!r}")
            return False
        # Vinted's CSRF token is embedded in the homepage HTML inside
        # <meta name="csrf-token" content="..."> — extract it for later
        # if we end up needing it. For probing, cookies alone may be enough.
        return True
    except Exception as e:
        log(f"warmup_exception type={type(e).__name__} err={e}")
        return False


def search(sess: requests.Session, keyword: str = "omega") -> Dict[str, Any]:
    """Hit the catalog API; return diagnostic dict."""
    params = {
        "search_text":     keyword,
        "catalog_ids":     WATCHES_CATEGORY,
        "order":           "newest_first",
        "page":            "1",
        "per_page":        "20",
    }
    try:
        r = sess.get(VINTED_API, headers=API_HEADERS, params=params, timeout=20)
    except Exception as e:
        return {"verdict": "exception", "error": f"{type(e).__name__}: {e}"}

    out: Dict[str, Any] = {
        "status": r.status_code,
        "content_length": len(r.content),
        "url": r.url[:200],
        "ctype": r.headers.get("Content-Type", ""),
    }
    if r.status_code != 200:
        out["body_excerpt"] = r.text[:400]
        out["verdict"] = f"http_{r.status_code}"
        return out

    try:
        data = r.json()
    except Exception as e:
        out["verdict"] = "non_json"
        out["body_excerpt"] = r.text[:400]
        out["json_error"] = str(e)
        return out

    if not isinstance(data, dict):
        out["verdict"] = "unexpected_root_type"
        out["root_type"] = type(data).__name__
        return out

    out["top_keys"] = list(data.keys())
    items = data.get("items")
    if not isinstance(items, list):
        out["verdict"] = "no_items_key"
        out["preview"] = json.dumps(data, ensure_ascii=False)[:500]
        return out

    out["item_count"] = len(items)
    if items:
        first = items[0]
        out["first_item_keys"] = list(first.keys()) if isinstance(first, dict) else None
        # Pull what we'd actually need for the scanner
        useful = {
            "id":          first.get("id"),
            "title":       first.get("title"),
            "price":       (first.get("price") or {}).get("amount") if isinstance(first.get("price"), dict) else first.get("price"),
            "currency":    (first.get("price") or {}).get("currency_code") if isinstance(first.get("price"), dict) else None,
            "url":         first.get("url"),
            "brand_title": first.get("brand_title"),
            "size_title":  first.get("size_title"),
            "status":      first.get("status"),
            "user_id":     (first.get("user") or {}).get("id") if isinstance(first.get("user"), dict) else None,
        }
        out["first_item_useful"] = useful

    out["verdict"] = "viable" if items else "empty_results"
    return out


def main() -> int:
    sess = requests.Session()
    log("=" * 60)
    log("STAGE 1: HOMEPAGE WARMUP")
    log("=" * 60)
    warmup(sess)

    # Polite pause before hitting the API
    time.sleep(2.0)

    log("")
    log("=" * 60)
    log("STAGE 2: CATALOG API SEARCH (keyword='omega')")
    log("=" * 60)
    result = search(sess, keyword="omega")
    for k, v in result.items():
        if isinstance(v, (dict, list)):
            v_repr = json.dumps(v, ensure_ascii=False)[:300]
        else:
            v_repr = str(v)[:300]
        log(f"  {k}: {v_repr}")

    log("")
    log("=" * 60)
    verdict = result.get("verdict", "?")
    if verdict == "viable":
        log(f"VERDICT: VIABLE  ({result.get('item_count', 0)} items returned)")
        log("Next step: build full scanner using the field schema above.")
        return 0
    elif verdict == "empty_results":
        log("VERDICT: API REACHED but returned 0 items for 'omega'.")
        log("Try a more common keyword or remove category filter.")
        return 1
    else:
        log(f"VERDICT: BLOCKED — {verdict}")
        log("Anti-bot intercepted. Need browser-real or proxy approach.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
