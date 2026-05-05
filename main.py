# main.py

import time

from config import load_settings, validate_settings
from clients.ebay_client import (
    EbayListing,
    enrich_listing_from_detail,
    get_listing_detail,
    get_oauth_token,
    search_listings,
)
from clients.telegram_client import send_crash_message, send_message
from gemini_vision import analyze_listing_photo, format_verdict_for_telegram
from pipeline.adapter_ebay import ebay_listing_to_candidate
from pipeline.cooldown import (
    is_in_cooldown,
    is_repost_due_to_price_drop,
    load_state,
    save_state,
    update_state,
)
from pipeline.engine import build_alert_payload, evaluate_candidate, passes_decision_gate
from pipeline.targets import load_target_bundle
from clients.atelier_client import send_offers_to_atelier


def dedupe_listings(listings: list[EbayListing]) -> list[EbayListing]:
    seen = set()
    output: list[EbayListing] = []

    for listing in listings:
        if not listing.item_id:
            continue
        if listing.item_id in seen:
            continue
        seen.add(listing.item_id)
        output.append(listing)

    return output


def format_alerts(
    target_version: str,
    targets_count: int,
    collected_count: int,
    cooldown_suppressed: int,
    repost_count: int,
    sent_count: int,
    alerts: list[dict],
    raw_candidates: list[dict],
) -> str:
    lines = [
        "🕗 TIMELAB eBay Scan",
        "",
        f"Target list version: {target_version or 'n/a'}",
        f"Targets: {targets_count}",
        f"Collected from eBay: {collected_count}",
        f"Cooldown suppressed: {cooldown_suppressed}",
        f"Reposted by price drop: {repost_count}",
        f"Sent: {sent_count}",
        "",
    ]

    if alerts:
        for idx, alert in enumerate(alerts, start=1):
            lines.extend(
                [
                    f"{idx}) {alert['title']}",
                    f"🏷️ Brand: {alert['brand'] or 'n/a'} | Model: {alert['model'] or 'n/a'} | Ref: {alert['reference'] or 'n/a'}",
                    f"⌚ Type: {alert['watch_type'] or 'n/a'} | Movement: {alert['movement_hint'] or 'n/a'}",
                    f"💶 Buy: {alert['price']:.2f}€ | 🚚 Shipping: {alert['shipping']:.2f}€",
                    f"🎯 Est. close: {alert['est_close']:.2f}€",
                    f"✅ Net est.: {alert['net_profit']:.2f}€ | ROI: {int(alert['roi'] * 100)}%",
                    f"🧠 Score: {alert['match_score']} | Class: {alert['match_band']}",
                    f"📊 sample={alert['sample_size']} | p50={alert['p50']:.2f}€ | p75={alert['p75']:.2f}€ | conf={alert['stats_confidence']}",
                    f"📌 Decision: {alert['decision']} | Reason: {alert['decision_reason']}",
                    f"🔎 KB hit: {'yes' if alert['reference_kb_hit'] else 'no'}",
                    f"📍 {alert['location'] or 'Unknown location'}",
                    f"🧾 Condition: {alert['condition'] or 'n/a'} | Category: {alert['category'] or 'n/a'}",
                    f"🔗 {alert['url']}",
                ]
            )
            # Vision verdict block (empty when Vision was skipped)
            vision_block = format_verdict_for_telegram(alert.get("vision")) if alert.get("vision") else ""
            if vision_block:
                lines.append(vision_block)
            lines.append("")
        return "\n".join(lines).strip()

    lines.append("❌ No opportunities passed the filters.")
    if raw_candidates:
        lines.extend(["", "🧪 Top near-misses:", ""])
        for idx, item in enumerate(raw_candidates[:5], start=1):
            lines.extend(
                [
                    f"{idx}) {item['title']}",
                    f"🏷️ Brand: {item['brand'] or 'n/a'} | Model: {item['model'] or 'n/a'} | Ref: {item['reference'] or 'n/a'}",
                    f"💶 Buy: {item['price']:.2f}€ | 🎯 Est. close: {item['est_close']:.2f}€",
                    f"✅ Net est.: {item['net_profit']:.2f}€ | ROI: {int(item['roi'] * 100)}%",
                    f"🧠 Score: {item['match_score']} | Class: {item['match_band']}",
                    f"📌 Decision: {item['decision']} | Reason: {item['decision_reason']}",
                    f"🔗 {item['url']}",
                    "",
                ]
            )

    return "\n".join(lines).strip()


def main() -> None:
    settings = load_settings()
    validate_settings(settings)

    targets, meta = load_target_bundle(settings.target_list_path)

    token = get_oauth_token(settings)
    state = load_state(settings.state_path)

    raw_listings: list[EbayListing] = []

    for target in targets:
        found = search_listings(
            settings=settings,
            token=token,
            query=target.query,
            category_id=target.ebay_category_id,
            limit=settings.ebay_limit,
        )
        raw_listings.extend(found)
        time.sleep(settings.ebay_throttle_s)

    listings = dedupe_listings(raw_listings)

    if not listings:
        send_message(
            settings,
            "\n".join(
                [
                    "🕗 TIMELAB eBay Scan",
                    "",
                    f"Target list version: {meta.get('version', '') or 'n/a'}",
                    f"Targets: {len(targets)}",
                    "⚠️ 0 listings collected from eBay API.",
                ]
            ),
        )
        return

    listings_to_enrich = listings[: max(0, settings.detail_fetch_n)]
    enriched_by_id: dict[str, EbayListing] = {}

    for listing in listings_to_enrich:
        detail = get_listing_detail(settings, token, listing.item_id)
        enriched = enrich_listing_from_detail(listing, detail)
        enriched_by_id[listing.item_id] = enriched
        time.sleep(settings.ebay_detail_throttle_s)

    final_listings = [enriched_by_id.get(item.item_id, item) for item in listings]

    alerts: list[dict] = []
    raw_candidates: list[dict] = []
    cooldown_suppressed = 0
    repost_count = 0
    sent_count = 0

    for listing in final_listings:
        if listing.category_id and settings.ebay_allowed_category_ids:
            if listing.category_id not in settings.ebay_allowed_category_ids:
                continue

        candidate = ebay_listing_to_candidate(listing)
        result = evaluate_candidate(
            candidate=candidate,
            comparables=[],
            settings=settings,
        )
        if result is None:
            continue

        payload = build_alert_payload(result)
        # Carry through photo_url and item_id for Vision lookup later
        payload["photo_url"] = getattr(listing, "photo_url", "") or ""
        payload["item_id"] = listing.item_id
        raw_candidates.append(payload)

        if not passes_decision_gate(result, settings):
            continue

        if is_in_cooldown(
            item_id=listing.item_id,
            price=listing.price_eur,
            state=state,
            cooldown_hours=settings.cooldown_hours,
            price_drop_repost=settings.price_drop_repost,
        ):
            cooldown_suppressed += 1
            continue

        if is_repost_due_to_price_drop(
            item_id=listing.item_id,
            price=listing.price_eur,
            state=state,
            cooldown_hours=settings.cooldown_hours,
            price_drop_repost=settings.price_drop_repost,
        ):
            repost_count += 1

        alerts.append(payload)
        update_state(listing.item_id, listing.price_eur, state)
        sent_count += 1

    save_state(settings.state_path, state)

    alerts.sort(key=lambda x: (x["net_profit"], x["roi"], x["match_score"]), reverse=True)
    raw_candidates.sort(key=lambda x: (x["net_profit"], x["roi"], x["match_score"]), reverse=True)

    # ─── Vision layer (Gemini Flash-Lite) on top alerts ───
    # Validates the listing's primary photo before sending the alert. Blocks
    # parts-only / replicas / wrong-brand. Failures yield 'skip' (no impact).
    top_alerts = alerts[:10]
    vision_diag = {"analyzed": 0, "ok": 0, "uncertain": 0, "blocked": 0, "skipped": 0}
    for vi_idx, vi_alert in enumerate(top_alerts):
        if vi_idx > 0:
            time.sleep(2.0)  # pace API requests
        vi_verdict = analyze_listing_photo(
            photo_url=vi_alert.get("photo_url", "") or "",
            brand_hint=vi_alert.get("brand", "") or "",
            target_id=vi_alert.get("target_id", "") or "",
            model_hint=vi_alert.get("model", "") or "",
            title=vi_alert.get("title", "") or "",
            source="eBay",
        )
        vi_alert["vision"] = vi_verdict
        vision_diag["analyzed"] += 1
        if vi_verdict.was_skipped:
            vision_diag["skipped"] += 1
        elif vi_verdict.is_blocking:
            vision_diag["blocked"] += 1
        elif vi_verdict.is_flagged:
            vision_diag["uncertain"] += 1
        else:
            vision_diag["ok"] += 1
        print(
            f"[VISION] item={vi_alert.get('item_id', '')} "
            f"verdict={vi_verdict.verdict} conf={vi_verdict.confidence} "
            f"flags={vi_verdict.red_flags} err={vi_verdict.error or '-'}",
            flush=True,
        )

    # Drop blocking verdicts before sending alerts
    top_alerts = [a for a in top_alerts if not a.get("vision") or not a["vision"].is_blocking]
    print(f"[VISION] summary: {vision_diag}", flush=True)

    message = format_alerts(
        target_version=meta.get("version", ""),
        targets_count=len(targets),
        collected_count=len(listings),
        cooldown_suppressed=cooldown_suppressed,
        repost_count=repost_count,
        sent_count=sent_count,
        alerts=top_alerts,
        raw_candidates=raw_candidates[:5],
    )
    send_message(settings, message)
    send_offers_to_atelier(top_alerts, source="eBay")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        settings = load_settings()
        try:
            send_crash_message(settings, f"{type(exc).__name__}: {str(exc)}")
        except Exception:
            pass
        raise
