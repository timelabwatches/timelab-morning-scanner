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
from bridge.identity import IdentityEngine
from pipeline.adapter_ebay import ebay_listing_to_candidate
from pipeline.cooldown import (
    is_in_cooldown,
    is_repost_due_to_price_drop,
    load_state,
    save_state,
    update_state,
)
from pipeline.engine import build_alert_payload, evaluate_candidate, passes_decision_gate
from pipeline.comparables import load_comparables
from pipeline.targets import load_target_bundle


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
                    f"💶 Buy: {alert['price']:.2f}€ | 🚚 Shipping: {alert['shipping']:.2f}€",
                    f"🎯 Est. close: {alert['est_close']:.2f}€",
                    f"✅ Net est.: {alert['net_profit']:.2f}€ | ROI: {int(alert['roi'] * 100)}%",
                    f"🧩 Target: {alert['target_id']} | Match: {alert['match_score']} ({alert['match_band']})",
                    f"📊 sample={alert['sample_size']} | p50={alert['p50']:.2f}€ | p75={alert['p75']:.2f}€ | conf={alert['stats_confidence']}",
                    f"📍 {alert['location'] or 'Unknown location'}",
                    f"🧾 Condition: {alert['condition'] or 'n/a'} | Category: {alert['category'] or 'n/a'}",
                    f"🔗 {alert['url']}",
                    "",
                ]
            )
        return "\n".join(lines).strip()

    lines.append("❌ No opportunities passed the filters.")
    if raw_candidates:
        lines.extend(["", "🧪 Top near-misses:", ""])
        for idx, item in enumerate(raw_candidates[:5], start=1):
            lines.extend(
                [
                    f"{idx}) {item['title']}",
                    f"💶 Buy: {item['price']:.2f}€ | 🎯 Est. close: {item['est_close']:.2f}€",
                    f"✅ Net est.: {item['net_profit']:.2f}€ | ROI: {int(item['roi'] * 100)}%",
                    f"🧩 Target: {item['target_id']} | Match: {item['match_score']} ({item['match_band']})",
                    f"📊 sample={item['sample_size']} | p50={item['p50']:.2f}€ | p75={item['p75']:.2f}€ | conf={item['stats_confidence']}",
                    f"🔗 {item['url']}",
                    "",
                ]
            )

    return "\n".join(lines).strip()


def main() -> None:
    settings = load_settings()
    validate_settings(settings)

    targets, meta = load_target_bundle(settings.target_list_path)
    identity_engine = IdentityEngine(settings.model_master_path)
    comparables = load_comparables(settings.comparables_path)

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
            identity_engine=identity_engine,
            comparables=comparables,
            settings=settings,
        )
        if result is None:
            continue

        payload = build_alert_payload(result)
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

    message = format_alerts(
        target_version=meta.get("version", ""),
        targets_count=len(targets),
        collected_count=len(listings),
        cooldown_suppressed=cooldown_suppressed,
        repost_count=repost_count,
        sent_count=sent_count,
        alerts=alerts[:10],
        raw_candidates=raw_candidates[:5],
    )
    send_message(settings, message)


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