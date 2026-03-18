def estimate_close_price(price_eur: float, target_p50: float, target_p75: float, listing_text: str) -> float:
    text = (listing_text or "").lower()

    base = target_p50 if target_p50 > 0 else max(price_eur * 1.35, price_eur + 80)

    strong_terms = {
        "new with box and papers",
        "nuevo con caja y documentación",
        "neu mit karton und unterlagen",
        "full set",
        "box and papers",
        "nuevo",
        "mint",
        "nos",
        "new old stock",
    }

    if target_p75 > target_p50 and any(term in text for term in strong_terms):
        base = target_p75

    return round(base, 2)


def estimate_net_profit(
    buy_price: float,
    shipping_cost: float,
    estimated_close: float,
    catwiki_commission: float,
    payment_processing: float,
    packaging_eur: float,
    misc_eur: float,
    ship_arbitrage_eur: float,
    effective_tax_rate_on_profit: float,
) -> tuple[float, float]:
    total_cost = buy_price + shipping_cost + packaging_eur + misc_eur
    total_revenue = estimated_close + ship_arbitrage_eur
    fees = estimated_close * catwiki_commission + payment_processing

    profit_before_tax = total_revenue - fees - total_cost
    tax = max(0.0, profit_before_tax) * effective_tax_rate_on_profit
    net_profit = profit_before_tax - tax

    roi = net_profit / max(1.0, total_cost)
    return round(net_profit, 2), round(roi, 4)


def passes_basic_profit_filters(
    net_profit: float,
    roi: float,
    min_net_eur: float,
    min_net_roi: float,
) -> bool:
    return net_profit >= min_net_eur and roi >= min_net_roi