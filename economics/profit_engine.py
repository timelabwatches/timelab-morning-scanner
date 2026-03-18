# economics/profit_engine.py

FEE_RATE = 0.15
DEFAULT_SHIPPING_ARBITRAGE = 35.0


def safe_float(value) -> float | None:
    """
    Convert value to float if possible.
    """

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def calculate_net_profit(
    buy_price: float | None,
    hammer_price: float | None,
    fee_rate: float = FEE_RATE,
    shipping_arbitrage: float = DEFAULT_SHIPPING_ARBITRAGE,
) -> float | None:
    """
    Net profit formula used by TIMELAB.

    net_profit = hammer_price * (1 - fee_rate) + shipping_arbitrage - buy_price
    """

    buy_price = safe_float(buy_price)
    hammer_price = safe_float(hammer_price)

    if buy_price is None or hammer_price is None:
        return None

    return hammer_price * (1 - fee_rate) + shipping_arbitrage - buy_price


def calculate_roi(
    buy_price: float | None,
    net_profit: float | None,
) -> float | None:
    """
    ROI = net_profit / buy_price
    """

    buy_price = safe_float(buy_price)
    net_profit = safe_float(net_profit)

    if buy_price is None or net_profit is None or buy_price <= 0:
        return None

    return net_profit / buy_price


def calculate_breakeven_hammer(
    buy_price: float | None,
    fee_rate: float = FEE_RATE,
    shipping_arbitrage: float = DEFAULT_SHIPPING_ARBITRAGE,
) -> float | None:
    """
    Hammer price needed to break even.

    0 = hammer * (1 - fee_rate) + shipping_arbitrage - buy_price
    """

    buy_price = safe_float(buy_price)

    if buy_price is None:
        return None

    denominator = 1 - fee_rate
    if denominator <= 0:
        return None

    return (buy_price - shipping_arbitrage) / denominator


def build_profit_scenario(
    buy_price: float | None,
    hammer_price: float | None,
    fee_rate: float = FEE_RATE,
    shipping_arbitrage: float = DEFAULT_SHIPPING_ARBITRAGE,
) -> dict:
    """
    Build one economic scenario.
    """

    net_profit = calculate_net_profit(
        buy_price=buy_price,
        hammer_price=hammer_price,
        fee_rate=fee_rate,
        shipping_arbitrage=shipping_arbitrage,
    )

    roi = calculate_roi(
        buy_price=buy_price,
        net_profit=net_profit,
    )

    return {
        "hammer_price": safe_float(hammer_price),
        "net_profit": net_profit,
        "roi": roi,
    }


def apply_profit_engine(record: dict) -> dict:
    """
    Add economics block from auction estimate + buy price.
    """

    buy_price = safe_float(record.get("price"))
    auction_estimate = record.get("auction_estimate") or {}

    expected_hammer = safe_float(auction_estimate.get("expected_hammer"))
    conservative_hammer = safe_float(auction_estimate.get("conservative_hammer"))
    optimistic_hammer = safe_float(auction_estimate.get("optimistic_hammer"))

    economics = {
        "economics_available": False,
        "fee_rate": FEE_RATE,
        "shipping_arbitrage": DEFAULT_SHIPPING_ARBITRAGE,
        "buy_price": buy_price,
        "breakeven_hammer": calculate_breakeven_hammer(
            buy_price=buy_price,
            fee_rate=FEE_RATE,
            shipping_arbitrage=DEFAULT_SHIPPING_ARBITRAGE,
        ),
        "expected_case": build_profit_scenario(
            buy_price=buy_price,
            hammer_price=expected_hammer,
            fee_rate=FEE_RATE,
            shipping_arbitrage=DEFAULT_SHIPPING_ARBITRAGE,
        ),
        "conservative_case": build_profit_scenario(
            buy_price=buy_price,
            hammer_price=conservative_hammer,
            fee_rate=FEE_RATE,
            shipping_arbitrage=DEFAULT_SHIPPING_ARBITRAGE,
        ),
        "optimistic_case": build_profit_scenario(
            buy_price=buy_price,
            hammer_price=optimistic_hammer,
            fee_rate=FEE_RATE,
            shipping_arbitrage=DEFAULT_SHIPPING_ARBITRAGE,
        ),
    }

    if expected_hammer is not None and buy_price is not None:
        economics["economics_available"] = True

    record["economics"] = economics

    return record