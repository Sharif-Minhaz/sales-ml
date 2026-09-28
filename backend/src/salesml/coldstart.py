"""Cold-start revenue estimate for a store with no sales history.

Time-series models need history; a brand-new store has none. The standard fallback is funnel math
(visitors x conversion rate x average order value) run as scenarios instead of a single number, with the
average order value borrowed from a comparable business - here, the Olist data we trained on.
"""

SCENARIOS = {"conservative": 0.5, "base": 1.0, "optimistic": 1.5}  # multipliers on the conversion rate


def estimate(
    monthly_visitors: float,
    conversion_rate_pct: float,
    average_order_value: float,
    monthly_traffic_growth_pct: float = 0.0,
    months: int = 12,
) -> dict:
    if monthly_visitors <= 0 or conversion_rate_pct <= 0 or average_order_value <= 0:
        raise ValueError("visitors, conversion rate and average order value must be positive")
    if not 1 <= months <= 36:
        raise ValueError("months must be between 1 and 36")

    growth = 1 + monthly_traffic_growth_pct / 100
    rows = []
    for m in range(months):
        visitors = monthly_visitors * growth**m
        row = {"month": m + 1, "visitors": round(visitors)}
        for name, multiplier in SCENARIOS.items():
            orders = visitors * conversion_rate_pct / 100 * multiplier
            row[name] = round(orders * average_order_value, 2)
        rows.append(row)

    totals = {name: round(sum(r[name] for r in rows), 2) for name in SCENARIOS}
    return {"months": rows, "totals": totals, "scenario_multipliers": SCENARIOS}
