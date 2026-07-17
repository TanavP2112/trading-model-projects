"""
Polymarket taker fee model.

Formula (Fee Structure V2, effective 2026-03-30):
    fee = shares * fee_rate * p * (1 - p)

This is charged to TAKERS (orders that cross the spread and fill
immediately) on the side they're trading. Makers (resting limit orders)
pay zero and instead earn a rebate funded by taker fees -- not modeled
here, since a signal-driven strategy that needs to act at a specific
time generally can't afford to patiently wait as a maker every time.

Fee peaks at p=0.5 (maximum uncertainty) and shrinks toward the
extremes (p near 0 or 1) -- an important practical detail: momentum
trades that chase a price that's already near 0.95+ pay very little
fee, while reversal/fade trades near p=0.5 pay the most.
"""

from config import FEE_RATE_BY_CATEGORY, DEFAULT_FEE_RATE, ASSUMED_SPREAD_COST


def taker_fee(price: float, shares: float, category: str) -> float:
    """Dollar fee for a taker order of `shares` contracts at `price`, in `category`."""
    rate = FEE_RATE_BY_CATEGORY.get(category, DEFAULT_FEE_RATE)
    return shares * rate * price * (1.0 - price)


def round_trip_cost(entry_price: float, exit_price: float, shares: float,
                     category: str, include_spread: bool = True) -> float:
    """
    Total round-trip trading cost in dollars: taker fee on entry + taker fee
    on exit (if the position is closed before resolution) + an assumed
    spread cost on each side.

    For positions held to resolution (settlement pays out $0 or $1 with no
    market order needed), there is no "exit fee" in the traditional sense --
    only the entry fee and entry spread apply. Use `exit_price=None` to
    signal a hold-to-resolution trade.
    """
    cost = taker_fee(entry_price, shares, category)
    spread_cost = shares * ASSUMED_SPREAD_COST if include_spread else 0.0
    if exit_price is not None:
        cost += taker_fee(exit_price, shares, category)
        spread_cost += shares * ASSUMED_SPREAD_COST if include_spread else 0.0
    return cost + spread_cost


def effective_fee_rate_bps(price: float, category: str) -> float:
    """Fee expressed as basis points of notional (shares*price) -- handy for reporting."""
    rate = FEE_RATE_BY_CATEGORY.get(category, DEFAULT_FEE_RATE)
    if price <= 0:
        return 0.0
    return 10_000 * rate * (1.0 - price)
