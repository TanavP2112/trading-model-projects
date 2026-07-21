"""
Polymarket taker fee model.

Formula (Fee Structure V2, effective 2026-03-30):
    fee = shares * fee_rate * (p * (1 - p)) ** exponent

This is charged to TAKERS (orders that cross the spread and fill
immediately) on the side they're trading. Makers (resting limit orders)
pay zero and instead earn a rebate funded by taker fees -- not modeled
here, since a signal-driven strategy that needs to act at a specific
time generally can't afford to patiently wait as a maker every time.

Fee peaks at p=0.5 (maximum uncertainty) and shrinks toward the
extremes (p near 0 or 1) -- an important practical detail: momentum
trades that chase a price that's already near 0.95+ pay very little
fee, while reversal/fade trades near p=0.5 pay the most. The exponent
controls HOW FAST it shrinks: exponent=1 (most categories) is the
"standard" curve; exponent=0.5 (Economics, Weather) is flatter, so fees
stay relatively higher even at extreme prices; exponent=2 (Other,
Mentions) is steeper, so fees drop off much faster away from 50c.

CONFIRMED against Polymarket's own docs (docs.polymarket.com/trading/fees):
their fee-parameter object is {r: feeRate, e: exponent, to: takerOnly} -- so
a per-category exponent field DOES genuinely exist. What's NOT confirmed:
the specific exponent VALUE per category. One third-party breakdown claims
Economics/Weather use 0.5 and Other/Mentions use 2.0 -- but combining those
exponents with this file's existing rate table produces peak fees that
CONFLICT with the "$X per 100 shares" figures given by multiple OTHER
sources (e.g. exponent=0.5 + rate=0.05 implies a $2.50 peak on 100 shares,
while several sources independently document economics' peak as $1.25,
which is only consistent with exponent=1). Rather than ship a specific
exponent value I can't reconcile across sources, EXPONENT_BY_CATEGORY below
defaults everything to 1.0 (matching what the more numerous, mutually-
consistent peak-dollar-figure sources imply) and is left as an explicit,
easy override point if you can confirm the real per-category values
directly against Polymarket's docs or live market data.

NOT CURRENTLY LIVE-FETCHABLE: the docs describe pulling real per-market fee
params via a getClobMarketInfo()-style call, but the installed SDK version
at the time this was written (polymarket-client==0.1.0b20) has no such
method, and the Market model has no fee-related fields either -- confirmed
by direct introspection, not assumed. This file's FEE_RATE_BY_CATEGORY /
EXPONENT_BY_CATEGORY tables are therefore a maintained static approximation,
not a live source of truth, and (as already happened twice -- crypto
0.072->0.07, sports 0.03->0.05 in July 2026) WILL drift out of date. If a
future SDK version exposes real per-market fee params, prefer that over
this table.
"""

from config import FEE_RATE_BY_CATEGORY, DEFAULT_FEE_RATE, ASSUMED_SPREAD_COST

# Per-category fee-curve exponent. The exponent field itself is confirmed
# real (see module docstring); the specific per-category values below are
# NOT independently confirmed and default to 1.0 everywhere pending that
# verification -- override here once you've checked the real numbers.
EXPONENT_BY_CATEGORY = {
    # "economics": 0.5,   # UNVERIFIED -- see module docstring before uncommenting
    # "weather": 0.5,     # UNVERIFIED
    # "other": 2.0,       # UNVERIFIED
    # "mentions": 2.0,    # UNVERIFIED
}
DEFAULT_EXPONENT = 1.0


def taker_fee(price: float, shares: float, category: str) -> float:
    """Dollar fee for a taker order of `shares` contracts at `price`, in `category`."""
    rate = FEE_RATE_BY_CATEGORY.get(category, DEFAULT_FEE_RATE)
    exponent = EXPONENT_BY_CATEGORY.get(category, DEFAULT_EXPONENT)
    return shares * rate * (price * (1.0 - price)) ** exponent


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
    exponent = EXPONENT_BY_CATEGORY.get(category, DEFAULT_EXPONENT)
    if price <= 0:
        return 0.0
    return 10_000 * rate * ((price * (1.0 - price)) ** exponent) / price
