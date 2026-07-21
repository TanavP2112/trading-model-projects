"""
Central configuration for the Polymarket momentum/reversal research project.

All endpoints below are public / read-only and require NO API key or wallet
signature (verified against current Polymarket docs, 2026). You only need
wallet-based EIP-712 signing once you move from research -> live order
placement on the CLOB.
"""

# ---------------------------------------------------------------------------
# Polymarket public REST endpoints
# ---------------------------------------------------------------------------
GAMMA_BASE = "https://gamma-api.polymarket.com"      # market discovery/metadata (public)
CLOB_BASE = "https://clob.polymarket.com"            # prices, orderbook, price-history (public reads)
DATA_API_BASE = "https://data-api.polymarket.com"    # trades, positions, open interest (fully public)

GAMMA_MARKETS_ENDPOINT = f"{GAMMA_BASE}/markets"
CLOB_PRICE_HISTORY_ENDPOINT = f"{CLOB_BASE}/prices-history"

# ---------------------------------------------------------------------------
# Polymarket fee schedule (Fee Structure V2, effective 2026-03-30)
# fee = size_in_shares * fee_rate * p * (1 - p)   [charged to TAKERS only]
# Makers (resting limit orders) pay 0 and receive a rebate -- not modeled
# here since this project assumes we cross the spread (taker fills), which
# is the conservative/realistic assumption for a signal-driven strategy
# that needs to enter/exit at specific times rather than patiently quote.
# ---------------------------------------------------------------------------
FEE_RATE_BY_CATEGORY = {
    "crypto": 0.07,
    "sports": 0.05,           # updated Jul 2026 (was 0.03 before)
    "finance": 0.04,
    "politics": 0.04,
    "mentions": 0.04,
    "tech": 0.04,
    "economics": 0.05,
    "culture": 0.05,
    "weather": 0.05,
    "geopolitics": 0.0,      # fee-free category
    "other": 0.05,
}
DEFAULT_FEE_RATE = 0.05

# Assumed effective bid-ask spread cost (in probability points) charged on
# both entry and exit as a TAKER, since our synthetic/demo data has no
# level-2 book. When you plug in real CLOB data, replace this with the
# actual observed bid/ask at each timestamp.
ASSUMED_SPREAD_COST = 0.01   # 1 cent of "price" per side, i.e. round-trip ~2c

# ---------------------------------------------------------------------------
# Strategy / backtest defaults
# ---------------------------------------------------------------------------
MIN_VOLUME_USD = 50_000          # ignore illiquid markets (unreliable fills)
MIN_DAYS_TO_RESOLUTION = 2.0     # don't trade signals firing this close to resolution
MAX_POSITION_FRACTION = 0.05     # hard cap: no single trade > 5% of bankroll
KELLY_FRACTION = 0.25            # trade at 1/4 Kelly (standard risk-of-ruin haircut)
ANNUALIZATION_DAYS = 252         # Sharpe annualization convention (see README for caveat)

RANDOM_SEED = 42
