BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
KELLY_FRACTION: float = 0.25        # Fractional Kelly scalar (e.g., Quarter-Kelly)
MAX_POSITION_FRACTION: float = 0.05  # Maximum % of bankroll allocated to a single trade

# Execution & Liquidity Filters
MIN_VOLUME_USD: float = 500.0        # Minimum dollar volume to qualify as tradeable
MIN_DAYS_TO_RESOLUTION: float = 0.5  # Ignore contracts resolving in under 12 hours
ANNUALIZATION_DAYS: float = 365.0
ASSUMED_SPREAD_COST: float = 0.01    # Assumed bid-ask spread cost (in probability points) for synthetic fills