import pandas as pd

from dataclasses import dataclass
from typing import List, Optional
from config import (KELLY_FRACTION, MAX_POSITION_FRACTION, MIN_VOLUME_USD,
                     MIN_DAYS_TO_RESOLUTION, ANNUALIZATION_DAYS, BASE_URL, ASSUMED_SPREAD_COST)
from fees import calculate_kalshi_taker_fee

# constants
N_DECILES: int = 10
ADAPTIVE_PERCENTILE: float = 97.0
BANKROLL = 100_000.0 # nominal starting bankroll for backtests

@dataclass
class Trade:
    market_id: int
    category: str
    entry_ts: pd.Timestamp
    strategy: str          # "momentum" or "reversal"
    direction: int          # +1 = bet YES, -1 = bet NO
    signal_value: float
    entry_price_for_bet: float   # price paid per share of the side we bought
    outcome: int
    win: int
    position_fraction: float     # fraction of bankroll risked (0 if skipped)
    pnl_dollars: float