"""
order_flow_signal.py -- Order-flow imbalance (OFI), a structurally different
signal from the momentum/reversal pair in signals.py.

MOTIVATION: Kyle (1985) and Glosten-Milgrom-style informed-trader models
predict that persistent one-directional trading PRESSURE reveals private
information before it is fully incorporated into price. This is a
volume-and-direction signal, not a price signal -- it asks "who is paying to
cross the spread, and in which direction", rather than "how has the price
moved". This is a genuinely different information channel from mom_vn/rev_vn,
which are both derived purely from the price series.

SIGNAL DEFINITION: for each market-hour, the volume-weighted imbalance
between YES-side and NO-side taker (aggressor) flow over a rolling window:

    ofi_t = (V_yes - V_no) / (V_yes + V_no)     summed over the lookback window

This is bounded in [-1, 1] by construction (it's a proportion), unlike raw
price differences -- so unlike momentum/reversal, OFI does NOT need a
separate vol-normalized variant. It's already scale-free.

TRADE DIRECTION: continuation, not reversal. This follows directly from the
informed-trader theoretical motivation: persistent buying pressure should
precede a price rise (the market hasn't yet incorporated what the informed
flow already reflects), so we bet WITH the sign of the imbalance -- buy YES
when imbalance is positive, buy NO when negative. This is the opposite
convention from reversal_naive/reversal_vol_normalized in signals.py, which
bet AGAINST price deviations. Do not conflate the two.

DATA REQUIREMENT: needs yes_volume and no_volume columns in the panel
(contract-count volume split by taker_side), added in data_fetcher.py. Older
cached panels built before this change will not have these columns --
rebuild the panel before using this signal.

DESIGN NOTE ON LEAKAGE: same discipline as signals.py -- every rolling
window uses only past bars up to and including t, and backtest.py's existing
_shift_signal_causally applies an additional one-bar shift before any entry
decision, so the signal is guaranteed to be built from information strictly
before the entry bar.
"""

import numpy as np
import pandas as pd
from typing import List

DEFAULT_OFI_LOOKBACKS: List[int] = [6, 24]  # hours; short vs. medium decay of informed flow


def order_flow_imbalance(df: pd.DataFrame, lookback: int) -> pd.Series:
    """
    Rolling volume-weighted order-flow imbalance per market, over `lookback`
    hours: (sum(yes_volume) - sum(no_volume)) / (sum(yes_volume) + sum(no_volume))
    within the trailing window ending at t (inclusive).

    Requires yes_volume, no_volume columns (added in data_fetcher.py). Bars
    with zero total flow in the window (e.g., a market with no trades in the
    entire lookback) get NaN, which downstream entry-threshold logic treats
    as "does not fire" -- consistent with how signals.py handles insufficient
    rolling-window data via min_periods.
    """
    if "yes_volume" not in df.columns or "no_volume" not in df.columns:
        raise ValueError(
            "order_flow_imbalance requires 'yes_volume' and 'no_volume' "
            "columns. These are added by data_fetcher.py's panel builder -- "
            "rebuild the panel if it predates this signal."
        )

    grouped = df.groupby("market_id")
    yes_roll = grouped["yes_volume"].transform(
        lambda s: s.rolling(lookback, min_periods=lookback).sum()
    )
    no_roll = grouped["no_volume"].transform(
        lambda s: s.rolling(lookback, min_periods=lookback).sum()
    )
    total = yes_roll + no_roll
    # NaN (not 0) when there's no flow at all in the window, so a genuinely
    # dead market doesn't spuriously read as "perfectly balanced" (0/0 would
    # otherwise silently become 0.0, which looks like real balance rather
    # than an undefined/no-data case).
    imbalance = (yes_roll - no_roll) / total.replace(0, np.nan)
    return imbalance


def add_ofi_signals(
    df: pd.DataFrame,
    lookbacks: List[int] = DEFAULT_OFI_LOOKBACKS,
) -> pd.DataFrame:
    """
    Adds one OFI column per lookback in `lookbacks`, named ofi_{lookback}h,
    e.g. ofi_6h, ofi_24h. Does NOT touch signals.py or its momentum/reversal
    columns -- this is deliberately a standalone signal family, not a
    combination with the existing ones (per the reviewer's point-4 logic:
    measure each signal family on its own before ever considering combining).
    """
    df = df.sort_values(["market_id", "timestamp"]).reset_index(drop=True).copy()
    for lb in lookbacks:
        df[f"ofi_{lb}h"] = order_flow_imbalance(df, lookback=lb).values
    return df


def ofi_entry_threshold(df: pd.DataFrame, signal_col: str, percentile: float = 0.93) -> float:
    """
    Entry threshold for an OFI column, set as a data-driven percentile of
    |signal| -- same convention backtest.py already uses for mom_naive/
    rev_naive (93rd percentile ~ equivalent firing rate to |z|=1.5 on a
    vol-normalized signal). OFI is already bounded in [-1, 1], so this
    threshold will typically land somewhere well inside that range (e.g.
    ~0.3-0.6 depending on how balanced flow typically is), not near the
    ±1.96/1.5 conventions used for z-score signals.
    """
    abs_signal = df[signal_col].abs().dropna()
    if len(abs_signal) < 100:
        raise ValueError(
            f"Only {len(abs_signal)} non-NaN {signal_col} values -- too few "
            f"to set a stable percentile threshold. Check that yes_volume/"
            f"no_volume are populated and the lookback window isn't larger "
            f"than most markets' lifetimes."
        )
    return float(abs_signal.quantile(percentile))


if __name__ == "__main__":
    # Smoke test / example usage against the cached panel.
    import warnings
    warnings.filterwarnings("ignore")

    df = pd.read_parquet("data/kalshi_hf_panel.parquet")
    missing = [c for c in ("yes_volume", "no_volume") if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Panel is missing {missing}. Rebuild it via data_fetcher.py "
            f"(this adds yes_volume/no_volume alongside the existing columns)."
        )

    df = add_ofi_signals(df, lookbacks=DEFAULT_OFI_LOOKBACKS)

    print("OFI signal diagnostics:")
    for lb in DEFAULT_OFI_LOOKBACKS:
        col = f"ofi_{lb}h"
        s = df[col].dropna()
        th = ofi_entry_threshold(df, col)
        print(f"  {col}: n_valid={len(s):,}, mean={s.mean():+.4f}, "
              f"std={s.std():.4f}, |·| p93 threshold={th:.4f}, "
              f"n_would_fire={int((df[col].abs() >= th).sum()):,}")