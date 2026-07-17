"""
Backtest engine for the momentum/reversal prediction-market strategy.

Pipeline:
  1. Generate ONE candidate trade per (market, strategy) at the first bar
     where the signal crosses a minimum candidate threshold, subject to
     liquidity + time-to-resolution filters.
  2. Split markets chronologically into TRAIN / TEST (no shuffling --
     this is a time series; shuffling would leak future information
     into calibration).
  3. On TRAIN only: bucket candidate trades into signal deciles and
     empirically measure win rate + average entry price per bucket.
     Convert each bucket's edge into a capped, fractional-Kelly position
     size. This calibration is FROZEN before touching TEST.
  4. On TEST: apply the frozen calibration (bucket boundaries + sizing)
     to size and simulate each trade, including Polymarket's actual fee
     formula and an assumed spread cost. Trades with a non-positive
     calibrated edge are sized at 0 (skipped).
  5. Report BOTH a daily-aggregated Sharpe and a trade-level Sharpe --
     see the warning in `compute_metrics` for why these can diverge a
     lot for a strategy that doesn't trade every single day.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fees import taker_fee
from config import (MIN_VOLUME_USD, MIN_DAYS_TO_RESOLUTION, MAX_POSITION_FRACTION,
                     KELLY_FRACTION, ANNUALIZATION_DAYS, ASSUMED_SPREAD_COST)

N_DECILES = 10
CANDIDATE_MIN_MOM = 0.30     # minimum |logit move| to even consider a momentum trade
CANDIDATE_MIN_REV = 1.00     # minimum |z-score| to even consider a reversal trade
CANDIDATE_MIN_STRUCT_MOM = 1.50   # structural signals are proper z-scores, so thresholds
CANDIDATE_MIN_STRUCT_REV = 0.75   # here are ordinary z-critical-values -- BUT momentum and
# reversal z-scores have genuinely different typical scales (momentum measures a cumulative
# move over the whole window, reversal measures a point deviation from the local mean, which
# is inherently smaller), so the same absolute threshold is NOT equally selective for both.
# These two values were set by checking each signal's own empirical quantiles on the demo
# panel (see README) to land at roughly the same selectivity (~top 3%) for both -- if you
# change lookback windows or move to real data, re-check panel['struct_*_signal'].abs()
# .quantile([0.9,0.95,0.99]) rather than assuming these fixed numbers still make sense.
BANKROLL = 100_000.0         # nominal bankroll for dollar-denominated reporting


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


def _first_candidate_per_market(df: pd.DataFrame, signal_col: str, min_abs: float) -> pd.DataFrame:
    """One row per market: the first bar where |signal| >= min_abs and filters pass."""
    ok = df["tradeable"] & (df[signal_col].abs() >= min_abs)
    cand = df[ok].sort_values(["market_id", "timestamp"])
    return cand.groupby("market_id", as_index=False).first()


def build_candidates(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    df = df.copy()
    df["tradeable"] = (df["volume"] >= MIN_VOLUME_USD) & (df["days_to_resolution"] >= MIN_DAYS_TO_RESOLUTION)

    if strategy == "momentum":
        cand = _first_candidate_per_market(df, "mom_signal", CANDIDATE_MIN_MOM)
        cand["signal_value"] = cand["mom_signal"]
        cand["direction"] = np.sign(cand["signal_value"]).astype(int)     # bet WITH the move
    elif strategy == "reversal":
        cand = _first_candidate_per_market(df, "rev_signal", CANDIDATE_MIN_REV)
        cand["signal_value"] = cand["rev_signal"]
        cand["direction"] = -np.sign(cand["signal_value"]).astype(int)    # FADE the extension
    elif strategy == "structural_momentum":
        cand = _first_candidate_per_market(df, "struct_mom_signal", CANDIDATE_MIN_STRUCT_MOM)
        cand["signal_value"] = cand["struct_mom_signal"]
        cand["direction"] = np.sign(cand["signal_value"]).astype(int)     # bet WITH the move
    elif strategy == "structural_reversal":
        cand = _first_candidate_per_market(df, "struct_rev_signal", CANDIDATE_MIN_STRUCT_REV)
        cand["signal_value"] = cand["struct_rev_signal"]
        cand["direction"] = -np.sign(cand["signal_value"]).astype(int)    # FADE the extension
    else:
        raise ValueError(strategy)

    cand = cand[cand["direction"] != 0].copy()
    # price paid per share of the side we actually bought (YES side price if direction=+1, else NO side price)
    cand["entry_price_for_bet"] = np.where(cand["direction"] == 1, cand["price"], 1 - cand["price"])
    cand["win"] = np.where(cand["direction"] == 1, cand["outcome"], 1 - cand["outcome"]).astype(int)
    cand["raw_edge"] = cand["win"] - cand["entry_price_for_bet"]
    return cand


def train_test_split_by_market(df: pd.DataFrame, train_frac: float = 0.65):
    market_start = df.groupby("market_id")["timestamp"].min().sort_values()
    n_train = int(len(market_start) * train_frac)
    train_ids = set(market_start.index[:n_train])
    test_ids = set(market_start.index[n_train:])
    return df[df["market_id"].isin(train_ids)].copy(), df[df["market_id"].isin(test_ids)].copy()


def calibrate_on_train(train_cand: pd.DataFrame):
    """
    Bucket TRAIN candidate trades into deciles of signed signal_value,
    return (bin_edges, calibration_table) where calibration_table maps
    bucket index -> capped fractional-Kelly position size.
    """
    if len(train_cand) < N_DECILES * 3:
        # too few trades to calibrate deciles reliably -- fall back to one bucket
        bins = 1
    else:
        bins = N_DECILES

    train_cand = train_cand.copy()
    try:
        train_cand["bucket"], bin_edges = pd.qcut(train_cand["signal_value"], bins,
                                                    labels=False, retbins=True, duplicates="drop")
    except ValueError:
        train_cand["bucket"] = 0
        bin_edges = np.array([-np.inf, np.inf])

    calib = {}
    for b, g in train_cand.groupby("bucket"):
        q = g["win"].mean()
        c = g["entry_price_for_bet"].mean()
        n = len(g)
        c = np.clip(c, 0.02, 0.98)
        kelly_full = q - (1 - q) * c / (1 - c)
        kelly_sized = max(kelly_full, 0.0) * KELLY_FRACTION
        kelly_sized = min(kelly_sized, MAX_POSITION_FRACTION)

        # Statistical-significance guard: don't just require a minimum
        # sample size -- require the estimated edge to be several
        # standard errors away from breakeven. A decile that "looks"
        # profitable with n=25 trades and no real underlying edge is
        # exactly the kind of noise that blows up out-of-sample (this is
        # the single most common way new quants fool themselves with a
        # decile-bucketed backtest). Edge here is measured as win_rate
        # minus the breakeven win rate implied by the price paid (c);
        # a bet is only profitable in expectation if q > c.
        #
        # z > 2.0 is deliberately stricter than a naive "95% confidence"
        # (z > 1.645) threshold: we are testing N_DECILES=10 buckets
        # SIMULTANEOUSLY, so under the null of "no real edge anywhere"
        # we'd still expect ~1 bucket in 10 to clear a lenient z > 1.28
        # threshold by chance alone (multiple-comparisons problem). A
        # properly Bonferroni-corrected threshold for 10 simultaneous
        # tests at 95% overall confidence would be z > ~2.5-2.8; z > 2.0
        # is a pragmatic middle ground for a demo, not a rigorous
        # correction -- tighten this further before trusting it with
        # real capital.
        se = np.sqrt(max(q * (1 - q), 1e-6) / max(n, 1))
        z_stat = (q - c) / se if se > 0 else 0.0
        if n < 15 or z_stat < 2.0:
            kelly_sized = 0.0

        calib[int(b)] = {"win_rate": q, "avg_price": c, "n_train": n,
                          "z_stat": z_stat, "position_fraction": kelly_sized}
    return bin_edges, calib


def apply_calibration_to_test(test_cand: pd.DataFrame, bin_edges: np.ndarray, calib: dict,
                               strategy: str, category_fee_map) -> list[Trade]:
    trades = []
    if len(bin_edges) < 2:
        return trades
    buckets = pd.cut(test_cand["signal_value"], bins=bin_edges, labels=False, include_lowest=True)
    for (_, row), bucket in zip(test_cand.iterrows(), buckets):
        info = calib.get(int(bucket), None) if pd.notna(bucket) else None
        frac = info["position_fraction"] if info else 0.0
        if frac <= 0:
            continue
        stake = frac * BANKROLL
        entry_p = row["entry_price_for_bet"]
        shares = stake / entry_p
        fee = taker_fee(entry_p, shares, row["category"])
        spread = shares * ASSUMED_SPREAD_COST
        gross_pnl = shares * (row["win"] - entry_p)
        net_pnl = gross_pnl - fee - spread
        trades.append(Trade(
            market_id=row["market_id"], category=row["category"], entry_ts=row["timestamp"],
            strategy=strategy, direction=int(row["direction"]), signal_value=row["signal_value"],
            entry_price_for_bet=entry_p, outcome=int(row["outcome"]), win=int(row["win"]),
            position_fraction=frac, pnl_dollars=net_pnl,
        ))
    return trades


def run_strategy(df: pd.DataFrame, strategy: str, train_frac: float = 0.65) -> list[Trade]:
    cand = build_candidates(df, strategy)
    train_df, test_df = train_test_split_by_market(df, train_frac)
    train_cand = cand[cand["market_id"].isin(set(train_df["market_id"]))]
    test_cand = cand[cand["market_id"].isin(set(test_df["market_id"]))]
    bin_edges, calib = calibrate_on_train(train_cand)
    trades = apply_calibration_to_test(test_cand, bin_edges, calib, strategy, None)
    return trades, calib


def trades_to_frame(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([t.__dict__ for t in trades])


def compute_metrics(trades_df: pd.DataFrame, bankroll: float = BANKROLL) -> dict:
    if trades_df.empty:
        return {"n_trades": 0}

    trades_df = trades_df.sort_values("entry_ts")
    trade_returns = trades_df["pnl_dollars"] / bankroll

    # --- Trade-level Sharpe (annualized by observed trade frequency) ---
    span_days = max((trades_df["entry_ts"].max() - trades_df["entry_ts"].min()).days, 1)
    trades_per_year = len(trades_df) / (span_days / 365.0)
    trade_sharpe = (trade_returns.mean() / trade_returns.std(ddof=1)) * np.sqrt(trades_per_year) \
        if trade_returns.std(ddof=1) > 0 else np.nan

    # --- Daily-aggregated Sharpe ---
    daily = trades_df.groupby(trades_df["entry_ts"].dt.floor("D"))["pnl_dollars"].sum() / bankroll
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx, fill_value=0.0)
    daily_sharpe = (daily.mean() / daily.std(ddof=1)) * np.sqrt(ANNUALIZATION_DAYS) \
        if daily.std(ddof=1) > 0 else np.nan

    equity = (1 + daily).cumprod()
    running_max = equity.cummax()
    max_dd = ((equity - running_max) / running_max).min()

    return {
        "n_trades": len(trades_df),
        "win_rate": trades_df["win"].mean(),
        "avg_pnl_per_trade": trades_df["pnl_dollars"].mean(),
        "total_pnl": trades_df["pnl_dollars"].sum(),
        "total_return_pct": 100 * trades_df["pnl_dollars"].sum() / bankroll,
        "trade_level_sharpe": trade_sharpe,
        "daily_aggregated_sharpe": daily_sharpe,
        "max_drawdown_pct": 100 * max_dd,
        "avg_position_fraction": trades_df["position_fraction"].mean(),
        "equity_curve": equity,
    }
