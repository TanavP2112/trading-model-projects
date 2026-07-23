from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    MAX_POSITION_FRACTION, MIN_VOLUME_USD,
    MIN_DAYS_TO_RESOLUTION, ANNUALIZATION_DAYS,
)
from fees import calculate_kalshi_taker_fee


BANKROLL = 100_000.0
DEFAULT_HOLDING_HOURS = [1, 6, 12, 24]
DEFAULT_ENTRY_THRESHOLD = 1.5   # |z| for vol-normalized signals to fire
DEFAULT_ENTRY_FRAC = 0.01       # 1% of bankroll per trade


# ---------------------------------------------------------------------------
# Fixed-horizon trade generation
# ---------------------------------------------------------------------------
def _shift_signal_causally(df: pd.DataFrame, signal_col: str) -> pd.Series:
    """
    Explicit one-bar shift before checking for a fire, so the signal is
    guaranteed built from info STRICTLY BEFORE the entry bar. Redundant
    if the signal was already built causally, but cheap and explicit.
    """
    return df.groupby("market_id")[signal_col].shift(1)


def _future_price(df: pd.DataFrame, horizon: int) -> pd.Series:
    return df.groupby("market_id")["price"].shift(-horizon)


def _future_spread(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Spread at the exit bar, same shift convention as _future_price."""
    return df.groupby("market_id")["spread"].shift(-horizon)


def _sign_by_signal(signal_col: str) -> int:
    """
    Momentum: bet WITH the signal (positive z -> buy YES).
    Reversal: bet AGAINST the signal (positive z means price is elevated
              -> short YES = buy NO).
    """
    if "mom" in signal_col:
        return +1
    if "rev" in signal_col:
        return -1
    return +1


def _fees(entry_price: float, exit_price: float, shares: float) -> float:
    """Round-trip Kalshi taker fees at entry and exit."""
    return (calculate_kalshi_taker_fee(entry_price, shares)
            + calculate_kalshi_taker_fee(exit_price, shares))


def generate_trades(
    df: pd.DataFrame,
    signal_col: str,
    horizon_hours: int,
    entry_threshold: float = DEFAULT_ENTRY_THRESHOLD,
    entry_frac: float = DEFAULT_ENTRY_FRAC,
    min_volume: float = MIN_VOLUME_USD,
    min_days_to_resolution: float = MIN_DAYS_TO_RESOLUTION,
    bankroll: float = BANKROLL,
    apply_slippage: bool = True,
    test_mask: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    apply_slippage : bool
        If True (default), charge half the (reconstructed) bid-ask spread
        on both entry and exit legs, in addition to Kalshi's stated taker
        fee. This approximates the cost of crossing the spread as a taker
        rather than transacting at the panel's volume-weighted mid price.

        Caveat: the `spread` column is itself a reconstruction from trade
        aggressor flow (see data_fetcher.py), not a directly observed
        quoted spread -- ~79% of bars fall through to the 0.01 minimum-tick
        floor (see README limitations). This makes the slippage estimate
        a conservative, directionally-correct nudge rather than a precise
        execution-cost model: it will never make a strategy's P&L look
        better, only equal or worse, which is the safe direction to err
        for an out-of-sample backtest.

        If a bar's spread value is missing (e.g., older cached panels
        without the `spread` column, or NaN at exit due to end-of-market
        truncation), slippage falls back to 0 for that trade rather than
        dropping it, so this flag is backwards-compatible with panels that
        predate spread reconstruction.
    """
    df = df.sort_values(["market_id", "timestamp"]).reset_index(drop=True).copy()
    df["_signal"] = _shift_signal_causally(df, signal_col) if signal_col != "unconditional" else 1.0
    df["_exit_price"] = _future_price(df, horizon_hours)
    has_spread_col = "spread" in df.columns
    if apply_slippage and has_spread_col:
        df["_exit_spread"] = _future_spread(df, horizon_hours)

    horizon_days = horizon_hours / 24.0
    fires = (
        df["_signal"].abs() >= entry_threshold
        if signal_col != "unconditional"
        else pd.Series(True, index=df.index)
    )
    eligible = (
        fires
        & df["_exit_price"].notna()
        & (df["volume"] >= min_volume)
        & (df["days_to_resolution"] >= max(min_days_to_resolution, horizon_days))
    )
    if "is_clean_bar" in df.columns:
        eligible = eligible & df["is_clean_bar"].fillna(False).astype(bool)
        if not test_mask.index.equals(df.index):
            test_mask = test_mask.reindex(df.index, fill_value=False)
        eligible = eligible & test_mask
    hits = df[eligible].copy()
    if hits.empty:
        return pd.DataFrame()

    base_sign = _sign_by_signal(signal_col)
    if signal_col == "unconditional":
        direction = pd.Series(+1, index=hits.index)
    else:
        direction = base_sign * np.sign(hits["_signal"]).astype(int)

    entry_p = hits["price"].astype(float)
    exit_p = hits["_exit_price"].astype(float)

    # Cost per share: entry price if buying YES, (1 - entry) if buying NO.
    cost_per_share_entry = np.where(direction > 0, entry_p, 1.0 - entry_p)
    dollars_at_risk = entry_frac * bankroll
    shares = dollars_at_risk / np.maximum(cost_per_share_entry, 0.01)

    pnl_per_share = direction.values * (exit_p.values - entry_p.values)
    gross_pnl = pnl_per_share * shares
    fees = np.array([
        _fees(float(ep), float(xp), float(sh))
        for ep, xp, sh in zip(entry_p, exit_p, shares)
    ])

    if apply_slippage and has_spread_col:
        entry_spread = hits["spread"].astype(float).fillna(0.0).to_numpy()
        exit_spread = hits["_exit_spread"].astype(float).fillna(0.0).to_numpy()
        # Half-spread cost on each leg: as a taker you cross from mid to the
        # near-side quote, i.e. pay approximately half the quoted spread
        # relative to the mid price used elsewhere in this backtest.
        slippage = shares * 0.5 * (entry_spread + exit_spread)
    else:
        slippage = np.zeros(len(hits))

    net_pnl = gross_pnl - fees - slippage

    return pd.DataFrame({
        "market_id": hits["market_id"].values,
        "category": hits.get("category", pd.Series(["Uncategorized"] * len(hits))).values,
        "entry_ts": hits["timestamp"].values,
        "exit_ts": (hits["timestamp"] + pd.to_timedelta(horizon_hours, unit="h")).values,
        "horizon_hours": horizon_hours,
        "signal_col": signal_col,
        "signal_value": hits["_signal"].values,
        "direction": direction.values,
        "entry_price": entry_p.values,
        "exit_price": exit_p.values,
        "shares": shares,
        "gross_pnl": gross_pnl,
        "fees": fees,
        "slippage": slippage,
        "net_pnl": net_pnl,
        "return_frac": net_pnl / bankroll,
    })

def compute_risk_stats(trades: pd.DataFrame, bankroll: float = BANKROLL) -> Dict[str, float]:
    if trades.empty:
        return {
            "n_trades": 0, "n_markets": 0, "n_trading_days": 0,
            "win_rate": np.nan, "total_pnl": 0.0, "mean_return": np.nan,
            "daily_sharpe_annualized": np.nan, "daily_vol": np.nan,
            "worst_day_pnl": np.nan, "turnover_pct_per_day": np.nan,
            "total_fees": 0.0, "total_slippage": 0.0, "sharpe_reliable": False,
        }

    tr = trades.copy()
    tr["exit_date"] = pd.to_datetime(tr["exit_ts"]).dt.floor("D")
    daily_pnl = tr.groupby("exit_date")["net_pnl"].sum().sort_index()
    daily_ret = daily_pnl / bankroll
    n_days = int(daily_ret.shape[0])

    if n_days >= 2:
        mu = float(daily_ret.mean())
        sigma = float(daily_ret.std(ddof=1))
        sharpe = mu / sigma * np.sqrt(ANNUALIZATION_DAYS) if sigma > 0 else np.nan
    else:
        mu, sigma, sharpe = np.nan, np.nan, np.nan

    # Worst-day P&L replaces max_drawdown. It's the single-day loss floor
    # in dollar terms, which IS a meaningful signal-quality diagnostic
    # (measures downside dispersion on the daily-bucketed series) without
    # requiring a valid portfolio-equity curve.
    worst_day = float(daily_pnl.min()) if len(daily_pnl) else np.nan

    # Conventional dollar-turnover: total notional traded per day, as a
    # percentage of bankroll. Each trade's notional is shares * entry_price
    # for YES-side trades and shares * (1 - entry_price) for NO-side trades
    # -- but "shares" is already computed as dollars_at_risk / cost_per_share,
    # so shares * cost_per_share = dollars_at_risk = entry_frac * bankroll
    # per trade. So notional per trade = entry_frac * bankroll = $1000 by
    # default. Sum across trades / n_days / bankroll gives turnover as a
    # fraction of bankroll per day.
    total_notional = float((tr["shares"] * np.where(
        tr["direction"] > 0, tr["entry_price"], 1.0 - tr["entry_price"]
    )).sum())
    turnover_pct_per_day = (total_notional / max(n_days, 1)) / bankroll

    return {
        "n_trades": int(len(tr)),
        "n_markets": int(tr["market_id"].nunique()),
        "n_trading_days": n_days,
        "win_rate": float((tr["net_pnl"] > 0).mean()),
        "total_pnl": float(tr["net_pnl"].sum()),
        "mean_return": float(tr["return_frac"].mean()),
        "daily_sharpe_annualized": float(sharpe) if not np.isnan(sharpe) else np.nan,
        "daily_vol": sigma if not np.isnan(sigma) else np.nan,
        "worst_day_pnl": worst_day,
        "turnover_pct_per_day": turnover_pct_per_day,
        "total_fees": float(tr["fees"].sum()),
        "total_slippage": float(tr["slippage"].sum()) if "slippage" in tr.columns else 0.0,
        "sharpe_reliable": bool(n_days >= 5),
    }


def by_category(trades: pd.DataFrame, bankroll: float = BANKROLL) -> pd.DataFrame:
    if trades.empty or "category" not in trades.columns:
        return pd.DataFrame()
    rows = []
    for cat, g in trades.groupby("category"):
        stats = compute_risk_stats(g, bankroll=bankroll)
        stats["category"] = cat
        rows.append(stats)
    return pd.DataFrame(rows).set_index("category")


def by_period(trades: pd.DataFrame, period_freq: str = "M",
              bankroll: float = BANKROLL) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    tr = trades.copy()
    tr["period"] = pd.to_datetime(tr["exit_ts"]).dt.to_period(period_freq)
    rows = []
    for p, g in tr.groupby("period"):
        stats = compute_risk_stats(g, bankroll=bankroll)
        stats["period"] = str(p)
        rows.append(stats)
    return pd.DataFrame(rows).set_index("period")

STRATEGIES = ["mom_vn", "mom_naive", "rev_vn", "rev_naive", "unconditional"]


def run_full_grid(
    df: pd.DataFrame,
    strategies: Optional[List[str]] = None,
    horizons: Optional[List[int]] = None,
    entry_threshold: float = DEFAULT_ENTRY_THRESHOLD,
    entry_frac: float = DEFAULT_ENTRY_FRAC,
    apply_slippage: bool = True,
    train_mask: Optional[pd.Series] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Runs the full strategy x horizon grid. Returns dict with:
      trades:    concatenated per-trade log with strategy & horizon
      overall:   one row per (strategy, horizon) with the full risk suite
      by_cat:    long-form (strategy, horizon, category) risk suite
      by_period: long-form (strategy, horizon, period) risk suite

    apply_slippage : bool
        Passed through to generate_trades. See its docstring -- charges an
        additional half-spread cost on each leg using the panel's
        reconstructed `spread` column, on top of Kalshi's stated taker fee.

    train_mask : pd.Series[bool], optional
        Boolean mask over df's rows indicating the training portion. Used
        to calibrate naive-strategy entry thresholds without look-ahead:
        the 93rd-percentile threshold for mom_naive / rev_naive is
        computed from the TRAINING signal distribution only, then frozen
        and applied to test-period firing decisions. If None, falls back
        to the whole-panel quantile (which leaks mildly, since the test
        period contributes to the threshold calibration -- kept as a
        backwards-compat default but not recommended for the headline run).
    """
    strategies = strategies or STRATEGIES
    horizons = horizons or DEFAULT_HOLDING_HOURS

    all_trades: List[pd.DataFrame] = []
    overall_rows: List[Dict] = []
    cat_rows: List[Dict] = []
    per_rows: List[Dict] = []

    # Per-strategy thresholds. Vol-normalized signals are z-scores (unitless),
    # so 1.5 sigma is a natural choice. Naive signals are RAW price differences
    # (dollars) with typical magnitudes of ±0.02-0.10, so a fixed 1.5 threshold
    # would filter them to zero trades.
    #
    # For each naive signal, set the threshold to the 93rd percentile of its
    # absolute value. If a train_mask is provided, use ONLY the training
    # portion of the data to compute this quantile, then freeze and apply to
    # the whole panel (so the test-period firing decisions don't leak info
    # about the test-period signal distribution). If no train_mask is
    # provided, fall back to the whole-panel quantile with a warning.
    if train_mask is None:
        print("[run_full_grid] WARNING: no train_mask provided; naive "
              "thresholds calibrated on whole-panel quantile (mild leakage). "
              "For a clean run, pass train_mask=<boolean series of train rows>.")
        threshold_source = df
    else:
        # Align train_mask to df's index if needed
        if not train_mask.index.equals(df.index):
            train_mask = train_mask.reindex(df.index, fill_value=False)
        threshold_source = df[train_mask]

    naive_thresholds = {}
    for strat in ("mom_naive", "rev_naive"):
        if strat in df.columns:
            abs_signal = threshold_source[strat].abs().dropna()
            if len(abs_signal) > 100:
                naive_thresholds[strat] = float(abs_signal.quantile(0.93))
            else:
                naive_thresholds[strat] = entry_threshold  # fallback

    # test_mask: complement of train_mask, restricts trade firing to test rows.
    # Signals and exit-price lookups still use the full df (so lookbacks/exits
    # can span the train/test boundary), but ENTRY decisions only happen in
    # the test period.
    test_mask = ~train_mask if train_mask is not None else None

    for strat in strategies:
        for H in horizons:
            if strat == "unconditional":
                th = 0.0
            elif strat in naive_thresholds:
                th = naive_thresholds[strat]
            else:
                th = entry_threshold  # vol-normalized default
            trades = generate_trades(df, signal_col=strat, horizon_hours=H,
                                     entry_threshold=th, entry_frac=entry_frac,
                                     apply_slippage=apply_slippage,
                                     test_mask=test_mask)
            if trades.empty:
                continue
            trades["strategy"] = strat
            all_trades.append(trades)

            stats = compute_risk_stats(trades)
            stats["strategy"] = strat
            stats["horizon_hours"] = H
            overall_rows.append(stats)

            cat_df = by_category(trades)
            for cat, row in cat_df.iterrows():
                r = row.to_dict(); r["strategy"] = strat
                r["horizon_hours"] = H; r["category"] = cat
                cat_rows.append(r)

            per_df = by_period(trades)
            for per, row in per_df.iterrows():
                r = row.to_dict(); r["strategy"] = strat
                r["horizon_hours"] = H; r["period"] = per
                per_rows.append(r)

    trades_out = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    overall = pd.DataFrame(overall_rows)
    if not overall.empty:
        overall = overall.set_index(["strategy", "horizon_hours"]).sort_index()

    return {
        "trades": trades_out,
        "overall": overall,
        "by_cat": pd.DataFrame(cat_rows),
        "by_period": pd.DataFrame(per_rows),
    }