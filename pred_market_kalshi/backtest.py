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
) -> pd.DataFrame:
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
            "max_drawdown": np.nan, "turnover_per_day": np.nan,
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

    equity = bankroll + daily_pnl.cumsum()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = float(drawdown.min()) if len(drawdown) else np.nan

    turnover_per_day = float(tr["shares"].sum() / max(n_days, 1))

    return {
        "n_trades": int(len(tr)),
        "n_markets": int(tr["market_id"].nunique()),
        "n_trading_days": n_days,
        "win_rate": float((tr["net_pnl"] > 0).mean()),
        "total_pnl": float(tr["net_pnl"].sum()),
        "mean_return": float(tr["return_frac"].mean()),
        "daily_sharpe_annualized": float(sharpe) if not np.isnan(sharpe) else np.nan,
        "daily_vol": sigma if not np.isnan(sigma) else np.nan,
        "max_drawdown": max_dd,
        "turnover_per_day": turnover_per_day,
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
) -> Dict[str, pd.DataFrame]:
    strategies = strategies or STRATEGIES
    horizons = horizons or DEFAULT_HOLDING_HOURS

    all_trades: List[pd.DataFrame] = []
    overall_rows: List[Dict] = []
    cat_rows: List[Dict] = []
    per_rows: List[Dict] = []
    naive_thresholds = {}
    for strat in ("mom_naive", "rev_naive"):
        if strat in df.columns:
            abs_signal = df[strat].abs().dropna()
            if len(abs_signal) > 100:
                naive_thresholds[strat] = float(abs_signal.quantile(0.93))
            else:
                naive_thresholds[strat] = entry_threshold  # fallback

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
                                     apply_slippage=apply_slippage)
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