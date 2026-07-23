import argparse
import os
import sys

import pandas as pd
import numpy as np

from trading.signals import (
    add_all_signals,
    get_phase1_winners,
)
from backtest import run_full_grid

# Constants used by the fold-level Sharpe computation. Kept module-level
# rather than duplicated in-loop for readability.
_BANKROLL = 100_000.0
_ANNUALIZATION_DAYS = 365.0


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default="data/kalshi_hf_panel.parquet",
                    help="Path to the hourly panel parquet (default: data/kalshi_hf_panel.parquet)")
    ap.add_argument("--model", default=None,
                    help="Vol model. Omit to load per-category Phase 1 winners dynamically. "
                         "Pass 'DR' / 'DR-AS' / 'GARCH' / 'GARCH+DR-AS' to apply one model uniformly.")
    ap.add_argument("--min-train-months", type=int, default=1,
                    help="Minimum number of months to use as initial training window "
                         "before the first out-of-sample test month (default: 1).")
    ap.add_argument("--out-dir", default="data/walk_forward",
                    help="Where to write walk-forward trade log + summary tables "
                         "(default: data/walk_forward/)")
    return ap.parse_args()


def _resolve_panel_path(explicit: str | None) -> str | None:
    if explicit:
        return explicit if os.path.exists(explicit) else None
    candidates = [
        "data/kalshi_hf_panel.parquet",
        "kalshi_hf_panel.parquet",
        os.path.expanduser("~/kalshi_hf_panel.parquet"),
        "/mnt/user-data/uploads/kalshi_hf_panel.parquet",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _fold_sharpe_stats(g: pd.DataFrame) -> pd.Series:
    fold_sharpes = []
    for _, fg in g.groupby("eval_period"):
        daily = fg.groupby("exit_date")["net_pnl"].sum() / _BANKROLL
        if len(daily) < 2 or daily.std(ddof=1) == 0:
            continue
        fold_sharpes.append(daily.mean() / daily.std(ddof=1) * np.sqrt(_ANNUALIZATION_DAYS))
    if not fold_sharpes:
        return pd.Series({
            "mean_fold_sharpe": np.nan,
            "std_fold_sharpe": np.nan,
            "n_folds_with_sharpe": 0,
        })
    arr = np.array(fold_sharpes)
    return pd.Series({
        "mean_fold_sharpe": float(arr.mean()),
        "std_fold_sharpe": float(arr.std(ddof=1)) if len(arr) >= 2 else np.nan,
        "n_folds_with_sharpe": int(len(arr)),
    })


def run_expanding_walk_forward(
    df: pd.DataFrame,
    vol_model=None,
    min_train_months: int = 1,
) -> dict:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["period"] = df["timestamp"].dt.to_period("M")

    periods = sorted(df["period"].unique())
    print(f"[Info] Found {len(periods)} monthly periods "
          f"({periods[0]} to {periods[-1]})")
    if len(periods) <= min_train_months:
        raise ValueError(
            f"Panel contains {len(periods)} periods, but min_train_months="
            f"{min_train_months}. Not enough periods for walk-forward."
        )
    if vol_model is None:
        vol_model = get_phase1_winners()
        print(f"[Info] Using dynamic Phase 1 winners: {vol_model}")

    monthly_test_results = []
    monthly_trade_logs = []

    for idx in range(min_train_months, len(periods)):
        test_period = periods[idx]
        train_periods = periods[:idx]

        print(f"\n{'=' * 70}")
        print(f"FOLD {idx}/{len(periods) - 1}: "
              f"train <= {train_periods[-1]} ({len(train_periods)} months) | "
              f"test = {test_period}")
        print("=" * 70)
        fold_df = df[df["period"].isin(train_periods + [test_period])].copy()
        fold_train_mask = fold_df["period"].isin(train_periods)
        train_bars = int(fold_train_mask.sum())
        test_bars = int((~fold_train_mask).sum())
        print(f"[fold {idx}] {train_bars:,} train / {test_bars:,} test bars")
        fold_df = add_all_signals(
            fold_df, train_mask=fold_train_mask, model=vol_model
        )
        fold_results = run_full_grid(fold_df, train_mask=fold_train_mask)

        # Tag results with this fold's eval period so concentration diagnostics
        # downstream can group by fold.
        if not fold_results["overall"].empty:
            df_overall = fold_results["overall"].reset_index().copy()
            df_overall["eval_period"] = str(test_period)
            monthly_test_results.append(df_overall)
        if not fold_results["trades"].empty:
            df_trades = fold_results["trades"].copy()
            df_trades["eval_period"] = str(test_period)
            monthly_trade_logs.append(df_trades)

    if not monthly_test_results:
        print("\n[Warning] No trades generated across any walk-forward fold.")
        return {"overall": pd.DataFrame(), "by_cat": pd.DataFrame(),
                "by_period": pd.DataFrame(), "trades": pd.DataFrame()}

    full_wf_results = pd.concat(monthly_test_results, ignore_index=True)
    full_wf_trades = (pd.concat(monthly_trade_logs, ignore_index=True)
                     if monthly_trade_logs else pd.DataFrame())
    agg_overall = (
        full_wf_results.groupby(["strategy", "horizon_hours"])
        .agg(
            n_trades=("n_trades", "sum"),
            n_markets=("n_markets", "max"),
            n_eval_periods=("eval_period", "nunique"),
            win_rate=("win_rate", "mean"),
            mean_return=("mean_return", "mean"),
            daily_sharpe_annualized=("daily_sharpe_annualized", "mean"),
            daily_vol=("daily_vol", "mean"),
            worst_day_pnl=("worst_day_pnl", "min"),
            turnover_pct_per_day=("turnover_pct_per_day", "mean"),
            total_fees=("total_fees", "sum"),
        )
        .reset_index()
    )
    if "category" in full_wf_trades.columns and not full_wf_trades.empty:
        tr = full_wf_trades.copy()
        tr["exit_date"] = pd.to_datetime(tr["exit_ts"]).dt.floor("D")

        def _pooled_sharpe(g: pd.DataFrame) -> float:
            daily = g.groupby("exit_date")["net_pnl"].sum() / _BANKROLL
            if len(daily) < 2 or daily.std(ddof=1) == 0:
                return np.nan
            return float(daily.mean() / daily.std(ddof=1) * np.sqrt(_ANNUALIZATION_DAYS))

        agg_rows = []
        for (strat, H, cat), g in tr.groupby(["strategy", "horizon_hours", "category"]):
            pooled_s = _pooled_sharpe(g)
            n_days = int(g["exit_date"].nunique())
            fs = _fold_sharpe_stats(g)
            daily_pnl_cat = g.groupby("exit_date")["net_pnl"].sum().sort_index()
            daily_ret_cat = daily_pnl_cat / _BANKROLL
            daily_vol_val = float(daily_ret_cat.std(ddof=1)) if len(daily_ret_cat) >= 2 else float("nan")
            worst_day_val = float(daily_pnl_cat.min()) if len(daily_pnl_cat) else float("nan")

            # Notional turnover per day as a fraction of bankroll.
            total_notional = float((g["shares"] * np.where(
                g["direction"] > 0, g["entry_price"], 1.0 - g["entry_price"]
            )).sum())
            turnover_val = (total_notional / max(n_days, 1)) / _BANKROLL if n_days > 0 else float("nan")

            agg_rows.append({
                "strategy": strat, "horizon_hours": H, "category": cat,
                "n_trades": int(len(g)),
                "n_markets": int(g["market_id"].nunique()),
                "n_trading_days": n_days,
                "n_folds_with_sharpe": int(fs["n_folds_with_sharpe"]),
                "win_rate": float((g["net_pnl"] > 0).mean()),
                "mean_pnl": float(g["net_pnl"].mean()),
                "total_pnl": float(g["net_pnl"].sum()),
                "total_fees": float(g["fees"].sum()),
                "total_slippage": float(g["slippage"].sum()) if "slippage" in g.columns else 0.0,
                "pooled_sharpe_annualized": pooled_s,
                "mean_fold_sharpe": fs["mean_fold_sharpe"],
                "std_fold_sharpe": fs["std_fold_sharpe"],
                "daily_vol": daily_vol_val,
                "worst_day_pnl": worst_day_val,
                "turnover_pct_per_day": turnover_val,
                "sharpe_reliable": bool(n_days >= 5 and fs["n_folds_with_sharpe"] >= 3),
            })
        agg_by_cat = pd.DataFrame(agg_rows)
    else:
        agg_by_cat = pd.DataFrame()

    return {
        "overall": agg_overall,
        "by_cat": agg_by_cat,
        "by_period": full_wf_results,
        "trades": full_wf_trades,
    }


def main():
    args = _parse_args()
    panel_path = _resolve_panel_path(args.panel)
    if panel_path is None:
        print("[Error] Could not find kalshi_hf_panel.parquet in any candidate location.")
        print("        Either move the parquet or pass --panel PATH.")
        sys.exit(1)
    print(f"[1/4] Loading panel from {panel_path}...")
    df = pd.read_parquet(panel_path)
    print(f"      {len(df):,} bars   {df['market_id'].nunique()} markets   "
          f"span {df['timestamp'].min()} -> {df['timestamp'].max()}")
    if "category" not in df.columns:
        raise ValueError("Panel is missing the 'category' column; rebuild via data_fetcher.py.")
    from scripts.data_fetcher import reindex_to_hourly_grid
    n_before = len(df)
    df = reindex_to_hourly_grid(df)
    n_after = len(df)
    n_active = int(df["is_clean_bar"].sum())
    print(f"      Reindexed to gap-free hourly grid: {n_before:,} -> {n_after:,} bars "
          f"({n_active:,} active / {n_after - n_active:,} forward-filled empty hours)")
    print(f"\n[2/4] Running expanding-window monthly walk-forward...")
    if args.model:
        print(f"      Using single model: {args.model}")
        vol_model = args.model
    else:
        print(f"      Using per-category Phase 1 winners (loaded dynamically)")
        vol_model = None  # add_all_signals will call get_phase1_winners()

    results = run_expanding_walk_forward(
        df, vol_model=vol_model, min_train_months=args.min_train_months
    )
    print("\n" + "=" * 70)
    print("[3/4] EXPANDING WALK-FORWARD OVERALL (aggregated across all OOS folds)")
    print("=" * 70)
    if results["overall"].empty:
        print("(no trades generated across any fold)")
    else:
        cols = ["strategy", "horizon_hours", "n_trades", "n_eval_periods",
                "win_rate", "daily_sharpe_annualized", "worst_day_pnl",
                "turnover_pct_per_day", "total_fees"]
        with pd.option_context("display.float_format", "{:,.4f}".format,
                                "display.width", 200):
            print(results["overall"][cols].to_string(index=False))

    if not results["by_cat"].empty:
        print("\n" + "=" * 70)
        print("BY CATEGORY (with concentration diagnostic)")
        print("=" * 70)
        cols = ["strategy", "horizon_hours", "category", "n_trades",
                "n_folds_with_sharpe", "win_rate", "total_pnl",
                "pooled_sharpe_annualized", "mean_fold_sharpe",
                "std_fold_sharpe", "sharpe_reliable"]
        with pd.option_context("display.float_format", "{:,.4f}".format,
                                "display.width", 200):
            print(results["by_cat"][cols].to_string(index=False))
        print("\nHow to read the concentration diagnostic:")
        print("  If pooled_sharpe >> mean_fold_sharpe, or std_fold_sharpe is large,")
        print("  the pooled Sharpe is driven by a small number of high-value folds")
        print("  rather than a stable per-fold edge -- event-concentration pattern.")
    os.makedirs(args.out_dir, exist_ok=True)
    if not results["overall"].empty:
        results["overall"].to_csv(f"{args.out_dir}/'phase2_overall.csv", index=False)
        results["by_period"].to_csv(f"{args.out_dir}/phase2_by_period.csv", index=False)
        if not results["by_cat"].empty:
            results["by_cat"].to_csv(f"{args.out_dir}/phase2_by_category.csv", index=False)
        if not results["trades"].empty:
            results["trades"].to_parquet(f"{args.out_dir}/phase2_trades.parquet")
        print(f"\n[4/4] Saved walk-forward results to {args.out_dir}/")


if __name__ == "__main__":
    main()