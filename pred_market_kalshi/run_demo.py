import argparse
import os
import sys
import pandas as pd
import numpy as np

from trading.signals import add_all_signals, PHASE1_WINNER_BY_CATEGORY
from backtest import run_full_grid


def run_expanding_walk_forward(
    df: pd.DataFrame,
    vol_model=PHASE1_WINNER_BY_CATEGORY,
    min_train_months: int = 1,
) -> dict:
    """Executes an expanding-window monthly walk-forward backtest.
    
    Fits models on prior observations [1..t-1] and evaluates out-of-sample on month t.
    """
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["period"] = df["timestamp"].dt.to_period("M")
    
    periods = sorted(df["period"].unique())
    print(f"[Info] Found {len(periods)} total monthly periods in panel: {[str(p) for p in periods]}")

    if len(periods) <= min_train_months:
        raise ValueError(
            f"Panel contains {len(periods)} periods, but min_train_months={min_train_months}. "
            "Not enough periods for walk-forward evaluation."
        )

    monthly_test_results = []
    monthly_trade_logs = []

    # Iterate through out-of-sample test months starting after initial training periods
    for idx in range(min_train_months, len(periods)):
        test_period = periods[idx]
        train_periods = periods[:idx]

        print(f"\n" + "=" * 70)
        print(f"WALK-FORWARD FOLD {idx}/{len(periods) - 1}: Train <= {train_periods[-1]} | Test = {test_period}")
        print("=" * 70)

        # 1. Define strict chronological masks
        train_mask = df["period"].isin(train_periods)
        test_mask = df["period"] == test_period

        train_bars = train_mask.sum()
        test_bars = test_mask.sum()
        print(f"[Fold {idx}] Train size: {train_bars:,} bars | Test size: {test_bars:,} bars")

        # 2. Slice slice of panel up to current test month to prevent look-ahead bias
        fold_df = df[df["period"].isin(train_periods + [test_period])].copy()
        
        # Train-mask local to fold slice
        fold_train_mask = fold_df["period"].isin(train_periods)

        # 3. Fit volatility models strictly on training window and generate signals
        fold_df = add_all_signals(fold_df, train_mask=fold_train_mask, model=vol_model)

        # 4. Extract OUT-OF-SAMPLE test segment only
        fold_test_df = fold_df[fold_df["period"] == test_period].copy()

        # 5. Run backtest grid on test month
        fold_results = run_full_grid(fold_test_df)

        # Tag results with current evaluation period
        if not fold_results["overall"].empty:
            df_overall = fold_results["overall"].reset_index()
            df_overall["eval_period"] = str(test_period)
            monthly_test_results.append(df_overall)

        if not fold_results["trades"].empty:
            df_trades = fold_results["trades"].copy()
            df_trades["eval_period"] = str(test_period)
            monthly_trade_logs.append(df_trades)

    # -----------------------------------------------------------------------
    # Aggregate Walk-Forward Results Across All Out-of-Sample Periods
    # -----------------------------------------------------------------------
    if not monthly_test_results:
        print("\n[Warning] No trades generated across any walk-forward period.")
        return {"overall": pd.DataFrame(), "by_cat": pd.DataFrame(), "trades": pd.DataFrame()}

    full_wf_results = pd.concat(monthly_test_results, ignore_index=True)
    full_wf_trades = pd.concat(monthly_trade_logs, ignore_index=True) if monthly_trade_logs else pd.DataFrame()

    # Aggregate Overall Statistics Across All Folds
    agg_overall = (
        full_wf_results.groupby(["strategy", "horizon_hours"])
        .agg(
            n_trades=("n_trades", "sum"),
            n_markets=("n_markets", "max"),
            n_eval_periods=("eval_period", "nunique"),
            win_rate=("win_rate", "mean"),
            mean_return=("mean_return", "mean"),
            daily_sharpe_annualized=("daily_sharpe_annualized", "mean"),
            max_drawdown=("max_drawdown", "min"),
            total_fees=("total_fees", "sum"),
        )
        .reset_index()
    )

    # Aggregate By-Category Performance Across Out-of-Sample Folds.
    # backtest.generate_trades already produces net_pnl (direction-aware,
    # share-sized, round-trip fees deducted) -- USE IT DIRECTLY. Earlier
    # versions tried to recompute PnL as (exit_price - price) * volume,
    # which fell through to zero because the column is called entry_price
    # (not price), and even the intended fallback formula was wrong: it
    # would have ignored direction (flipping sign on NO trades), used
    # panel volume instead of shares held, and skipped fees.
    if "category" in full_wf_trades.columns and not full_wf_trades.empty:
        agg_by_cat = (
            full_wf_trades.groupby(["strategy", "horizon_hours", "category"])
            .agg(
                n_trades=("market_id", "count"),
                win_rate=("net_pnl", lambda x: (x > 0).mean() if len(x) > 0 else 0.0),
                mean_pnl=("net_pnl", "mean"),
                total_pnl=("net_pnl", "sum"),
                total_fees=("fees", "sum"),
            )
            .reset_index()
        )
    else:
        agg_by_cat = pd.DataFrame()

    return {
        "overall": agg_overall,
        "by_cat": agg_by_cat,
        "by_period": full_wf_results,
        "trades": full_wf_trades,
    }


def main():
    parser = argparse.ArgumentParser(description="Expanding Monthly Walk-Forward Benchmark")
    parser.add_argument("--panel", default="data/kalshi_hf_panel.parquet", help="Path to parquet panel")
    parser.add_argument("--out-dir", default="data/walk_forward", help="Output directory")
    args = parser.parse_args()

    if not os.path.exists(args.panel):
        print(f"[Error] Panel path '{args.panel}' does not exist.")
        sys.exit(1)

    print(f"[1/3] Loading panel from {args.panel}...")
    df = pd.read_parquet(args.panel)

    print(f"[2/3] Running Expanding Monthly Walk-Forward Evaluation...")
    wf_metrics = run_expanding_walk_forward(df, min_train_months=1)

    print("\n" + "=" * 80)
    print("EXPANDING WALK-FORWARD OVERALL PERFORMANCE (AGGREGATED OOS)")
    print("=" * 80)
    if not wf_metrics["overall"].empty:
        with pd.option_context("display.float_format", "{:,.4f}".format, "display.width", 200):
            print(wf_metrics["overall"].to_string(index=False))

    if not wf_metrics["by_cat"].empty:
        print("\n" + "=" * 80)
        print("OUT-OF-SAMPLE PERFORMANCE BY MARKET CATEGORY")
        print("=" * 80)
        with pd.option_context("display.float_format", "{:,.4f}".format, "display.width", 200):
            print(wf_metrics["by_cat"].to_string(index=False))

    # Save Walk-Forward Results
    os.makedirs(args.out_dir, exist_ok=True)
    if not wf_metrics["overall"].empty:
        wf_metrics["overall"].to_csv(f"{args.out_dir}/wf_overall.csv", index=False)
        wf_metrics["by_period"].to_csv(f"{args.out_dir}/wf_by_period.csv", index=False)
        if not wf_metrics["by_cat"].empty:
            wf_metrics["by_cat"].to_csv(f"{args.out_dir}/wf_by_category.csv", index=False)
        print(f"\n[Saved] Walk-forward CSV reports written to {args.out_dir}/")


if __name__ == "__main__":
    main()