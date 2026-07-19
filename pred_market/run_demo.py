"""
Run the full pipeline end-to-end.

    python run_demo.py

Currently configured to pull REAL Polymarket history via the raw-REST
fallback (data_fetcher.build_market_panel, "Path 2"). Swap to
data_fetcher.build_market_panel_sdk(...) for the official-SDK path
("Path 1"), or to synthetic_data.simulate_market_panel(...) to go back to
the zero-network pipeline-validation demo -- everything downstream
(signals, backtest, metrics) is identical regardless of data source.

NOTE: build_market_panel() (Path 2) does not produce a 'spread' column, so
the AS (adverse-selection) channel of the structural volatility model will
be inactive here (K fits to 0) -- you'll get the DR-only structural model,
which is still a real, defensible model per the paper (see README), just
missing the second channel. The GARCH-on-residuals layer works regardless,
since it only needs h2 (always computed), not spread.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_fetcher import build_market_panel
from signals import add_signals, add_structural_signals
from backtest import run_strategy, trades_to_frame, compute_metrics, train_test_split_by_market
from config import RANDOM_SEED

pd.set_option("display.width", 120)


def banner():
    print("=" * 78)
    print("  REAL POLYMARKET DATA RUN (Path 2 / raw REST fallback)")
    print("  Results here reflect actual historical Polymarket markets --")
    print("  still read every caveat in the README before trusting a Sharpe number.")
    print("=" * 78)


def print_metrics(name: str, m: dict):
    print(f"\n--- {name} ---")
    if m.get("n_trades", 0) == 0:
        print("  No trades passed calibration (no positive-edge buckets found).")
        return
    print(f"  Trades (test set):        {m['n_trades']}")
    print(f"  Win rate:                 {m['win_rate']:.1%}")
    print(f"  Avg position size:        {m['avg_position_fraction']:.2%} of bankroll")
    print(f"  Total return (test):      {m['total_return_pct']:.2f}%")
    print(f"  Avg PnL / trade:          ${m['avg_pnl_per_trade']:.2f}")
    print(f"  Trade-level Sharpe:       {m['trade_level_sharpe']:.2f}")
    print(f"  Daily-aggregated Sharpe:  {m['daily_aggregated_sharpe']:.2f}   "
          f"(see README -- this number inflates for low-frequency strategies)")
    print(f"  Max drawdown:             {m['max_drawdown_pct']:.2f}%")


def main():
    banner()

    print("\n[1/4] Fetching real resolved Polymarket markets...")
    # build_market_panel() now has its own built-in checkpointing/resume
    # logic (checkpoint_path below) -- it saves progress every
    # checkpoint_every markets, and on a fresh call, skips any market_id
    # already present in that file. This subsumes the separate cache-check
    # wrapper that used to live here: if CACHE_PATH already has everything
    # you asked for, the resume logic sees that on its very first iteration
    # and returns almost instantly; if it's a full fresh start, there's
    # nothing to skip and it behaves like a normal fetch; if a previous run
    # crashed partway through (this is what motivated adding this at all --
    # a real ConnectionTerminated error lost 43 minutes of progress at
    # 1300/1500 markets with no checkpointing), it picks up from wherever
    # it stopped instead of starting over.
    CACHE_PATH = "data/real_panel_cache.parquet"
    panel = build_market_panel(min_volume=10_000, max_markets=1350,
                                checkpoint_path=CACHE_PATH, checkpoint_every=50, max_retries=5)
    if panel.empty:
        print("      Got 0 markets/rows back. Stopping here -- fix the fetch before")
        print("      going any further (see diagnose_fetch.py / the README's smoke-test")
        print("      instructions). No point running signals/backtest on empty data.")
        return
    print(f"      {panel['market_id'].nunique()} markets, {len(panel)} total price bars")

    print("[2/4] Computing logit-scale momentum & reversal signals...")
    panel = add_signals(panel, mom_lookback=5, rev_lookback=10)

    print("      Fitting DR-AS structural volatility model (K fit on train markets only)...")
    train_df_for_fit, _ = train_test_split_by_market(panel)
    # spread_col="spread" is harmless to pass even though build_market_panel()
    # doesn't produce one -- add_structural_signals checks for the column's
    # presence and falls back to DR-only (K=0) rather than erroring. See the
    # module docstring at the top of this file.
    panel, fitted_K = add_structural_signals(
        panel, train_market_ids=set(train_df_for_fit["market_id"]),
        struct_mom_lookback=5, struct_rev_lookback=10, spread_col="spread"
    )
    print(f"      Fitted AS-channel scale K = {fitted_K:.4f} "
          f"(expected 0.0 here -- no spread column from Path 2, so this is DR-only)")

    print("[3/4] Running backtests (train/test split by market start date, 65/35)...")
    mom_trades, mom_calib = run_strategy(panel, "momentum")
    rev_trades, rev_calib = run_strategy(panel, "reversal")
    smom_trades, smom_calib = run_strategy(panel, "structural_momentum")
    srev_trades, srev_calib = run_strategy(panel, "structural_reversal")

    mom_df = trades_to_frame(mom_trades)
    rev_df = trades_to_frame(rev_trades)
    smom_df = trades_to_frame(smom_trades)
    srev_df = trades_to_frame(srev_trades)
    all_frames = [d for d in [mom_df, rev_df, smom_df, srev_df] if not d.empty]
    combined_df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()

    mom_metrics = compute_metrics(mom_df)
    rev_metrics = compute_metrics(rev_df)
    smom_metrics = compute_metrics(smom_df)
    srev_metrics = compute_metrics(srev_df)
    combined_metrics = compute_metrics(combined_df) if not combined_df.empty else {"n_trades": 0}

    print("[4/4] Results:")
    print_metrics("MOMENTUM strategy (naive logit-diff)", mom_metrics)
    print_metrics("REVERSAL strategy (naive rolling z-score)", rev_metrics)
    print_metrics("STRUCTURAL MOMENTUM (DR-AS vol-normalized)", smom_metrics)
    print_metrics("STRUCTURAL REVERSAL (DR-AS vol-normalized)", srev_metrics)
    print_metrics("COMBINED portfolio (all six)", combined_metrics)

    print("\n--- Train-set calibration tables (frozen before touching test set) ---")
    for name, calib in [("momentum", mom_calib), ("reversal", rev_calib),
                        ("structural_momentum", smom_calib), ("structural_reversal", srev_calib)]:
        print(f"\n{name} buckets (win_rate, avg_price, n_train, z_stat, position_fraction):")
        for b, info in sorted(calib.items()):
            print(f"  bucket {b}: win_rate={info['win_rate']:.2f}  avg_price={info['avg_price']:.2f}  "
                  f"n={info['n_train']:3d}  z={info['z_stat']:+.2f}  size={info['position_fraction']:.2%}")

    # --- Save outputs (clear stale files first -- a strategy producing 0
    # trades this run should not leave behind a CSV from a previous run) ---
    import os
    all_trade_files = ["results/momentum_trades.csv", "results/reversal_trades.csv",
                       "results/structural_momentum_trades.csv", "results/structural_reversal_trades.csv",
                       "results/structural_momentum_garch_trades.csv", "results/structural_reversal_garch_trades.csv"]
    for fname in all_trade_files:
        if os.path.exists(fname):
            os.remove(fname)
    written = []
    for fname, d in zip(all_trade_files, [mom_df, rev_df, smom_df, srev_df]):
        if not d.empty:
            d.to_csv(fname, index=False)
            written.append(fname)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, m, color in [("Momentum (naive)", mom_metrics, "#2563eb"),
                           ("Reversal (naive)", rev_metrics, "#dc2626"),
                           ("Structural Momentum (DR-AS)", smom_metrics, "#7c3aed"),
                           ("Structural Reversal (DR-AS)", srev_metrics, "#ea580c"),
                           ("Combined (all four)", combined_metrics, "#16a34a")]:
        if m.get("n_trades", 0) > 0 and "equity_curve" in m:
            ax.plot(m["equity_curve"].index, m["equity_curve"].values, label=name, color=color, linewidth=1.6)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title("Real Polymarket backtest equity curves (test set)")
    ax.set_ylabel("Equity (starting = 1.0)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/equity_curves.png", dpi=140)
    written.append("results/equity_curves.png")

    print(f"\nSaved: {', '.join(written)}")


if __name__ == "__main__":
    main()