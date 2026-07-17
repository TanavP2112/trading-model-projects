"""
Run the full pipeline end-to-end on SYNTHETIC data to validate that
everything fits together correctly, and to show what the report output
looks like.

    python run_demo.py

To run against REAL Polymarket history instead, replace the
`simulate_market_panel(...)` call below with:

    from data_fetcher import build_market_panel
    panel = build_market_panel(min_volume=50_000, max_markets=300)

everything downstream (signals, backtest, metrics) is identical.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synthetic_data import simulate_market_panel
from signals import add_signals, add_structural_signals
from backtest import run_strategy, trades_to_frame, compute_metrics, train_test_split_by_market
from config import RANDOM_SEED

pd.set_option("display.width", 120)


def banner():
    print("=" * 78)
    print("  SYNTHETIC DATA DEMO -- validates pipeline mechanics only.")
    print("  This does NOT demonstrate real edge on Polymarket.")
    print("  Swap in data_fetcher.build_market_panel() for real history before")
    print("  drawing ANY conclusion about live tradeability.")
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

    print("\n[1/4] Simulating synthetic resolved-market panel...")
    panel = simulate_market_panel(n_markets=3500, seed=RANDOM_SEED)
    print(f"      {panel['market_id'].nunique()} markets, {len(panel)} total price bars")

    print("[2/4] Computing logit-scale momentum & reversal signals...")
    panel = add_signals(panel, mom_lookback=5, rev_lookback=10)

    print("      Fitting DR-AS structural volatility model (K fit on train markets only)...")
    train_df_for_fit, _ = train_test_split_by_market(panel)
    panel, fitted_K = add_structural_signals(
        panel, train_market_ids=set(train_df_for_fit["market_id"]),
        struct_mom_lookback=5, struct_rev_lookback=10, spread_col="spread",
    )
    print(f"      Fitted AS-channel scale K = {fitted_K:.4f} "
          f"(K=0 would mean DR-only; synthetic data includes a spread column so both channels are active)")

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
    print_metrics("COMBINED portfolio (all four)", combined_metrics)

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
    for fname in ["results/momentum_trades.csv", "results/reversal_trades.csv",
                  "results/structural_momentum_trades.csv", "results/structural_reversal_trades.csv"]:
        if os.path.exists(fname):
            os.remove(fname)
    written = []
    for fname, d in [("results/momentum_trades.csv", mom_df),
                     ("results/reversal_trades.csv", rev_df),
                     ("results/structural_momentum_trades.csv", smom_df),
                     ("results/structural_reversal_trades.csv", srev_df)]:
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
    ax.set_title("Synthetic backtest equity curves (test set) -- illustrative only")
    ax.set_ylabel("Equity (starting = 1.0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/equity_curves.png", dpi=140)
    written.append("results/equity_curves.png")

    print(f"\nSaved: {', '.join(written)}")


if __name__ == "__main__":
    main()
