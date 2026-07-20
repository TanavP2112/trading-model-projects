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

    print("\n[1/4] Fetching real resolved Polymarket sports markets...")
    # tag_slug="sports" filters SERVER-SIDE via list_markets(tag_id=...) --
    # the real tag_id is looked up at runtime via get_tag(slug="sports")
    # rather than hardcoded, so this self-corrects if Polymarket ever
    # changes tag IDs. This is meaningfully better than fetching broadly
    # and filtering client-side afterward: the whole max_markets budget now
    # goes toward sports specifically, instead of being diluted across
    # every category and mostly thrown away (sports was previously getting
    # maybe 40% of a pooled fetch; now it gets 100%).
    #
    # Separate cache file from the pooled fetch on purpose -- don't want a
    # sports-only run to silently overwrite the full multi-category dataset
    # you already fetched, in case you want both.
    CACHE_PATH = "data/real_panel_cache_sports.parquet"
    panel = build_market_panel(min_volume=10_000, max_markets=1500, tag_slug="sports",
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
    #
    # min_tau/bar_length=1/24 (one hour, in days): MUST match this data's
    # actual fidelity_minutes=60 fetch. A real bug was found and fixed by
    # comparing predicted vs realized variance on this exact data: without
    # this override, h2 defaults to assuming DAILY bars and comes out
    # systematically ~24x too large, which inflates the denominator of
    # every structural signal and makes genuine edges harder to detect,
    # not just a cosmetic scaling issue.
    BAR_LENGTH_DAYS = 1 / 24
    panel, fitted_K, garch_params, joint_garch_params = add_structural_signals(
        panel, train_market_ids=set(train_df_for_fit["market_id"]),
        struct_mom_lookback=5, struct_rev_lookback=10, spread_col="spread",
        fit_garch=True, fit_joint_garch=True,
        min_tau=BAR_LENGTH_DAYS, bar_length=BAR_LENGTH_DAYS,
    )
    print(f"      Fitted AS-channel scale K = {fitted_K:.4f} "
          f"(expected 0.0 here -- no spread column from Path 2, so this is DR-only)")
    if joint_garch_params is not None:
        print(f"      Fitted JOINT additive GARCH+DR-AS: K={joint_garch_params['K']:.4f}  "
              f"omega={joint_garch_params['omega']:.6f}  alpha={joint_garch_params['alpha']:.4f}  "
              f"beta={joint_garch_params['beta']:.4f}  c={joint_garch_params['c']:.4f}  "
              f"persistence={joint_garch_params['persistence']:.4f}  success={joint_garch_params['success']}")
    if garch_params is not None:
        print(f"      Fitted GARCH(1,1) on structural residuals: omega={garch_params['omega']:.4f}  "
              f"alpha={garch_params['alpha']:.4f}  beta={garch_params['beta']:.4f}  "
              f"persistence(a+b)={garch_params['alpha']+garch_params['beta']:.4f}  "
              f"uncond_var={garch_params['uncond_var']:.4f}  nu={garch_params.get('nu')}  fit_ok={garch_params['fit_ok']}")
        if garch_params['uncond_var'] > 5.0:
            print(f"      !! uncond_var={garch_params['uncond_var']:.2f} is far from the ~1.0 a healthy fit "
                  f"should give (z is constructed to have unit variance if h2 is well-calibrated) -- "
                  f"the multiplicative GARCH layer is probably overstating variance broadly on this data. "
                  f"Absolute cap in garch_multiplier_per_market will still bound it, but treat "
                  f"struct_*_garch_signal results with real skepticism until this is understood.")

    print("[3/4] Running backtests (train/test split by market start date, 65/35)...")
    mom_trades, mom_calib = run_strategy(panel, "momentum")
    rev_trades, rev_calib = run_strategy(panel, "reversal")
    smom_trades, smom_calib = run_strategy(panel, "structural_momentum")
    srev_trades, srev_calib = run_strategy(panel, "structural_reversal")
    smomg_trades, smomg_calib = run_strategy(panel, "structural_momentum_garch")
    srevg_trades, srevg_calib = run_strategy(panel, "structural_reversal_garch")
    smomjg_trades, smomjg_calib = run_strategy(panel, "structural_momentum_jointgarch")
    srevjg_trades, srevjg_calib = run_strategy(panel, "structural_reversal_jointgarch")

    mom_df = trades_to_frame(mom_trades)
    rev_df = trades_to_frame(rev_trades)
    smom_df = trades_to_frame(smom_trades)
    srev_df = trades_to_frame(srev_trades)
    smomg_df = trades_to_frame(smomg_trades)
    srevg_df = trades_to_frame(srevg_trades)
    smomjg_df = trades_to_frame(smomjg_trades)
    srevjg_df = trades_to_frame(srevjg_trades)
    all_frames = [d for d in [mom_df, rev_df, smom_df, srev_df, smomg_df, srevg_df, smomjg_df, srevjg_df]
                  if not d.empty]
    combined_df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()

    mom_metrics = compute_metrics(mom_df)
    rev_metrics = compute_metrics(rev_df)
    smom_metrics = compute_metrics(smom_df)
    srev_metrics = compute_metrics(srev_df)
    smomg_metrics = compute_metrics(smomg_df)
    srevg_metrics = compute_metrics(srevg_df)
    smomjg_metrics = compute_metrics(smomjg_df)
    srevjg_metrics = compute_metrics(srevjg_df)
    combined_metrics = compute_metrics(combined_df) if not combined_df.empty else {"n_trades": 0}

    print("[4/4] Results:")
    print_metrics("MOMENTUM strategy (naive logit-diff)", mom_metrics)
    print_metrics("REVERSAL strategy (naive rolling z-score)", rev_metrics)
    print_metrics("STRUCTURAL MOMENTUM (DR-AS vol-normalized)", smom_metrics)
    print_metrics("STRUCTURAL REVERSAL (DR-AS vol-normalized)", srev_metrics)
    print_metrics("STRUCTURAL MOMENTUM + GARCH (multiplicative, approximate)", smomg_metrics)
    print_metrics("STRUCTURAL REVERSAL + GARCH (multiplicative, approximate)", srevg_metrics)
    print_metrics("STRUCTURAL MOMENTUM + GARCH (joint additive, paper-correct)", smomjg_metrics)
    print_metrics("STRUCTURAL REVERSAL + GARCH (joint additive, paper-correct)", srevjg_metrics)
    print_metrics("COMBINED portfolio (all eight)", combined_metrics)

    print("\n--- Train-set calibration tables (frozen before touching test set) ---")
    for name, calib in [("momentum", mom_calib), ("reversal", rev_calib),
                        ("structural_momentum", smom_calib), ("structural_reversal", srev_calib),
                        ("structural_momentum_garch", smomg_calib), ("structural_reversal_garch", srevg_calib),
                        ("structural_momentum_jointgarch", smomjg_calib),
                        ("structural_reversal_jointgarch", srevjg_calib)]:
        print(f"\n{name} buckets (win_rate, avg_price, n_train, z_stat, position_fraction):")
        for b, info in sorted(calib.items()):
            print(f"  bucket {b}: win_rate={info['win_rate']:.2f}  avg_price={info['avg_price']:.2f}  "
                  f"n={info['n_train']:3d}  z={info['z_stat']:+.2f}  size={info['position_fraction']:.2%}")

    # --- Save outputs (clear stale files first -- a strategy producing 0
    # trades this run should not leave behind a CSV from a previous run) ---
    import os
    all_trade_files = ["results/momentum_trades.csv", "results/reversal_trades.csv",
                       "results/structural_momentum_trades.csv", "results/structural_reversal_trades.csv",
                       "results/structural_momentum_garch_trades.csv", "results/structural_reversal_garch_trades.csv",
                       "results/structural_momentum_jointgarch_trades.csv",
                       "results/structural_reversal_jointgarch_trades.csv"]
    for fname in all_trade_files:
        if os.path.exists(fname):
            os.remove(fname)
    written = []
    all_dfs = [mom_df, rev_df, smom_df, srev_df, smomg_df, srevg_df, smomjg_df, srevjg_df]
    for fname, d in zip(all_trade_files, all_dfs):
        if not d.empty:
            d.to_csv(fname, index=False)
            written.append(fname)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, m, color in [("Momentum (naive)", mom_metrics, "#2563eb"),
                           ("Reversal (naive)", rev_metrics, "#dc2626"),
                           ("Structural Momentum (DR-AS)", smom_metrics, "#7c3aed"),
                           ("Structural Reversal (DR-AS)", srev_metrics, "#ea580c"),
                           ("Structural Momentum + GARCH (mult.)", smomg_metrics, "#0891b2"),
                           ("Structural Reversal + GARCH (mult.)", srevg_metrics, "#be185d"),
                           ("Structural Momentum + GARCH (joint)", smomjg_metrics, "#65a30d"),
                           ("Structural Reversal + GARCH (joint)", srevjg_metrics, "#c2410c"),
                           ("Combined (all eight)", combined_metrics, "#16a34a")]:
        if m.get("n_trades", 0) > 0 and "equity_curve" in m:
            ax.plot(m["equity_curve"].index, m["equity_curve"].values, label=name, color=color, linewidth=1.6)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title("Real Polymarket backtest equity curves (test set)")
    ax.set_ylabel("Equity (starting = 1.0)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No strategy produced any trades this run", ha="center", va="center",
                transform=ax.transAxes, color="grey")
    fig.tight_layout()
    fig.savefig("results/equity_curves.png", dpi=140)
    written.append("results/equity_curves.png")

    print(f"\nSaved: {', '.join(written)}")


if __name__ == "__main__":
    main()