"""
Volatility risk premium diagnostic.

A different hypothesis from everything else tested: not "is price wrong"
(momentum/reversal/calibration), but "is the market's own SENSE of its
uncertainty wrong." Borrowed from the options-market "variance risk
premium" literature: implied volatility tends to systematically overstate
subsequently realized volatility on average -- a structural premium
collected by whoever is effectively selling insurance against big moves.

Here, the structural DR-AS model's h^2 plays the role of "implied"
variance (it's the market's price/time-state-implied uncertainty, not a
fitted/backward-looking statistic), and the actual squared price change
plays the role of "realized" variance. We test whether h^2 systematically
over- or under-predicts what actually happens:

    VRP = h^2 (predicted) - realized_sq_move (actual)

VRP > 0 on average: the model (and by extension, the market's own implied
uncertainty) overstates realized volatility -- markets move less than
their structural "should." This is the classic risk-premium direction.

VRP < 0 on average: realized moves exceed what the model expects --
either the model is misspecified for that regime, or something genuinely
unusual is happening (real information shocks, or per the manipulation-
detection literature, potentially unusual/manipulative order flow).

METHODOLOGY NOTE: this is a pure diagnostic on FITTED-PARAMETER-FREE h^2
(K=0, DR-only -- no spread data, so no fitting/leakage concern at all,
unlike the AS channel or GARCH layer elsewhere in this project). No
train/test split is needed here since nothing is being fit; this can run
directly on the full cached panel.

SIGNIFICANCE TEST: this compares two continuous variance estimates, not
two proportions, so it does NOT have the Wald-vs-score boundary pathology
that broke calibration_check.py's original formula. This uses a standard
one-sample t-test on the paired difference (h^2 - realized_sq_move),
which is the statistically appropriate test for "is this paired
difference significantly different from zero" -- validated against
synthetic data with a known injected premium before trusting it on
anything real, same discipline that caught the last bug.
"""

import numpy as np
import pandas as pd

from volatility_model import dr_variance
from config import MIN_VOLUME_USD, MIN_DAYS_TO_RESOLUTION

N_BUCKETS = 10


def compute_vrp(df: pd.DataFrame, min_volume: float = MIN_VOLUME_USD,
                 min_days_to_resolution: float = MIN_DAYS_TO_RESOLUTION,
                 min_tau: float = None, bar_length: float = None) -> pd.DataFrame:
    """
    Adds h2 (DR-only structural prediction), realized_sq_move (next-bar
    actual squared price change), and vrp (their difference) to a filtered
    copy of the panel. One row per (market, bar) pair where a "next bar"
    exists and the standard liquidity/time filters pass.

    min_tau/bar_length MUST match your actual bar spacing in days (see
    volatility_model.py's module docstring). Defaults to daily bars
    (1.0) if not specified -- for real hourly Polymarket data
    (fidelity_minutes=60), pass min_tau=bar_length=1/24 explicitly, or
    h2 will be systematically inflated (confirmed: this exact omission
    produced an implausible ~27x "premium" on real data that was actually
    just this units mismatch, not a real finding -- see the caveat this
    module's own results triggered).
    """
    from volatility_model import DEFAULT_MIN_TAU, DEFAULT_BAR_LENGTH
    if min_tau is None:
        min_tau = DEFAULT_MIN_TAU
    if bar_length is None:
        bar_length = DEFAULT_BAR_LENGTH

    df = df.sort_values(["market_id", "timestamp"]).copy()
    df["h2"] = dr_variance(df["price"].values, df["days_to_resolution"].values,
                            min_tau=min_tau, bar_length=bar_length)
    # realized_sq_move at time t = (price at t+1 - price at t)^2, i.e. the
    # NEXT bar's move -- this is what h2 at time t is actually predicting
    # (a one-step-AHEAD forecast), so shift(-1) not diff().
    df["_next_price"] = df.groupby("market_id")["price"].shift(-1)
    df["realized_sq_move"] = (df["_next_price"] - df["price"]) ** 2
    df = df.drop(columns=["_next_price"])

    tradeable = (df["volume"] >= min_volume) & (df["days_to_resolution"] >= min_days_to_resolution)
    df = df[tradeable & df["realized_sq_move"].notna()].copy()
    df["vrp"] = df["h2"] - df["realized_sq_move"]
    return df


def compute_vrp_with_garch(panel: pd.DataFrame, min_volume: float = MIN_VOLUME_USD,
                            min_days_to_resolution: float = MIN_DAYS_TO_RESOLUTION,
                            min_tau: float = None, bar_length: float = None,
                            train_frac: float = 0.65) -> pd.DataFrame:
    """
    Extends the VRP check to compare realized variance against BOTH the
    DR-only h2 AND the GARCH-combined h2_combined -- the natural next test
    once a DR-only VRP pattern looks statistically real but structurally
    weird (not the clean, monotonic, extremes-strengthening shape a genuine
    risk premium should have): if the pattern is really leftover DR-AS
    misspecification, the GARCH layer (fit specifically to absorb exactly
    that) should shrink or clean it up. If the pattern persists just as
    strongly under h2_combined, that's evidence it's NOT simple
    misspecification GARCH can fix, and is worth trusting more.

    Evaluated on the TEST set only, held out from GARCH's own fitting --
    comparing model fit on the same data used to fit it would be
    optimistic, not a fair test of whether GARCH actually generalizes.
    """
    from signals import add_signals, add_structural_signals
    from backtest import train_test_split_by_market

    from volatility_model import DEFAULT_MIN_TAU, DEFAULT_BAR_LENGTH
    if min_tau is None:
        min_tau = DEFAULT_MIN_TAU
    if bar_length is None:
        bar_length = DEFAULT_BAR_LENGTH

    panel = add_signals(panel, mom_lookback=5, rev_lookback=10)
    train_df, test_df = train_test_split_by_market(panel, train_frac)
    train_ids = set(train_df["market_id"])
    test_ids = set(test_df["market_id"])

    # FIX: previously this called add_structural_signals TWICE -- the first
    # call's output (and fitting work) was immediately discarded by the
    # second, and the second call dropped train_market_ids=train_ids
    # (leaking test data into the GARCH fit) and referenced a bare
    # BAR_LENGTH_DAYS global that only exists in the __main__ block below,
    # so calling this function from anywhere else raised NameError. Now
    # there's a single call using the local min_tau/bar_length and the
    # train-only market ids, matching the "test set only, held out from
    # GARCH's own fitting" contract described above.
    panel, fitted_K, garch_params, joint_garch_params = add_structural_signals(
        panel, train_market_ids=train_ids,
        struct_mom_lookback=5, struct_rev_lookback=10, spread_col="spread",
        fit_garch=True, fit_joint_garch=True,
        min_tau=min_tau, bar_length=bar_length,
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

    panel = panel.sort_values(["market_id", "timestamp"]).copy()
    panel["_next_price"] = panel.groupby("market_id")["price"].shift(-1)
    panel["realized_sq_move"] = (panel["_next_price"] - panel["price"]) ** 2
    panel = panel.drop(columns=["_next_price"])

    tradeable = (panel["volume"] >= min_volume) & (panel["days_to_resolution"] >= min_days_to_resolution)
    test_only = tradeable & panel["realized_sq_move"].notna() & panel["market_id"].isin(test_ids)
    out = panel[test_only].copy()
    out["vrp_dr_only"] = out["h2"] - out["realized_sq_move"]
    if "h2_combined" in out.columns:
        out["vrp_garch"] = out["h2_combined"] - out["realized_sq_move"]
    return out


def compare_dr_vs_garch_table(df_with_both: pd.DataFrame, n_buckets: int = N_BUCKETS,
                               bucket_col: str = "price") -> pd.DataFrame:
    """Side-by-side comparison: does the GARCH layer shrink the DR-only VRP pattern?"""
    df = df_with_both.copy()
    df["_bucket"] = pd.qcut(df[bucket_col], n_buckets, duplicates="drop")

    rows = []
    for bucket, g in df.groupby("_bucket", observed=True):
        n = len(g)
        row = {"bucket_range": f"{bucket.left:.3f}-{bucket.right:.3f}", "n": n}
        for col, label in [("vrp_dr_only", "dr"), ("vrp_garch", "garch")]:
            if col not in g.columns:
                continue
            mean_vrp = g[col].mean()
            std_vrp = g[col].std(ddof=1)
            se = std_vrp / np.sqrt(n) if n > 1 and std_vrp > 0 else np.nan
            t_stat = mean_vrp / se if se and se > 0 else 0.0
            row[f"{label}_vrp"] = mean_vrp
            row[f"{label}_t"] = t_stat
        rows.append(row)
    return pd.DataFrame(rows)


def print_comparison_table(table: pd.DataFrame, label: str = ""):
    print(f"\n{'=' * 90}")
    print(f"DR-ONLY vs GARCH-COMBINED VRP COMPARISON (test set only){f': {label}' if label else ''}")
    print(f"{'=' * 90}")
    print(f"{'bucket':<16}{'n':>7}{'DR-only VRP':>13}{'DR t-stat':>11}{'GARCH VRP':>13}{'GARCH t-stat':>13}")
    for _, r in table.iterrows():
        dr_sig = "*" if abs(r.get("dr_t", 0)) >= 2.0 else " "
        garch_sig = "*" if abs(r.get("garch_t", 0)) >= 2.0 else " "
        print(f"{r['bucket_range']:<16}{r['n']:>7.0f}{r.get('dr_vrp', float('nan')):>+13.5f}"
              f"{r.get('dr_t', float('nan')):>+10.2f}{dr_sig}{r.get('garch_vrp', float('nan')):>+13.5f}"
              f"{r.get('garch_t', float('nan')):>+10.2f}{garch_sig}")
    n_dr_sig = (table["dr_t"].abs() >= 2.0).sum() if "dr_t" in table.columns else 0
    n_garch_sig = (table["garch_t"].abs() >= 2.0).sum() if "garch_t" in table.columns else 0
    print(f"\nDR-only: {n_dr_sig}/{len(table)} significant.  GARCH-combined: {n_garch_sig}/{len(table)} significant.")
    if n_garch_sig < n_dr_sig:
        print("GARCH layer reduced the number of significant buckets -- consistent with the DR-only")
        print("pattern being (at least partly) leftover misspecification that GARCH successfully absorbs.")
    elif n_garch_sig >= n_dr_sig:
        print("GARCH layer did NOT reduce significant buckets -- the pattern is not simple leftover")
        print("misspecification GARCH can fix. Worth taking more seriously as a potential real effect,")
        print("or investigating further (category-specific, or a genuinely different model gap).")


def vrp_table(df_with_vrp: pd.DataFrame, n_buckets: int = N_BUCKETS,
              bucket_col: str = "price") -> pd.DataFrame:
    """
    Buckets by `bucket_col` (default: price level) and reports, per bucket:
    mean predicted h2, mean realized_sq_move, their gap (the VRP), and a
    one-sample t-test of whether that gap is significantly different from
    zero -- properly accounting for the actual variance of the (h2 -
    realized) differences within the bucket, not an assumed/plugged-in one.
    """
    df = df_with_vrp.copy()
    df["_bucket"] = pd.qcut(df[bucket_col], n_buckets, duplicates="drop")

    rows = []
    for bucket, g in df.groupby("_bucket", observed=True):
        n = len(g)
        mean_h2 = g["h2"].mean()
        mean_realized = g["realized_sq_move"].mean()
        mean_vrp = g["vrp"].mean()
        std_vrp = g["vrp"].std(ddof=1)
        se = std_vrp / np.sqrt(n) if n > 1 and std_vrp > 0 else np.nan
        t_stat = mean_vrp / se if se and se > 0 else 0.0
        rows.append({
            "bucket_range": f"{bucket.left:.3f}-{bucket.right:.3f}",
            "mean_h2_predicted": mean_h2, "mean_realized": mean_realized,
            "vrp": mean_vrp, "n": n, "t_stat": t_stat,
        })
    return pd.DataFrame(rows)


def print_vrp_table(table: pd.DataFrame, label: str = ""):
    print(f"\n{'=' * 82}")
    print(f"VOLATILITY RISK PREMIUM CHECK{f': {label}' if label else ''}")
    print(f"{'=' * 82}")
    print(f"{'bucket':<16}{'h2 predicted':>14}{'realized':>12}{'VRP':>10}{'n':>8}{'t-stat':>9}  significant?")
    for _, r in table.iterrows():
        sig = "  <-- SIGNIFICANT" if abs(r["t_stat"]) >= 2.0 else ""
        print(f"{r['bucket_range']:<16}{r['mean_h2_predicted']:>14.5f}{r['mean_realized']:>12.5f}"
              f"{r['vrp']:>+10.5f}{r['n']:>8.0f}{r['t_stat']:>+9.2f}{sig}")

    n_buckets = len(table)
    n_significant = (table["t_stat"].abs() >= 2.0).sum()
    print(f"\n{n_significant}/{n_buckets} buckets significant at |t|>2.0.")
    positive = (table["t_stat"] >= 2.0).sum()
    negative = (table["t_stat"] <= -2.0).sum()
    if positive > 0:
        print(f"{positive} bucket(s) show POSITIVE VRP (model overstates realized vol -- "
              f"classic risk-premium direction, 'sell the phantom risk').")
    if negative > 0:
        print(f"{negative} bucket(s) show NEGATIVE VRP (realized vol exceeds model prediction -- "
              f"either model misspecification or genuinely unusual activity).")
    if n_buckets > 0:
        from scipy.stats import norm
        alpha = 2 * (1 - norm.cdf(2.0))  # two-tailed, matches |t|>=2.0 usage here
        p_at_least_one_fp = 1 - (1 - alpha) ** n_buckets
        print(f"(Reminder: with {n_buckets} simultaneous buckets, ~{p_at_least_one_fp:.0%} chance of "
              f"at least one false positive from noise alone -- look for a coherent pattern, "
              f"not an isolated bucket.)")


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_garch = "--garch" in sys.argv
    cache_path = args[0] if len(args) > 0 else "data/real_panel_cache.parquet"
    bucket_by = args[1] if len(args) > 1 else "price"
    print(f"Loading {cache_path}...")
    panel = pd.read_parquet(cache_path)
    print(f"{panel['market_id'].nunique()} markets, {len(panel)} bars")

    # This project's real-data fetch defaults to fidelity_minutes=60 (hourly
    # bars) -- min_tau/bar_length=1/24 day matches that. If you fetched at a
    # different fidelity, change this to match (e.g. 1.0 for daily bars,
    # matching the synthetic demo's convention).
    BAR_LENGTH_DAYS = 1 / 24
    print(f"Using bar_length=min_tau={BAR_LENGTH_DAYS:.4f} days (hourly bars) -- "
          f"change this in the script if your data uses a different fidelity.")

    if use_garch:
        vrp_df = compute_vrp_with_garch(panel, min_tau=BAR_LENGTH_DAYS, bar_length=BAR_LENGTH_DAYS)
        print(f"{len(vrp_df)} TEST-SET (market, bar) observations.")
        table = compare_dr_vs_garch_table(vrp_df, bucket_col=bucket_by)
        print_comparison_table(table, label=f"{cache_path} (bucketed by {bucket_by})")
    else:
        vrp_df = compute_vrp(panel, min_tau=BAR_LENGTH_DAYS, bar_length=BAR_LENGTH_DAYS)
        print(f"{len(vrp_df)} (market, bar) observations with a valid next-bar move to test against.")
        table = vrp_table(vrp_df, bucket_col=bucket_by)
        print_vrp_table(table, label=f"{cache_path} (bucketed by {bucket_by})")