"""
Favorite-longshot bias / raw calibration check.

A fundamentally different hypothesis from everything else in this project:
not "does recent price movement predict future movement" (momentum/reversal),
but "does CURRENT price already reflect true probability, full stop." This
is one of the most robust findings in betting markets generally -- longshots
(low prices) tend to be systematically OVERpriced relative to their true win
probability, and favorites (high prices) tend to be systematically
UNDERpriced. Worth testing directly on Polymarket rather than assuming it
transfers from other betting markets.

METHODOLOGY NOTE (read before trusting results): this uses ONE observation
per market -- the first bar that passes the same liquidity/time-to-
resolution filters used everywhere else in this project -- not every bar.
Pooling all bars from every market's full history would badly contaminate
this test: a market's price mechanically converges toward 0 or 1 as
resolution approaches (that's the whole point of the DR structural term),
so bars close to resolution are "accurate" almost by construction. That's
not mispricing, it's just time passing. A single early snapshot per market
avoids this and gives a genuine test of whether STANDING prices (analogous
to opening odds) reflect true probability.

For each price decile, we test whether the realized win rate is
significantly different from the average price in that bucket, using the
same z-test-on-edge-vs-standard-error logic already used throughout
backtest.py -- consistent statistical standard across the whole project.
"""

import numpy as np
import pandas as pd

from config import MIN_VOLUME_USD, MIN_DAYS_TO_RESOLUTION

N_BUCKETS = 10


def build_snapshot_candidates(df: pd.DataFrame, min_volume: float = MIN_VOLUME_USD,
                               min_days_to_resolution: float = MIN_DAYS_TO_RESOLUTION) -> pd.DataFrame:
    """One row per market: the first bar passing the standard liquidity/time filters."""
    df = df.copy()
    tradeable = (df["volume"] >= min_volume) & (df["days_to_resolution"] >= min_days_to_resolution)
    cand = df[tradeable].sort_values(["market_id", "timestamp"])
    return cand.groupby("market_id", as_index=False).first()


def calibration_table(cand: pd.DataFrame, n_buckets: int = N_BUCKETS) -> pd.DataFrame:
    """
    Buckets snapshot candidates by price into deciles, and for each bucket
    reports: average price (what the market implied), realized win rate
    (what actually happened), and a z-test of whether the gap between them
    is distinguishable from zero given the sample size.

    z_stat here uses the SAME formula as backtest.py's significance guard:
    (win_rate - avg_price) / sqrt(win_rate*(1-win_rate)/n). Positive z means
    the bucket won MORE than its price implied (that price level is
    underpriced -- buying YES there would have had positive edge). Negative
    z means it won LESS than implied (overpriced -- buying NO, i.e. fading
    it, would have had positive edge).
    """
    cand = cand.copy()
    cand["price_bucket"] = pd.qcut(cand["price"], n_buckets, duplicates="drop")

    rows = []
    for bucket, g in cand.groupby("price_bucket", observed=True):
        avg_price = g["price"].mean()
        win_rate = g["outcome"].mean()
        n = len(g)
        # SCORE-test convention: standard error uses the HYPOTHESIZED rate
        # (avg_price -- "is this bucket priced correctly") in the
        # denominator, not the observed win_rate. This matters a lot near
        # the boundaries: a Wald-style SE using the observed rate collapses
        # toward zero as win_rate -> 0 or 1 (x(1-x) is minimized there),
        # which can blow up z-stats to nonsensical values (confirmed: a
        # trivial-looking gap of 0.04 at a near-zero win rate produced
        # z=-259 with the wrong formula, vs a sane z=-1.3 with this one).
        se = np.sqrt(max(avg_price * (1 - avg_price), 1e-6) / max(n, 1))
        z = (win_rate - avg_price) / se if se > 0 else 0.0
        rows.append({
            "price_range": f"{bucket.left:.2f}-{bucket.right:.2f}",
            "avg_price": avg_price, "win_rate": win_rate, "n": n,
            "gap": win_rate - avg_price, "z_stat": z,
        })
    return pd.DataFrame(rows)


def print_calibration_table(table: pd.DataFrame, label: str = ""):
    print(f"\n{'=' * 78}")
    print(f"CALIBRATION CHECK{f': {label}' if label else ''}")
    print(f"{'=' * 78}")
    print(f"{'price range':<14}{'avg price':>11}{'win rate':>11}{'n':>7}{'gap':>9}{'z-stat':>9}  significant?")
    for _, r in table.iterrows():
        sig = "  <-- SIGNIFICANT" if abs(r["z_stat"]) >= 2.0 else ""
        print(f"{r['price_range']:<14}{r['avg_price']:>11.3f}{r['win_rate']:>11.3f}"
              f"{r['n']:>7.0f}{r['gap']:>+9.3f}{r['z_stat']:>+9.2f}{sig}")

    n_buckets = len(table)
    n_significant = (table["z_stat"].abs() >= 2.0).sum()
    print(f"\n{n_significant}/{n_buckets} buckets significant at z>2.0.")
    if n_buckets > 0:
        from scipy.stats import norm
        alpha = 1 - norm.cdf(2.0)
        p_at_least_one_fp = 1 - (1 - alpha) ** n_buckets
        print(f"(Reminder: with {n_buckets} simultaneous buckets tested, there's a "
              f"{p_at_least_one_fp:.0%} chance of at least one false positive from pure "
              f"noise alone -- a single significant bucket isn't enough on its own; look "
              f"for a coherent PATTERN across adjacent buckets, e.g. a monotonic gap that "
              f"grows toward the price extremes, which is the classic favorite-longshot shape.)")


if __name__ == "__main__":
    import sys
    cache_path = sys.argv[1] if len(sys.argv) > 1 else "data/real_panel_cache.parquet"
    print(f"Loading {cache_path}...")
    panel = pd.read_parquet(cache_path)
    print(f"{panel['market_id'].nunique()} markets, {len(panel)} bars, "
          f"categories: {panel['category'].unique().tolist()}")

    cand = build_snapshot_candidates(panel)
    print(f"\n{len(cand)} markets pass liquidity/time-to-resolution filters "
          f"(one snapshot each, out of {panel['market_id'].nunique()} total).")

    table = calibration_table(cand)
    print_calibration_table(table, label=cache_path)
