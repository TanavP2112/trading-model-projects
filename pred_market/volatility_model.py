"""
Structural volatility model for binary prediction markets, implementing the
DR-AS specification from Xi, Moallemi, Pai & Wang, "Volatility in Prediction
Markets: A Structural Approach" (arXiv:2607.08199, 2026).

One-step conditional variance is decomposed into two additive channels:

    h^2 = p(1-p)/tau        <- Wright-Fisher DEADLINE-RESOLUTION (DR) channel
        + K * nu(V) * s^2/4 <- Glosten-Milgrom ADVERSE-SELECTION (AS) channel

DR channel (zero free parameters): a martingale on [0,1] that is forced to
resolve to {0,1} exactly at the deadline. p(1-p) is remaining binary
uncertainty; tau is time-to-resolution. This term alone peaks at p=0.5 and
mechanically explodes as tau -> 0 -- both are real structural features of
a binary market approaching settlement, not noise to be filtered out with
an arbitrary cutoff.

AS channel (one free parameter K, fit empirically): order-flow-driven
adverse-selection variance, scaling with the squared bid-ask spread and a
concave function of trading volume (more informed flow => wider realized
moves, but with diminishing returns to raw volume).
"""

import numpy as np
import pandas as pd

EPS_P = 1e-4
DEFAULT_MIN_TAU = 1.0   # floor for time-to-resolution, in the SAME units as your bar spacing.
# IMPORTANT: this must be >= your bar length (e.g. 1.0 for daily bars, 1/24 for
# hourly bars measured in days), NOT a near-zero epsilon. The paper's own
# boundary identity (their Appendix, Eq. 3) proves Var[p_T | F_t] -> p(1-p)
# exactly as tau -> T -- i.e. the correct limiting variance at expiration is
# BOUNDED, not infinite. Flooring tau at a near-zero epsilon instead of one
# bar-length makes h^2 = p(1-p)/tau explode on every market's terminal bar,
# which is a units bug, not a property of the model: with tau floored at the
# bar length Delta, h^2 = p(1-p)/Delta * Delta = p(1-p) exactly at expiration,
# recovering the correct bounded limit.


def dr_variance(p: np.ndarray, tau: np.ndarray, min_tau: float = DEFAULT_MIN_TAU) -> np.ndarray:
    """
    Wright-Fisher deadline-resolution variance component: p(1-p)/tau.

    p: price (probability), array-like in [0,1]
    tau: time remaining to resolution, in the SAME TIME UNITS as the bar
         frequency of your series (e.g. if bars are daily, tau is in days;
         this project's panels already carry `days_to_resolution` in days).
    min_tau: floor for tau, in those same units -- set this to your bar
             length (see module-level note above), not a tiny epsilon.
    """
    p = np.clip(np.asarray(p, dtype=float), EPS_P, 1 - EPS_P)
    tau = np.clip(np.asarray(tau, dtype=float), min_tau, None)
    return p * (1 - p) / tau


def nu(volume: np.ndarray) -> np.ndarray:
    """Concave activity-scaling function. See module docstring caveat above."""
    volume = np.clip(np.asarray(volume, dtype=float), 0, None)
    return np.log1p(volume)


def as_variance(volume: np.ndarray, spread: np.ndarray, K: float) -> np.ndarray:
    """Glosten-Milgrom adverse-selection variance component: K * nu(V) * s^2/4."""
    spread = np.asarray(spread, dtype=float)
    return K * nu(volume) * (spread ** 2) / 4.0


def fit_K(realized_sq_moves: np.ndarray, p: np.ndarray, tau: np.ndarray,
          volume: np.ndarray, spread: np.ndarray) -> float:
    """
    Fit the single AS scale parameter K via OLS, following the paper's
    variance-decomposition logic: E[(Δp)^2] = h^2 = DR + K*AS_unit, so

        (Δp)^2 - DR  ≈  K * nu(V) * s^2/4  + noise

    This regresses the DR-residual on the AS unit term with NO intercept
    (forcing the DR coefficient to exactly 1, as the model specifies) and
    returns the fitted slope K, clipped to be non-negative (K >= 0 is a
    model requirement -- it's a variance scale, not a signed effect).
    """
    dr = dr_variance(p, tau)
    residual = realized_sq_moves - dr
    as_unit = nu(volume) * (np.asarray(spread, dtype=float) ** 2) / 4.0
    # Simple OLS through the origin: K = sum(x*y) / sum(x*x)
    denom = np.sum(as_unit ** 2)
    if denom <= 0:
        return 0.0
    K = float(np.sum(as_unit * residual) / denom)
    return max(K, 0.0)


def structural_h2(p, tau, volume=None, spread=None, K: float = 0.0) -> np.ndarray:
    """
    Full one-step conditional variance forecast h^2. Falls back to DR-only
    (K effectively 0, or volume/spread not supplied) when spread data isn't
    available -- see module docstring.
    """
    h2 = dr_variance(p, tau)
    if volume is not None and spread is not None and K > 0:
        h2 = h2 + as_variance(volume, spread, K)
    return h2


def add_structural_vol(df: pd.DataFrame, K: float = 0.0,
                        volume_col: str = "volume", spread_col: str | None = None) -> pd.DataFrame:
    df = df.copy()
    volume = df[volume_col].values if (spread_col is not None and volume_col in df.columns) else None
    spread = df[spread_col].values if (spread_col is not None and spread_col in df.columns) else None
    df["h2"] = structural_h2(df["price"].values, df["days_to_resolution"].values,
                              volume=volume, spread=spread, K=K)
    df["h"] = np.sqrt(df["h2"].clip(lower=0))
    return df


def fetch_spread_series_note():
    """
    Not a runnable fetcher -- a pointer. To pull REAL bid-ask spreads and
    enable the full DR-AS model (not just the DR-only fallback), use the
    official SDK inside your own data_fetcher pull, per-token, e.g.:

        with PublicClient() as client:
            spread = client.get_spread(token_id=token_id)   # {'spread': Decimal(...)}

    or client.get_order_book(token_id=...) for the full book if you want
    to compute your own spread/mid at each timestamp instead of relying on
    Gamma's snapshot fields. This adds one extra API call per market beyond
    what build_market_panel_sdk() already does, which is why it's opt-in
    rather than default in this project.
    """
    pass
