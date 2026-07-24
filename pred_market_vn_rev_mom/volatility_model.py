"""
One-step conditional variance is decomposed into two additive channels:
    h^2 = p(1-p)/tau + K * nu(V) * s^2/4
"""

import math

import numpy as np
import pandas as pd
from numba import njit
from typing import Optional, Dict, Tuple, List

EPS_P = 1e-4
DEFAULT_MIN_TAU = 1.0 / 24.0    # Default floor: 1 hour (in days)
DEFAULT_BAR_LENGTH = 1.0 / 24.0 # Default bar spacing: 1 hour (in days)


def dr_variance(
    p: np.ndarray, 
    tau: np.ndarray, 
    min_tau: float = DEFAULT_MIN_TAU,
    bar_length: float = DEFAULT_BAR_LENGTH
) -> np.ndarray:
    """
    Wright-Fisher deadline-resolution variance component: [p(1-p) / tau] * bar_length.
    """
    p = np.clip(np.asarray(p, dtype=float), EPS_P, 1.0 - EPS_P)
    tau = np.clip(np.asarray(tau, dtype=float), min_tau, None)
    return (p * (1.0 - p) / tau) * bar_length


def nu(volume: np.ndarray) -> np.ndarray:
    """Concave activity-scaling function: log1p(V)."""
    volume = np.clip(np.asarray(volume, dtype=float), 0, None)
    return np.log1p(volume)


def as_variance(
    volume: np.ndarray, 
    spread: np.ndarray, 
    K: float,
    min_spread: float = 0.01
) -> np.ndarray:
    """
    Glosten-Milgrom adverse-selection variance component: K * nu(V) * s^2/4.
    Applies a minimum spread floor (1 cent) to avoid zero-variance bugs on tight books.
    """
    spread = np.clip(np.asarray(spread, dtype=float), min_spread, None)
    return K * nu(volume) * (spread ** 2) / 4.0


def fit_K(
    realized_moves: np.ndarray, 
    p: np.ndarray, 
    tau: np.ndarray,
    volume: np.ndarray, 
    spread: np.ndarray,
    active_mask: Optional[np.ndarray] = None,
    min_tau: float = DEFAULT_MIN_TAU, 
    bar_length: float = DEFAULT_BAR_LENGTH
) -> float:
    """
    Fit AS scale parameter K via a mle approach on active-update bars:
        (Δp)^2 - DR ≈ K * [nu(V) * s^2/4]
    """
    from scipy.optimize import minimize
    
    dr = dr_variance(p, tau, min_tau=min_tau, bar_length=bar_length)
    spread_clean = np.clip(np.asarray(spread, dtype=float), 0.01, None)
    as_unit = nu(volume) * (spread_clean ** 2) / 4.0

    if active_mask is not None:
        realized_moves = realized_moves[active_mask]
        dr = dr[active_mask]
        as_unit = as_unit[active_mask]

    eps_sq = realized_moves ** 2

    # Negative log likelihood loss
    def nll(k_val):
        h2 = np.maximum(dr + k_val[0] * as_unit, 1e-8)
        return 0.5 * np.sum(np.log(2 * np.pi * h2) + eps_sq / h2)

    res = minimize(nll, x0=[0.1], bounds=[(1e-5, None)], method="L-BFGS-B")
    return float(res.x[0]) if res.success else 0.0


def structural_h2(
    p: np.ndarray, 
    tau: np.ndarray, 
    volume: Optional[np.ndarray] = None, 
    spread: Optional[np.ndarray] = None, 
    K: float = 0.0,
    min_tau: float = DEFAULT_MIN_TAU, 
    bar_length: float = DEFAULT_BAR_LENGTH
) -> np.ndarray:
    """
    Full structural conditional variance forecast h^2 = DR + K*AS.
    Falls back to DR-only if volume/spread are omitted or K=0.
    """
    h2 = dr_variance(p, tau, min_tau=min_tau, bar_length=bar_length)
    if volume is not None and spread is not None and K > 0:
        h2 = h2 + as_variance(volume, spread, K)
    return h2


def fit_garch_dr_as_joint(
    df: pd.DataFrame,
    spread_col: Optional[str] = None,
    min_tau: float = DEFAULT_MIN_TAU,
    bar_length: float = DEFAULT_BAR_LENGTH,
    constrain_c_zero: bool = False,
) -> Dict[str, float]:
    """
    Joint QMLE estimation of the DR-AS + GARCH model on TRAIN data only.
    Filters out non-active updates during optimization for parameter stability.

    Uses the Gaussian working likelihood. Under Bollerslev-Wooldridge (1992),
    the parameter estimates are consistent regardless of the true innovation
    distribution.

    constrain_c_zero : bool
        Setting this to True forces c=0 and K=0, yielding plain GARCH(1,1) -- useful
        for plain GARCH test.
    """
    from scipy.optimize import minimize

    df_sorted = df.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    eps = df_sorted.groupby("market_id")["price"].diff().to_numpy()
    p = df_sorted["price"].to_numpy()
    tau = df_sorted["days_to_resolution"].to_numpy()

    volume = df_sorted["volume"].to_numpy() if (spread_col and spread_col in df_sorted.columns) else None
    spread = df_sorted[spread_col].to_numpy() if (spread_col and spread_col in df_sorted.columns) else None
    boundaries = _market_boundaries(df_sorted["market_id"].to_numpy())

    args = (eps, p, tau, volume, spread, boundaries, min_tau, bar_length)

    x0 = np.array([0.0, 1e-6, 0.05, 0.80, 1.0])
    bounds = [(0, None), (0, None), (0, 0.97), (0, 0.97), (0, None)]
    if constrain_c_zero:
        x0[0] = 0.0
        x0[4] = 0.0
        bounds[0] = (0, 0)
        bounds[4] = (0, 0)
    res = minimize(_neg_log_likelihood, x0, args=args,
                   method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 200})
    K, omega, alpha, beta, c = res.x

    return {
        "K": float(K),
        "omega": float(omega),
        "alpha": float(alpha),
        "beta": float(beta),
        "c": float(c),
        "persistence": float(alpha + beta),
        "success": bool(res.success),
        "neg_log_likelihood": float(res.fun),
    }

def garch_dr_as_h2(
    df: pd.DataFrame, 
    params: dict, 
    spread_col: str | None = None,
    min_tau: float = DEFAULT_MIN_TAU, 
    bar_length: float = DEFAULT_BAR_LENGTH
) -> pd.Series:
    """
    Applies frozen joint parameters (fit on train only) to compute h2 via 
    the joint additive recursion on full or test DataFrames.
    """
    df_sorted = df.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    eps = df_sorted.groupby("market_id")["price"].diff().to_numpy()
    p = df_sorted["price"].to_numpy()
    tau = df_sorted["days_to_resolution"].to_numpy()
    
    volume = df_sorted["volume"].to_numpy() if spread_col and spread_col in df_sorted.columns else None
    spread = df_sorted[spread_col].to_numpy() if spread_col and spread_col in df_sorted.columns else None
    boundaries = _market_boundaries(df_sorted["market_id"].to_numpy())

    b = structural_h2(
        p, tau, volume=volume, spread=spread, K=params["K"],
        min_tau=min_tau, bar_length=bar_length
    )
    h2 = _joint_h2_recursion(
        eps, b, params["omega"], params["alpha"], params["beta"],
        params["c"], boundaries
    )
    
    result = pd.Series(h2, index=df_sorted.index)
    return result.reindex(df.index) if not df.index.equals(df_sorted.index) else result


def _market_boundaries(market_ids: np.ndarray) -> np.ndarray:
    """
    Returns an (n_markets, 2) int64 array of [start, end) index pairs per
    market.
    """
    ranges = []
    start = 0
    n = len(market_ids)
    for i in range(1, n + 1):
        if i == n or market_ids[i] != market_ids[start]:
            ranges.append((start, i))
            start = i
    return np.array(ranges, dtype=np.int64) if ranges else np.zeros((0, 2), dtype=np.int64)


def _joint_h2_recursion(
    eps: np.ndarray,
    b: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
    c: float,
    boundaries: np.ndarray,
    h2_ceiling: Optional[float] = None,
) -> np.ndarray:
    if h2_ceiling is None:
        b_finite = b[np.isfinite(b)]
        h2_ceiling = 25.0 * max(float(np.mean(b_finite)) if b_finite.size else 1e-12, 1e-12)
    return _joint_h2_recursion_kernel(eps, b, float(omega), float(alpha),
                                      float(beta), float(c), boundaries,
                                      float(h2_ceiling))


@njit(cache=True)
def _joint_h2_recursion_kernel(
    eps: np.ndarray,
    b: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
    c: float,
    boundaries: np.ndarray,
    h2_ceiling: float,
) -> np.ndarray:
    n = eps.shape[0]
    h2 = np.empty(n)
    n_markets = boundaries.shape[0]
    for m in range(n_markets):
        start = boundaries[m, 0]
        end = boundaries[m, 1]
        b0 = b[start] if b[start] > 1e-12 else 1e-12
        if b0 > h2_ceiling:
            b0 = h2_ceiling
        h2[start] = b0
        for t in range(start + 1, end):
            prev_eps = eps[t - 1]
            if prev_eps != prev_eps:  # NaN check
                prev_eps = 0.0
            b_prev = b[t - 1]
            if b_prev < 0.0:
                b_prev = 0.0
            raw = omega + alpha * (prev_eps * prev_eps) + beta * h2[t - 1] + c * b_prev
            if raw < 1e-12:
                raw = 1e-12
            elif raw > h2_ceiling:
                raw = h2_ceiling
            h2[t] = raw
    return h2


def _neg_log_likelihood(
    params: np.ndarray,
    eps: np.ndarray,
    p: np.ndarray,
    tau: np.ndarray,
    volume: Optional[np.ndarray],
    spread: Optional[np.ndarray],
    boundaries: np.ndarray,
    min_tau: float,
    bar_length: float,
) -> float:
    K, omega, alpha, beta, c = params
    if K < 0 or omega < 0 or alpha < 0 or beta < 0 or c < 0 or (alpha + beta) >= 0.98:
        return 1e10

    b = structural_h2(p, tau, volume=volume, spread=spread, K=K,
                       min_tau=min_tau, bar_length=bar_length)
    h2 = _joint_h2_recursion(eps, b, omega, alpha, beta, c, boundaries)
    val = _gauss_nll_kernel(eps, h2)
    return val if np.isfinite(val) else 1e10


@njit(cache=True)
def _gauss_nll_kernel(eps: np.ndarray, h2: np.ndarray) -> float:
    """Sum Gaussian negative log-likelihood over active bars."""
    n = eps.shape[0]
    total = 0.0
    log_two_pi = math.log(2.0 * math.pi)
    for i in range(n):
        e = eps[i]
        # NaN check + inactive-bar filter
        if e != e or e == 0.0:
            continue
        h2i = h2[i]
        if h2i < 1e-12:
            h2i = 1e-12
        total += 0.5 * (log_two_pi + math.log(h2i) + (e * e) / h2i)
    return total