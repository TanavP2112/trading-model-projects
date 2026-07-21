"""
Structural volatility model for binary prediction markets, implementing the
DR-AS specification from Xi, Moallemi, Pai & Wang, "Volatility in Prediction
Markets: A Structural Approach" (arXiv:2607.08199, 2026).

One-step conditional variance is decomposed into two additive channels:
    h^2 = p(1-p)/tau + K * nu(V) * s^2/4
"""

import numpy as np
import pandas as pd
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
    Fit AS scale parameter K via OLS on active-update bars:
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
    constrain_c_zero: bool = False
) -> Dict[str, float]:
    """
    Joint Quasi-MLE estimation of (K, omega, alpha, beta, c) on TRAIN data only.
    Filters out non-active updates during optimization for parameter stability.

    constrain_c_zero=True forces c=0 (and K=0), yielding a PLAIN GARCH(1,1)
    baseline with NO structural term -- h^2 = omega + alpha*eps^2 + beta*h^2.
    This is the reviewer's 4th model ("GARCH"), the baseline that isolates
    whether the STRUCTURAL variables add value OVER generic volatility
    clustering. It shares this exact recursion/QMLE machinery, so plain-GARCH
    vs GARCH+DR-AS is a clean apples-to-apples comparison (the latter just
    frees c and K).
    """
    from scipy.optimize import minimize

    df_sorted = df.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    eps = df_sorted.groupby("market_id")["price"].diff().to_numpy()
    p = df_sorted["price"].to_numpy()
    tau = df_sorted["days_to_resolution"].to_numpy()
    
    volume = df_sorted["volume"].to_numpy() if (spread_col and spread_col in df_sorted.columns) else None
    spread = df_sorted[spread_col].to_numpy() if (spread_col and spread_col in df_sorted.columns) else None
    boundaries = _market_boundaries(df_sorted["market_id"].to_numpy())

    # Initial parameter guess
    x0 = np.array([0.0, 1e-6, 0.05, 0.80, 1.0])
    args = (eps, p, tau, volume, spread, boundaries, min_tau, bar_length)
    bounds = [(0, None), (0, None), (0, 0.97), (0, 0.97), (0, None)]

    if constrain_c_zero:
        # Pin both the structural scale c (index 4) and the AS scale K (index 0)
        # to zero -> pure GARCH(1,1) with no prediction-market structure at all.
        x0[0] = 0.0
        x0[4] = 0.0
        bounds[0] = (0, 0)
        bounds[4] = (0, 0)

    res = minimize(
        _neg_log_likelihood, 
        x0, 
        args=args, 
        method="L-BFGS-B", 
        bounds=bounds,
        options={"maxiter": 200}
    )
    
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


def _market_boundaries(market_ids: np.ndarray) -> List[Tuple[int, int]]:
    boundaries = []
    start = 0
    n = len(market_ids)
    for i in range(1, n + 1):
        if i == n or market_ids[i] != market_ids[start]:
            boundaries.append((start, i))
            start = i
    return boundaries


def _joint_h2_recursion(
    eps: np.ndarray, 
    b: np.ndarray, 
    omega: float, 
    alpha: float,
    beta: float, 
    c: float, 
    boundaries: List[Tuple[int, int]],
    h2_ceiling: Optional[float] = None
) -> np.ndarray:
    n = len(eps)
    h2 = np.empty(n)
    if h2_ceiling is None:
        h2_ceiling = 25.0 * max(np.mean(b[np.isfinite(b)]), 1e-12)
        
    for start, end in boundaries:
        h2[start] = min(max(b[start], 1e-12), h2_ceiling)
        for t in range(start + 1, end):
            prev_eps = eps[t - 1]
            prev_eps = 0.0 if not np.isfinite(prev_eps) else prev_eps
            raw = omega + alpha * (prev_eps ** 2) + beta * h2[t - 1] + c * max(b[t - 1], 0.0)
            h2[t] = min(max(raw, 1e-12), h2_ceiling)
    return h2


def _neg_log_likelihood(
    params: np.ndarray, 
    eps: np.ndarray, 
    p: np.ndarray, 
    tau: np.ndarray,
    volume: Optional[np.ndarray], 
    spread: Optional[np.ndarray], 
    boundaries: List[Tuple[int, int]],
    min_tau: float, 
    bar_length: float
) -> float:
    K, omega, alpha, beta, c = params
    if K < 0 or omega < 0 or alpha < 0 or beta < 0 or c < 0 or (alpha + beta) >= 0.98:
        return 1e10
        
    b = structural_h2(p, tau, volume=volume, spread=spread, K=K, min_tau=min_tau, bar_length=bar_length)
    h2 = _joint_h2_recursion(eps, b, omega, alpha, beta, c, boundaries)
    h2 = np.clip(h2, 1e-12, None)
    
    valid = np.isfinite(eps) & np.isfinite(h2) & (eps != 0)  # Filter inactive updates
    if not np.any(valid):
        return 1e10
        
    ll = -0.5 * np.sum(np.log(2 * np.pi * h2[valid]) + (eps[valid] ** 2) / h2[valid])
    return -ll if np.isfinite(ll) else 1e10