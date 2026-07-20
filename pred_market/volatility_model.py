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

IMPORTANT — what's exact vs. approximated here:
  - The DR term (p(1-p)/tau) is exact, parameter-free, and requires only
    data we already have in the panel (price, days_to_resolution).
  - The AS term's functional FORM (K * nu(V) * s^2/4) is exact per the
    paper's derivation. The specific concave nu(.) they found strongest
    ("concave volume scaling") was not fully specified in the portion of
    the paper reviewed here, so this module uses log1p(V) as a standard,
    defensible concave choice -- treat this as an engineering choice, not
    a claim of reproducing their exact fitted functional form.
  - K is fit here via simple OLS on your own data (regressing realized
    squared price moves, net of the DR term, on nu(V)*s^2/4). The paper
    fits K via a monthly expanding-window procedure on ~880k Kalshi
    contract-hours; a single-shot OLS on a smaller panel is a much
    lower-power version of the same idea.
  - Real bid-ask spread (s) is not in this project's default data panel
    (data_fetcher.py currently pulls the price series only). If you don't
    supply a spread column, this module falls back to DR-ONLY, which the
    paper itself reports "already improves substantially" on generic
    GARCH benchmarks -- so the fallback is still a real, defensible model,
    just missing the second channel. See fetch_spread_series() below for
    how to pull real spreads via the official SDK's get_spread()/
    get_order_book() methods if you want the full DR-AS model.
"""

import numpy as np
import pandas as pd

EPS_P = 1e-4
DEFAULT_MIN_TAU = 1.0   # floor for time-to-resolution, in the SAME units as your bar spacing.
DEFAULT_BAR_LENGTH = 1.0   # Delta: your actual bar spacing, in the SAME units as tau (days).
# IMPORTANT -- a real bug found by comparing predicted vs realized variance on
# real hourly Polymarket data: the paper's actual one-step variance formula is
#     Var(Delta_p_i | F_ti) ~= [p_ti(1-p_ti)/tau_i] * Delta_i
# where Delta_i is the BAR LENGTH. An earlier version of this module omitted
# the explicit Delta factor entirely (implicitly always assuming Delta=1 day),
# which is fine for the synthetic demo's daily bars but silently wrong for
# real data fetched at any other fidelity. Confirmed via direct measurement:
# on real hourly (fidelity_minutes=60, Delta=1/24 day) sports data, h^2
# averaged ~27x larger than realized squared price moves across every price
# bucket uniformly -- almost exactly the predicted 24x mismatch from treating
# hourly bars as daily ones, not a genuine ~2700% volatility risk premium
# (which would be economically implausible on its face; real documented risk
# premia in options markets are more like 10-20% overstatement, not 30-50x).
#
# SET bar_length TO MATCH YOUR ACTUAL DATA: 1.0 for daily bars (the
# synthetic demo's convention), 1/24 for hourly bars (fidelity_minutes=60,
# this project's real-data default), 1/1440 for minute bars, etc. Getting
# this wrong doesn't just shift results by a constant -- since bar_length
# and min_tau BOTH need to match your actual bar spacing, an unadjusted
# default will systematically inflate h^2 (and therefore the DENOMINATOR of
# every structural signal in signals.py), making genuine signals harder to
# detect, not easier -- this is a false-negative risk, not just a cosmetic
# scaling issue.


def dr_variance(p: np.ndarray, tau: np.ndarray, min_tau: float = DEFAULT_MIN_TAU,
                 bar_length: float = DEFAULT_BAR_LENGTH) -> np.ndarray:
    """
    Wright-Fisher deadline-resolution variance component: p(1-p)/tau * Delta.

    p: price (probability), array-like in [0,1]
    tau: time remaining to resolution, in the SAME TIME UNITS as bar_length
         (e.g. if bars are hourly, both tau and bar_length should be in
         days, with bar_length=1/24; this project's panels carry
         `days_to_resolution` in days).
    min_tau: floor for tau, in those same units -- should equal bar_length
             in the typical case (see module-level note above).
    bar_length: your actual bar spacing (Delta), in the same units as tau.
                DEFAULTS TO 1.0 (one day) -- override this explicitly for
                any data not on daily bars, or h^2 will be systematically
                wrong by the ratio of 1-day to your-actual-bar-length.
    """
    p = np.clip(np.asarray(p, dtype=float), EPS_P, 1 - EPS_P)
    tau = np.clip(np.asarray(tau, dtype=float), min_tau, None)
    return (p * (1 - p) / tau) * bar_length


def nu(volume: np.ndarray) -> np.ndarray:
    """Concave activity-scaling function. See module docstring caveat above."""
    volume = np.clip(np.asarray(volume, dtype=float), 0, None)
    return np.log1p(volume)


def as_variance(volume: np.ndarray, spread: np.ndarray, K: float) -> np.ndarray:
    """Glosten-Milgrom adverse-selection variance component: K * nu(V) * s^2/4."""
    spread = np.asarray(spread, dtype=float)
    return K * nu(volume) * (spread ** 2) / 4.0


def fit_K(realized_sq_moves: np.ndarray, p: np.ndarray, tau: np.ndarray,
          volume: np.ndarray, spread: np.ndarray,
          min_tau: float = DEFAULT_MIN_TAU, bar_length: float = DEFAULT_BAR_LENGTH) -> float:
    """
    Fit the single AS scale parameter K via OLS, following the paper's
    variance-decomposition logic: E[(Δp)^2] = h^2 = DR + K*AS_unit, so

        (Δp)^2 - DR  ≈  K * nu(V) * s^2/4  + noise

    This regresses the DR-residual on the AS unit term with NO intercept
    (forcing the DR coefficient to exactly 1, as the model specifies) and
    returns the fitted slope K, clipped to be non-negative (K >= 0 is a
    model requirement -- it's a variance scale, not a signed effect).
    """
    dr = dr_variance(p, tau, min_tau=min_tau, bar_length=bar_length)
    residual = realized_sq_moves - dr
    as_unit = nu(volume) * (np.asarray(spread, dtype=float) ** 2) / 4.0
    # Simple OLS through the origin: K = sum(x*y) / sum(x*x)
    denom = np.sum(as_unit ** 2)
    if denom <= 0:
        return 0.0
    K = float(np.sum(as_unit * residual) / denom)
    return max(K, 0.0)


def structural_h2(p, tau, volume=None, spread=None, K: float = 0.0,
                   min_tau: float = DEFAULT_MIN_TAU, bar_length: float = DEFAULT_BAR_LENGTH) -> np.ndarray:
    """
    Full one-step conditional variance forecast h^2. Falls back to DR-only
    (K effectively 0, or volume/spread not supplied) when spread data isn't
    available -- see module docstring.
    """
    h2 = dr_variance(p, tau, min_tau=min_tau, bar_length=bar_length)
    if volume is not None and spread is not None and K > 0:
        h2 = h2 + as_variance(volume, spread, K)
    return h2


def add_structural_vol(df: pd.DataFrame, K: float = 0.0,
                        volume_col: str = "volume", spread_col: str | None = None,
                        min_tau: float = DEFAULT_MIN_TAU, bar_length: float = DEFAULT_BAR_LENGTH) -> pd.DataFrame:
    """
    Adds columns:
      h2        -- one-step structural conditional variance forecast
      h         -- structural conditional standard deviation (sqrt of h2)
    to a panel that has 'price' and 'days_to_resolution' columns (and,
    optionally, a spread column for the full DR-AS model).

    min_tau/bar_length MUST match your actual bar spacing (see module-level
    note) -- defaults assume daily bars; override both for hourly/other
    fidelity data, e.g. min_tau=bar_length=1/24 for fidelity_minutes=60.
    """
    df = df.copy()
    volume = df[volume_col].values if (spread_col is not None and volume_col in df.columns) else None
    spread = df[spread_col].values if (spread_col is not None and spread_col in df.columns) else None
    df["h2"] = structural_h2(df["price"].values, df["days_to_resolution"].values,
                              volume=volume, spread=spread, K=K,
                              min_tau=min_tau, bar_length=bar_length)
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


# ===========================================================================
# CORRECT GARCH+DR-AS: additive, JOINTLY-estimated recursion, per the paper's
# confirmed equation (18) -- NOT the multiplicative two-stage approximation
# in fit_residual_garch()/garch_multiplier_per_market() above, which was
# built before the exact equation was available and is now known to differ
# from the paper's actual specification in two ways: (1) additive, not
# multiplicative, and (2) all parameters estimated jointly in one procedure,
# not DR-AS fit first with GARCH layered on afterward. Both functions above
# are kept for comparison rather than deleted, matching this project's
# running discipline of never silently replacing an approximation with a
# "better" one without being able to check the difference.
#
# The confirmed equation:
#     h_i^2 = omega + alpha*eps_{i-1}^2 + beta*h_{i-1}^2 + c*b_i,   c >= 0
# where b_i is the closed-form DR-AS structural prediction (structural_h2
# above, itself parameterized by K), eps_{i-1} is the RAW (not standardized)
# lagged price innovation, and (K, omega, alpha, beta, c) are ALL estimated
# together via Gaussian quasi-maximum-likelihood -- the paper explicitly
# states the dynamic fit does not first estimate DR-AS and then append a
# separate GARCH correction.
# ===========================================================================
def _market_boundaries(market_ids: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous (start, end) index ranges per market in a SORTED array."""
    boundaries = []
    start = 0
    n = len(market_ids)
    for i in range(1, n + 1):
        if i == n or market_ids[i] != market_ids[start]:
            boundaries.append((start, i))
            start = i
    return boundaries


def _joint_h2_recursion(eps: np.ndarray, b: np.ndarray, omega: float, alpha: float,
                         beta: float, c: float, boundaries: list[tuple[int, int]],
                         h2_ceiling: float = None) -> np.ndarray:
    """
    h_i^2 = omega + alpha*eps_{i-1}^2 + beta*h_{i-1}^2 + c*b_i, reset at each
    market boundary (initialized to the structural prediction b_i itself at
    the first bar of each market, since there's no lagged eps/h2 yet -- a
    reasonable, bounded starting value, not an arbitrary one).

    h2_ceiling: hard cap on h2, same defense-in-depth rationale as
    garch_multiplier_per_market's cap -- a fitted (alpha, beta) close to or
    at the stationarity boundary can still compound over long real markets
    (hundreds to thousands of hourly bars) well beyond what any short
    synthetic validation would reveal. Defaults to 25x the mean of b if not
    specified.
    """
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


def _neg_log_likelihood(params: np.ndarray, eps: np.ndarray, p: np.ndarray, tau: np.ndarray,
                         volume, spread, boundaries: list[tuple[int, int]],
                         min_tau: float, bar_length: float) -> float:
    K, omega, alpha, beta, c = params
    # Hard feasibility region: all non-negative (paper's own constraints),
    # plus a stationarity margin on the GARCH part (same MAX_PERSISTENCE
    # discipline as the multiplicative model above, for the same reason --
    # confirmed necessary, not theoretical caution).
    if K < 0 or omega < 0 or alpha < 0 or beta < 0 or c < 0 or (alpha + beta) >= 0.98:
        return 1e10
    b = structural_h2(p, tau, volume=volume, spread=spread, K=K, min_tau=min_tau, bar_length=bar_length)
    h2 = _joint_h2_recursion(eps, b, omega, alpha, beta, c, boundaries)
    h2 = np.clip(h2, 1e-12, None)
    valid = np.isfinite(eps) & np.isfinite(h2)
    ll = -0.5 * np.sum(np.log(2 * np.pi * h2[valid]) + (eps[valid] ** 2) / h2[valid])
    if not np.isfinite(ll):
        return 1e10
    return -ll


def fit_garch_dr_as_joint(df: pd.DataFrame, spread_col: str | None = None,
                           min_tau: float = None, bar_length: float = None) -> dict:
    """
    Joint Gaussian quasi-MLE fit of (K, omega, alpha, beta, c) on df -- pass
    TRAIN markets only, same discipline as every other fitted parameter in
    this project. Returns a dict of fitted params plus a 'success' flag.

    Uses scipy.optimize.minimize (L-BFGS-B) rather than the `arch` package,
    since this is a custom structural-baseline-plus-additive-GARCH hybrid,
    not a plain return-series GARCH the `arch` package is built for.
    """
    from scipy.optimize import minimize

    if min_tau is None:
        min_tau = DEFAULT_MIN_TAU
    if bar_length is None:
        bar_length = DEFAULT_BAR_LENGTH

    df = df.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    eps = df.groupby("market_id")["price"].diff().to_numpy()
    p = df["price"].to_numpy()
    tau = df["days_to_resolution"].to_numpy()
    volume = df["volume"].to_numpy() if spread_col and spread_col in df.columns else None
    spread = df[spread_col].to_numpy() if spread_col and spread_col in df.columns else None
    boundaries = _market_boundaries(df["market_id"].to_numpy())

    # Reasonable, bounded starting point: no AS/GARCH effect, small ARCH/GARCH
    # persistence, structural term weighted at 1 -- lets the optimizer find
    # its own way rather than starting at an edge of the feasible region.
    x0 = np.array([0.0, 1e-6, 0.05, 0.80, 1.0])
    args = (eps, p, tau, volume, spread, boundaries, min_tau, bar_length)
    bounds = [(0, None), (0, None), (0, 0.97), (0, 0.97), (0, None)]

    result = minimize(_neg_log_likelihood, x0, args=args, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 200})
    K, omega, alpha, beta, c = result.x
    return {
        "K": float(K), "omega": float(omega), "alpha": float(alpha),
        "beta": float(beta), "c": float(c), "persistence": float(alpha + beta),
        "success": bool(result.success), "neg_log_likelihood": float(result.fun),
    }


def garch_dr_as_h2(df: pd.DataFrame, params: dict, spread_col: str | None = None,
                    min_tau: float = None, bar_length: float = None) -> pd.Series:
    """
    Applies FROZEN params (fit on train only) to compute h2 via the joint
    additive recursion on the FULL df (train+test) -- a genuine one-step-
    ahead forecast at each point (uses only eps/h2 up to i-1), safe for
    out-of-sample application, same discipline as garch_multiplier_per_market.
    """
    if min_tau is None:
        min_tau = DEFAULT_MIN_TAU
    if bar_length is None:
        bar_length = DEFAULT_BAR_LENGTH

    df_sorted = df.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    eps = df_sorted.groupby("market_id")["price"].diff().to_numpy()
    p = df_sorted["price"].to_numpy()
    tau = df_sorted["days_to_resolution"].to_numpy()
    volume = df_sorted["volume"].to_numpy() if spread_col and spread_col in df_sorted.columns else None
    spread = df_sorted[spread_col].to_numpy() if spread_col and spread_col in df_sorted.columns else None
    boundaries = _market_boundaries(df_sorted["market_id"].to_numpy())

    b = structural_h2(p, tau, volume=volume, spread=spread, K=params["K"],
                       min_tau=min_tau, bar_length=bar_length)
    h2 = _joint_h2_recursion(eps, b, params["omega"], params["alpha"], params["beta"],
                              params["c"], boundaries)
    result = pd.Series(h2, index=df_sorted.index)
    return result.reindex(df.index) if not df.index.equals(df_sorted.index) else result


# ---------------------------------------------------------------------------
# GARCH-on-residuals layer: "GARCH+DR-AS" from the paper (their best overall
# specification). Key idea: standardize the realized price move by the
# structural h_i, z_i = epsilon_i / h_i. If the structural model fully
# explained the variance dynamics, z_i would behave like unit-variance iid
# noise. If real clustering remains ON TOP of the structural baseline, a
# GARCH(1,1) fit to z_i will pick it up, and its conditional-variance path
# becomes a MULTIPLICATIVE correction: h^2_combined = h^2_structural * g^2,
# where g^2 hovers around 1 by construction (z has ~unit variance if the
# structural model isn't systematically mis-scaled).
#
# THE PANEL WRINKLE (specific to this project, not generic GARCH advice):
# GARCH needs a continuous time series with real volatility clustering
# across consecutive observations. Our data is a PANEL of many short-lived
# markets (dozens of bars each), not one long series -- fitting a separate
# GARCH per market would be statistically hopeless (nowhere near enough
# observations per contract), and pooling every market's residuals into one
# undifferentiated sequence would treat cross-market boundaries as if they
# were real lag-1 relationships, which they aren't.
#
# The compromise used here, matching how the paper's own "pooled headline"
# result works: fit ONE set of (omega, alpha, beta) on the pooled residuals
# across all TRAIN markets (a reasonable simplification, not a claim of
# reproducing their more careful category-level treatment), but APPLY the
# recursive g^2_i forecast SEPARATELY WITHIN each market's own chronological
# sequence, resetting to the fitted unconditional variance at the start of
# every market. That reset matters: volatility clustering is a property of
# a single contract's own path, not something that should carry over from
# an unrelated prior market just because it happened to appear earlier in
# the pooled training data.
# ---------------------------------------------------------------------------
def fit_residual_garch(standardized_residuals: np.ndarray, dist: str = "t") -> dict:
    """
    Fits GARCH(1,1) on structural-model standardized residuals z_i = eps_i/h_i
    via the `arch` package. Returns fitted (omega, alpha, beta, uncond_var).

    dist="t" (Student-t, default) rather than "normal": a real bug found on
    real data, root-caused rather than just patched. A near-certain market
    (p close to 0 or 1) has a structurally TINY predicted variance -- if it
    then experiences a genuine surprise (a real upset, which does happen),
    the standardized residual z can be enormous (confirmed: a market at
    p=0.0001 upset to p=0.5 produces z~50) purely because the denominator
    is small, not because anything is wrong with the data. Even a handful
    of such points (3 out of 3000 in a direct test) is enough to drag a
    Gaussian MLE fit toward a pathological solution (confirmed: uncond_var
    inflated on real data by 46x and 516x in two separate runs). Switching
    to Student-t, which down-weights extreme points instead of squaring
    their outsized influence, fixed this directly in the same test (3/3000
    outliers: uncond_var 3.44 -> 1.04, correctly identifying nu~6.6 as the
    tail heaviness). The persistence cap and absolute ceiling elsewhere in
    this module are kept as defense-in-depth on top of this, not replaced
    by it -- no single safeguard should be trusted alone.

    NOTE: fit this on TRAIN data only and freeze the params before touching
    test data -- same discipline as fit_K() above. This function itself
    doesn't enforce that; the caller (signals.add_structural_signals) does.
    """
    from arch import arch_model
    z = np.asarray(standardized_residuals, dtype=float)
    z = z[np.isfinite(z)]
    if len(z) < 100:
        # Not enough residuals to fit a GARCH model reliably -- fall back to
        # "no extra clustering info", i.e. a flat multiplier of 1 everywhere.
        return {"omega": 0.0, "alpha": 0.0, "beta": 0.0, "uncond_var": 1.0, "nu": None, "fit_ok": False}

    # WINSORIZATION -- added after Student-t alone was confirmed insufficient
    # on real data (uncond_var stayed at ~46 even with nu correctly fit to
    # ~4.0, i.e. real heavy tails). Root cause: z=eps/h is a RATIO, and h can
    # be genuinely tiny near price boundaries (structurally correct -- a
    # near-certain market has near-zero predicted variance). If a real
    # surprise (a true upset) lands on exactly such a low-h observation, the
    # resulting z is enormous NOT because the overall distribution is
    # fat-tailed in a way Student-t's single shape parameter can absorb, but
    # because that ONE observation's denominator was pathologically small.
    # Confirmed via direct construction: a single such point can leave a
    # DIFFERENCE-based average (h2 vs realized, what the plain VRP check
    # uses) looking essentially balanced while a RATIO-based average (z^2,
    # what feeds uncond_var) is inflated by 10x+ from that one point alone.
    # Winsorizing clips the SYMPTOM regardless of cause -- whether it's
    # genuine fat tails or denominator heterogeneity, no single point can
    # dominate the likelihood after this.
    winsor_limit = np.percentile(np.abs(z), 99.0)
    winsor_limit = max(winsor_limit, 3.0)  # never clip tighter than 3 sigma even on very clean data
    n_winsorized = int(np.sum(np.abs(z) > winsor_limit))
    z = np.clip(z, -winsor_limit, winsor_limit)

    scale = 100.0  # rescale for optimizer numerical stability; unscaled below
    am = arch_model(z * scale, mean="Zero", vol="GARCH", p=1, q=1, dist=dist)
    try:
        res = am.fit(disp="off", show_warning=False)
    except Exception:
        return {"omega": 0.0, "alpha": 0.0, "beta": 0.0, "uncond_var": 1.0, "nu": None, "fit_ok": False}

    omega = float(res.params["omega"]) / (scale ** 2)
    alpha = float(res.params["alpha[1]"])
    beta = float(res.params["beta[1]"])
    nu = float(res.params["nu"]) if "nu" in res.params else None
    # NU FLOOR -- Student-t variance is only finite for nu>2; a fit landing
    # near that boundary (confirmed on real data: nu=2.05) is inherently
    # unstable regardless of what's driving it, since the distribution's own
    # variance is nearly undefined there. This isn't a fix for the root
    # cause (found to be a genuinely high rate of near-boundary-market
    # surprises in this data, not something a single distributional choice
    # fully absorbs -- see the joint additive model as the real fix for
    # that), just a last-resort sanity floor so a near-degenerate nu can't
    # silently produce a near-degenerate downstream forecast.
    if nu is not None and nu < 4.0:
        nu = 4.0
    persistence = alpha + beta

    # STATIONARITY SAFEGUARD -- a real bug found on real data: with no cap
    # here, a fit landing at or above persistence=1 (plausible on real,
    # heavier-tailed hourly data with far more bars per market than the
    # synthetic validation ever had -- hundreds to thousands vs ~20-45)
    # makes the recursive g^2 formula non-mean-reverting, and it can grow
    # essentially without bound over a long market's bar sequence. Confirmed
    # in practice: this produced h2_combined inflated ~100-270x over h2
    # alone on real data, uniformly across every price bucket -- not a
    # finding, a runaway recursion. Proportionally rescale (alpha, beta) to
    # cap persistence at a safety margin below 1 if the raw fit exceeds it,
    # preserving their relative ratio rather than clipping just one.
    MAX_PERSISTENCE = 0.98
    if persistence >= MAX_PERSISTENCE:
        shrink = MAX_PERSISTENCE / persistence
        alpha *= shrink
        beta *= shrink
        persistence = alpha + beta

    uncond_var = omega / (1 - persistence) if persistence < 1 else float(np.mean(z ** 2))
    return {"omega": omega, "alpha": alpha, "beta": beta, "uncond_var": uncond_var, "nu": nu,
            "fit_ok": True, "n_winsorized": n_winsorized, "winsor_limit": float(winsor_limit)}


def garch_multiplier_per_market(df: pd.DataFrame, resid_col: str, garch_params: dict,
                                 max_multiplier: float = 25.0) -> pd.Series:
    """
    Recursively computes the GARCH(1,1) conditional-variance multiplier g^2_i
    WITHIN each market's own chronological sequence (see module note above
    for why this is done per-market rather than pooled):

        g^2_i = omega + alpha * z_{i-1}^2 + beta * g^2_{i-1},  g^2 reset to
        uncond_var at the start of every market.

    This is a genuine one-step-ahead recursive forecast (uses only z up to
    i-1 at each step), so it's safe to apply on test data with FROZEN params.

    max_multiplier: ABSOLUTE hard cap on g^2 -- not relative to the fitted
    uncond_var. A real bug found on real data: the previous version capped
    g^2 at `uncond_var * 25`, which sounds like a safeguard but isn't one if
    uncond_var itself comes out large from the fit (confirmed: real,
    heavier-tailed hourly data produced this). Since z = eps/h is
    constructed to have ~unit variance if h is well-calibrated, g^2 SHOULD
    hover around 1 in a healthy fit -- an uncond_var far from 1 is itself a
    symptom something's off, not a number to scale the safety margin by.
    Both the reset value (uncond_var) AND the recursive ceiling are now
    capped at this same fixed, absolute value, so a bad uncond_var can't
    inflate the starting point either.
    """
    omega, alpha, beta = garch_params["omega"], garch_params["alpha"], garch_params["beta"]
    uncond_var = garch_params["uncond_var"]
    reset_value = min(uncond_var, max_multiplier)
    ceiling = max_multiplier

    df_sorted = df.sort_values(["market_id", "timestamp"])
    z_all = df_sorted[resid_col].to_numpy()
    market_ids = df_sorted["market_id"].to_numpy()

    # Explicit per-market recursion into a flat output array, rather than
    # groupby().apply() -- a real bug was found where apply() returns a
    # DataFrame instead of a Series in the single-market-group edge case
    # (a known pandas ambiguity when the applied function's return index
    # matches the group's own index), which would have silently produced
    # NaN downstream rather than a loud error. This form can't hit that
    # ambiguity regardless of how many distinct markets are present.
    out = np.empty(len(z_all))
    start = 0
    for i in range(1, len(market_ids) + 1):
        if i == len(market_ids) or market_ids[i] != market_ids[start]:
            out[start] = reset_value
            for t in range(start + 1, i):
                prev_z = z_all[t - 1]
                prev_z = 0.0 if not np.isfinite(prev_z) else prev_z
                raw = omega + alpha * (prev_z ** 2) + beta * out[t - 1]
                out[t] = min(raw, ceiling)
            start = i

    result = pd.Series(out, index=df_sorted.index)
    return result.reindex(df.index)