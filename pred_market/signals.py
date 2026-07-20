"""
Signal construction for prediction-market momentum/reversal research.

Key design decision: we compute everything on the LOGIT (log-odds) scale,
not raw price.

    logit(p) = ln(p / (1-p))

Why: prediction-market prices are bounded in [0,1] and represent
probabilities. A move from 0.50 -> 0.55 and a move from 0.95 -> 0.99 are
NOT the same "amount" of information -- the second is a much bigger
implied shift in confidence, even though it's a smaller raw price change.
The logit transform is the standard way to make these moves comparable
(it's the same trick used in Brier-score decompositions and betting-odds
analysis), and it also stops signals from misbehaving near the [0,1]
boundary.
"""

import numpy as np
import pandas as pd

from volatility_model import add_structural_vol, fit_K, fit_residual_garch, garch_multiplier_per_market

EPS = 1e-4  # clip to avoid logit(0) / logit(1) = +/-inf


def logit(p: pd.Series | np.ndarray) -> np.ndarray:
    p_clipped = np.clip(p, EPS, 1 - EPS)
    return np.log(p_clipped / (1 - p_clipped))


def add_logit_price(df: pd.DataFrame, price_col: str = "price") -> pd.DataFrame:
    df = df.copy()
    df["logit_price"] = logit(df[price_col].values)
    return df


def momentum_signal(df: pd.DataFrame, lookback: int) -> pd.Series:
    """
    Momentum signal = change in logit-price over the last `lookback` bars,
    computed PER MARKET (grouped) so history never leaks across markets.

    Interpretation: signal > 0 means the market has been moving toward
    YES; the momentum hypothesis says it keeps moving that way
    (underreaction to information). Interpretation is symmetric for < 0.
    """
    return df.groupby("market_id")["logit_price"].diff(lookback)


def reversal_signal(df: pd.DataFrame, lookback: int) -> pd.Series:
    """
    Reversal signal = z-score of the current logit-price relative to its
    own rolling mean/std over `lookback` bars, per market.

    Interpretation: |z| large means the price is "overextended" relative
    to its recent range. The reversal hypothesis bets it snaps back
    (overreaction to information / liquidity-driven overshoot).
    """
    grp = df.groupby("market_id")["logit_price"]
    roll_mean = grp.transform(lambda s: s.rolling(lookback, min_periods=lookback).mean())
    roll_std = grp.transform(lambda s: s.rolling(lookback, min_periods=lookback).std())
    z = (df["logit_price"] - roll_mean) / roll_std.replace(0, np.nan)
    return z


def add_signals(df: pd.DataFrame, mom_lookback: int = 5, rev_lookback: int = 10) -> pd.DataFrame:
    df = add_logit_price(df)
    df = df.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    df["mom_signal"] = momentum_signal(df, mom_lookback)
    df["rev_signal"] = reversal_signal(df, rev_lookback)
    return df


# ---------------------------------------------------------------------------
# STRUCTURAL-VOLATILITY-NORMALIZED signals (Xi, Moallemi, Pai & Wang 2026 DR-AS
# model -- see volatility_model.py). These replace the naive rolling-std /
# raw-logit-diff normalization with a theoretically grounded, no-lookback-
# window-needed one-step variance forecast: h^2 = p(1-p)/tau (+ AS term if
# spread data supplied). Momentum/reversal signal VALUES here are proper
# z-scores / t-stats (raw move divided by sqrt of the model's predicted
# variance over that window), not unitless logit-diffs -- which is also why
# the natural entry thresholds for these (see backtest.py's
# CANDIDATE_MIN_STRUCT_MOM/REV) look like ordinary z-critical-values (~1.5-2)
# rather than the ad-hoc 0.3 / 1.0 thresholds used for the naive versions.
# ---------------------------------------------------------------------------
def structural_momentum_signal(df: pd.DataFrame, lookback: int, var_col: str = "h2") -> pd.Series:
    """
    Vol-normalized momentum z-score: raw price change over `lookback` bars,
    divided by sqrt(cumulative structural variance over that same window).
    Requires df to already have a variance column (`var_col`) -- either the
    DR-AS-only 'h2' (see volatility_model.add_structural_vol) or the
    GARCH-combined 'h2_combined' (see add_structural_signals below).

    This is the direct discrete-time analogue of the paper's own calendar-
    time variance-budget identity (their Appendix A.3): variance of a sum of
    increments over a window equals the (expected) sum of per-step variances,
    so summing h2 over the lookback window gives the correct denominator for
    a window-length price move, rather than assuming constant volatility
    (what a naive fixed logit-diff threshold implicitly does).
    """
    grp = df.groupby("market_id")
    price_change = grp["price"].diff(lookback)
    cum_var = grp[var_col].transform(lambda s: s.rolling(lookback, min_periods=lookback).sum())
    return price_change / np.sqrt(cum_var.clip(lower=1e-12))


def structural_reversal_signal(df: pd.DataFrame, lookback: int, var_col: str = "h2") -> pd.Series:
    """
    Vol-normalized reversal z-score: deviation of current price from its own
    rolling mean over `lookback` bars, divided by sqrt(cumulative structural
    variance over that window) -- the DR-AS-grounded replacement for the
    naive logit-price z-score in reversal_signal() above.
    """
    grp = df.groupby("market_id")
    roll_mean_p = grp["price"].transform(lambda s: s.rolling(lookback, min_periods=lookback).mean())
    cum_var = grp[var_col].transform(lambda s: s.rolling(lookback, min_periods=lookback).sum())
    return (df["price"] - roll_mean_p) / np.sqrt(cum_var.clip(lower=1e-12))


def add_structural_signals(df: pd.DataFrame, train_market_ids: set, struct_mom_lookback: int = 5,
                            struct_rev_lookback: int = 10, spread_col: str | None = None,
                            fit_garch: bool = True, fit_joint_garch: bool = False,
                            min_tau: float = None, bar_length: float = None) -> pd.DataFrame:
    """
    Fits K (the AS-channel scale parameter) on TRAIN markets only, then adds
    h2/h and the two DR-AS structural signal columns to the FULL df. Fitting
    K only on train and freezing it before touching test data mirrors the
    same train/test discipline backtest.py already uses for Kelly calibration
    -- K is a model parameter, and letting it see test-set realized moves
    would be a second, easy-to-miss form of look-ahead leakage.

    If fit_garch=True (default), also adds a MULTIPLICATIVE two-stage GARCH
    approximation (struct_mom_garch_signal / struct_rev_garch_signal), built
    BEFORE the paper's exact equation (18) was confirmed. This is now known
    to differ from the paper's real specification in two ways: it's
    multiplicative rather than additive, and it fits DR-AS first with GARCH
    layered on afterward rather than jointly. Kept for comparison, not
    removed -- see fit_joint_garch below for the confirmed-correct version.

    If fit_joint_garch=True (opt-in, default False -- newly built, not yet
    validated on real data), adds the CONFIRMED-correct specification from
    the paper's equation (18): h_i^2 = omega + alpha*eps_{i-1}^2 +
    beta*h_{i-1}^2 + c*b_i, with (K, omega, alpha, beta, c) estimated
    JOINTLY via Gaussian quasi-MLE (see volatility_model.fit_garch_dr_as_joint).
    Adds a third signal pair (struct_mom_jointgarch_signal /
    struct_rev_jointgarch_signal) built on h2_joint. NOTE: synthetic
    validation found a real parameter-identifiability issue between beta and
    c (they can partially trade off against each other on finite samples,
    especially when the structural predictor itself evolves smoothly) --
    individual fitted parameters shouldn't be over-interpreted, though this
    doesn't necessarily hurt out-of-sample forecast quality, which is a
    separate question the train/test backtest is built to answer empirically.

    min_tau/bar_length: MUST match your actual bar spacing, in days (see
    volatility_model.py's module docstring -- this is a real bug that was
    found and fixed: h^2 is systematically wrong by the ratio of 1 day to
    your actual bar length if these aren't set correctly). Defaults to
    volatility_model's DEFAULT_MIN_TAU/DEFAULT_BAR_LENGTH (1.0, i.e. daily
    bars) if not provided -- explicitly pass min_tau=bar_length=1/24 for
    hourly data (this project's real-data fetch default).
    """
    from volatility_model import DEFAULT_MIN_TAU, DEFAULT_BAR_LENGTH
    if min_tau is None:
        min_tau = DEFAULT_MIN_TAU
    if bar_length is None:
        bar_length = DEFAULT_BAR_LENGTH

    df = df.sort_values(["market_id", "timestamp"]).reset_index(drop=True)

    K = 0.0
    if spread_col is not None and spread_col in df.columns:
        train_df = df[df["market_id"].isin(train_market_ids)]
        realized_sq_moves = train_df.groupby("market_id")["price"].diff().pow(2)
        valid = realized_sq_moves.notna()
        if valid.sum() > 10:
            K = fit_K(
                realized_sq_moves[valid].values,
                train_df.loc[valid, "price"].values,
                train_df.loc[valid, "days_to_resolution"].values,
                train_df.loc[valid, "volume"].values,
                train_df.loc[valid, spread_col].values,
                min_tau=min_tau, bar_length=bar_length,
            )

    df = add_structural_vol(df, K=K, spread_col=spread_col, min_tau=min_tau, bar_length=bar_length)
    df["struct_mom_signal"] = structural_momentum_signal(df, struct_mom_lookback, var_col="h2")
    df["struct_rev_signal"] = structural_reversal_signal(df, struct_rev_lookback, var_col="h2")

    garch_params = None
    if fit_garch:
        # Standardized one-step residual z_i = epsilon_i / h_i. If the
        # structural model fully explained variance, z_i would be ~unit-
        # variance iid noise; leftover clustering here is exactly the
        # "residual GARCH dynamics" the paper layers on top.
        next_price = df.groupby("market_id")["price"].shift(-1)
        eps = next_price - df["price"]
        df["_z_struct"] = eps / np.sqrt(df["h2"].clip(lower=1e-12))

        train_z = df.loc[df["market_id"].isin(train_market_ids), "_z_struct"]
        garch_params = fit_residual_garch(train_z.dropna().values)

        g2 = garch_multiplier_per_market(df, "_z_struct", garch_params)
        df["h2_combined"] = df["h2"] * g2.clip(lower=1e-6)
        df["struct_mom_garch_signal"] = structural_momentum_signal(df, struct_mom_lookback, var_col="h2_combined")
        df["struct_rev_garch_signal"] = structural_reversal_signal(df, struct_rev_lookback, var_col="h2_combined")
        df = df.drop(columns=["_z_struct"])

    joint_garch_params = None
    if fit_joint_garch:
        from volatility_model import fit_garch_dr_as_joint, garch_dr_as_h2
        train_df = df[df["market_id"].isin(train_market_ids)]
        joint_garch_params = fit_garch_dr_as_joint(train_df, spread_col=spread_col,
                                                    min_tau=min_tau, bar_length=bar_length)
        df["h2_joint"] = garch_dr_as_h2(df, joint_garch_params, spread_col=spread_col,
                                        min_tau=min_tau, bar_length=bar_length)
        df["struct_mom_jointgarch_signal"] = structural_momentum_signal(df, struct_mom_lookback, var_col="h2_joint")
        df["struct_rev_jointgarch_signal"] = structural_reversal_signal(df, struct_rev_lookback, var_col="h2_joint")

    return df, K, garch_params, joint_garch_params


def liquidity_and_horizon_filter(df: pd.DataFrame, min_volume: float,
                                  min_days_to_resolution: float) -> pd.Series:
    """
    Boolean mask for tradeable rows.

    The min_days_to_resolution filter matters more than it might look:
    right before a market resolves, its price legitimately races toward
    0 or 1 as real-world uncertainty is resolved. That's not exploitable
    momentum, it's just... the market being correct. Trading a "momentum"
    signal in that window means front-running true convergence, which
    looks great in a naive backtest and loses money live (you're betting
    WITH the crowd on something that's about to be common knowledge, at
    a price that's already mostly there). Excluding this window is one
    of the single highest-value lines in this file.
    """
    return (df["volume"] >= min_volume) & (df["days_to_resolution"] >= min_days_to_resolution)


# ---------------------------------------------------------------------------
# BONUS: true (risk-free) arbitrage scanner
# ---------------------------------------------------------------------------
def complementary_mispricing_scan(yes_price: float, no_price: float,
                                   yes_fee_rate: float, no_fee_rate: float,
                                   shares: float = 100.0) -> dict:
    """
    Real, textbook arbitrage (NOT momentum/reversal): in a binary market,
    a complete set of YES+NO tokens redeems for exactly $1. If you can buy
    BOTH sides for less than $1 combined (after fees), you lock in a
    riskless profit regardless of outcome.

    cost_yes = yes_price + fee(yes_price)
    cost_no  = no_price + fee(no_price)
    profit_per_share = 1.0 - cost_yes - cost_no

    In practice this window is usually closed instantly by bots, but it's
    worth screening for continuously -- it costs nothing to check and,
    unlike momentum/reversal, a hit here is a genuine free-money event,
    not a statistical edge that can go wrong.
    """
    fee_yes = yes_fee_rate * yes_price * (1 - yes_price)
    fee_no = no_fee_rate * no_price * (1 - no_price)
    cost_yes = yes_price + fee_yes
    cost_no = no_price + fee_no
    profit_per_share = 1.0 - cost_yes - cost_no
    return {
        "is_arb": profit_per_share > 0,
        "profit_per_share": profit_per_share,
        "total_profit": profit_per_share * shares if profit_per_share > 0 else 0.0,
        "cost_yes": cost_yes,
        "cost_no": cost_no,
    }
