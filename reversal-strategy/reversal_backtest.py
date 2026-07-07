"""
Cross-Sectional Short-Term Reversal Backtest
=============================================

A stat-arb style backtest of the classic short-term reversal effect:
stocks with the worst recent returns tend to outperform stocks with the
best recent returns over the following few days, in a dollar-neutral
long/short portfolio.

Pipeline:
    1. Download daily adjusted prices for a universe of tickers
    2. Build a market-relative reversal signal (residualize vs. universe return)
    3. Rank into a signal-weighted, dollar-neutral long/short portfolio
    4. Apply an explicit transaction cost model (spread + square-root impact)
    5. Walk-forward backtest (no lookahead: signal at t uses data through t only,
       traded into position at t+1 close, that position is what earns t+1->t+2 return)
    6. Report gross vs. net performance, turnover, and a simple factor decomposition

Author: quant new-grad project skeleton
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 1. CONFIG
# ----------------------------------------------------------------------

CONFIG = {
    "strategy": "reversal",  # "reversal" or "momentum"
    "start_date": "2015-01-01",
    "end_date": "2025-01-01",
    "lookback": 1,          # days of past return used to build signal (1 = classic ST reversal)
    "skip_days": 0,         # days to skip between signal window and present (momentum uses this)
    "rebalance_days": 1,    # only re-form the portfolio every N days; 1 = daily
    "rebalance_band": 0.0,  # no-trade zone: skip a trade if |target - current weight| < this
    "n_quantiles": 5,       # split universe into this many buckets; trade top vs bottom
    "vol_lookback": 20,     # days for realized vol used in cost model & risk scaling
    "target_gross_leverage": 2.0,   # 1.0 long + 1.0 short
    "half_spread_bps": 2,   # one-way half bid-ask spread cost, in bps of notional.
                            # Default assumes liquid large caps (~2bp); tighten
                            # further for true mega-caps, widen for small/mid-cap.
    "impact_coeff": 0.1,    # square-root impact model coefficient (tune per universe)
    "adv_participation": 0.02,  # assume we trade 2% of average daily volume

    # --- Volatility regime overlay ---
    "regime_vol_lookback": 20,   # days used to estimate realized vol for regime detection
    # Annualized realized-vol cutoffs (roughly VIX-equivalent terms) that define regimes.
    # e.g. ann. vol < 15% -> "low", 15-25% -> "medium", 25-40% -> "high", >40% -> "extreme"
    "regime_thresholds": {"low": 0.15, "medium": 0.25, "high": 0.40},
    # Leverage multiplier applied to the base portfolio weights in each regime.
    # "extreme" set near-zero on the prior that crisis-level moves are more likely
    # information-driven than liquidity-driven and reversal tends to break down.
    "regime_scalars": {"low": 0.7, "medium": 1.0, "high": 1.3, "extreme": 0.1},
}

# Cross-sectional momentum variant: 12-month lookback, skip the most recent
# month (classic Jegadeesh-Titman convention -- avoids contaminating the
# momentum signal with short-term reversal), vol-adjusted score, rebalanced
# roughly monthly rather than daily since it's a slow-moving signal.
MOMENTUM_CONFIG = dict(CONFIG)
MOMENTUM_CONFIG.update({
    "strategy": "momentum",
    "lookback": 252,        # ~12 months of trading days
    "skip_days": 21,        # ~1 month skipped
    "rebalance_days": 21,   # re-form portfolio monthly, not daily
    "vol_lookback": 252,    # momentum score normalizes by vol over the same window
})


# ----------------------------------------------------------------------
# 2. DATA
# ----------------------------------------------------------------------

def get_sp500_tickers():
    """
    Pull the current S&P 500 constituent list from a public dataset (updated
    periodically, not point-in-time / not survivorship-bias-free -- fine for
    a project, but disclose this limitation if you use it for anything more
    rigorous). Gives you ~500 names instead of a hand-picked list of 30,
    which matters a lot for a cross-sectional strategy's breadth.
    """
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    df = pd.read_csv(url)
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()  # BRK.B -> BRK-B for yfinance
    return tickers


def download_universe(tickers, start, end):
    """Download adjusted close prices and volume for a list of tickers."""
    px = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False, group_by="ticker")
    close = pd.DataFrame({t: px[t]["Close"] for t in tickers if t in px.columns.get_level_values(0)})
    volume = pd.DataFrame({t: px[t]["Volume"] for t in tickers if t in px.columns.get_level_values(0)})
    close = close.dropna(axis=1, how="all")
    volume = volume[close.columns]
    return close, volume


# ----------------------------------------------------------------------
# 3. SIGNAL
# ----------------------------------------------------------------------

def build_reversal_signal(returns, lookback=1):
    """
    Market-relative reversal signal.
    signal_t = -(r_i,t - cross_sectional_mean_r_t), using trailing `lookback`-day return.
    Negative sign: worst recent performers get the highest (most positive) signal.
    """
    trailing_ret = returns.rolling(lookback).sum()
    cs_mean = trailing_ret.mean(axis=1)
    excess = trailing_ret.sub(cs_mean, axis=0)
    signal = -excess
    return signal


def build_momentum_signal(returns, lookback=252, skip=21, vol_lookback=None):
    """
    Cross-sectional, volatility-adjusted momentum signal (positive sign: past
    winners get the highest signal -- opposite of reversal).

    signal_t = sum(returns from t-lookback to t-skip) / realized_vol(same window)

    The `skip` window excludes the most recent `skip` days from the return
    calculation (classic Jegadeesh-Titman "skip-month" convention), since very
    recent returns are contaminated by short-term reversal rather than
    genuine trend persistence.
    """
    if vol_lookback is None:
        vol_lookback = lookback

    # Cumulative return over [t-lookback, t-skip], excluding the most recent `skip` days.
    cum_ret_full = (1 + returns).rolling(lookback).apply(np.prod, raw=True) - 1
    cum_ret_recent = (1 + returns).rolling(skip).apply(np.prod, raw=True) - 1
    # Ratio of full-window growth to recent-window growth isolates the "older" part of the move.
    trailing_momentum_ret = (1 + cum_ret_full) / (1 + cum_ret_recent) - 1

    vol = returns.rolling(vol_lookback).std() * np.sqrt(252)
    signal = trailing_momentum_ret / vol
    return signal


def build_signal(returns, config):
    """Dispatch to the signal builder named in config['strategy']."""
    if config["strategy"] == "reversal":
        return build_reversal_signal(returns, lookback=config["lookback"])
    elif config["strategy"] == "momentum":
        return build_momentum_signal(returns, lookback=config["lookback"],
                                      skip=config["skip_days"],
                                      vol_lookback=config["vol_lookback"])
    else:
        raise ValueError(f"Unknown strategy: {config['strategy']}")


def zscore_cross_section(signal):
    """Standardize signal cross-sectionally each day (mean 0, std 1 across names)."""
    mu = signal.mean(axis=1)
    sd = signal.std(axis=1)
    z = signal.sub(mu, axis=0).div(sd, axis=0)
    return z


# ----------------------------------------------------------------------
# 4. PORTFOLIO CONSTRUCTION
# ----------------------------------------------------------------------

def build_weights(signal_z, target_gross=2.0):
    """
    Signal-weighted, dollar-neutral portfolio.
    Weight_i = z_i / sum(|z|) * target_gross, so long leg sums to +target_gross/2
    and short leg sums to -target_gross/2 (roughly, if z is symmetric).
    """
    abs_sum = signal_z.abs().sum(axis=1)
    weights = signal_z.div(abs_sum, axis=0) * target_gross
    weights = weights.fillna(0)
    return weights


def apply_rebalance_schedule(weights, rebalance_days=1):
    """
    Only update the traded portfolio every `rebalance_days` days; hold the
    prior weights constant in between. This is what actually implements a
    holding period / reduces turnover for slower signals like momentum,
    rather than re-forming the full book every single day.
    """
    if rebalance_days <= 1:
        return weights
    schedule_mask = np.arange(len(weights)) % rebalance_days == 0
    rebalanced = weights.where(pd.Series(schedule_mask, index=weights.index), other=np.nan)
    rebalanced = rebalanced.ffill().fillna(0)
    return rebalanced


def apply_rebalance_band(weights, band=0.0):
    """
    No-trade zone: only actually move a position if the target weight has
    drifted from the currently-held weight by more than `band` (in absolute
    weight units, e.g. 0.01 = 1% of gross). Otherwise hold the prior position.

    This directly targets the turnover problem for fast-rebalancing signals
    like daily reversal -- a lot of day-to-day weight changes are small noise
    around a similar target, not a meaningfully different position, and
    trading them anyway is pure cost with no signal benefit.
    """
    if band <= 0:
        return weights

    held = weights.copy()
    prev_held = weights.iloc[0].copy()
    for i in range(1, len(weights)):
        target = weights.iloc[i]
        diff = (target - prev_held).abs()
        move_mask = diff > band
        new_held = prev_held.copy()
        new_held[move_mask] = target[move_mask]
        held.iloc[i] = new_held
        prev_held = new_held
    return held


# ----------------------------------------------------------------------
# 4b. VOLATILITY REGIME DETECTION & OVERLAY
# ----------------------------------------------------------------------

def compute_regime(returns, config, vix=None):
    """
    Classify each day into a volatility regime: 'low', 'medium', 'high', 'extreme'.

    If `vix` (a Series of VIX levels, e.g. pulled via yfinance ticker '^VIX') is
    provided, regimes are based on VIX/100 as the annualized-vol proxy directly.
    Otherwise, regimes are estimated from the universe's own trailing realized
    vol (equal-weighted average return, annualized).

    IMPORTANT: uses only trailing/rolling data (shifted where needed) so there
    is no lookahead -- the regime on day t is knowable using data through t-1.
    """
    lb = config["regime_vol_lookback"]
    thresh = config["regime_thresholds"]

    if vix is not None:
        ann_vol = (vix / 100.0).reindex(returns.index).ffill().shift(1)
    else:
        universe_ret = returns.mean(axis=1)
        ann_vol = (universe_ret.rolling(lb).std() * np.sqrt(252)).shift(1)

    regime = pd.Series(index=ann_vol.index, dtype="object")
    regime[ann_vol < thresh["low"]] = "low"
    regime[(ann_vol >= thresh["low"]) & (ann_vol < thresh["medium"])] = "medium"
    regime[(ann_vol >= thresh["medium"]) & (ann_vol < thresh["high"])] = "high"
    regime[ann_vol >= thresh["high"]] = "extreme"
    return regime, ann_vol


def apply_regime_scaling(weights, regime, config):
    """Scale portfolio weights by a per-regime leverage multiplier."""
    scalar_map = config["regime_scalars"]
    scalar = regime.map(scalar_map).reindex(weights.index).fillna(1.0)
    scaled_weights = weights.mul(scalar, axis=0)
    return scaled_weights, scalar


# ----------------------------------------------------------------------
# 5. TRANSACTION COST MODEL
# ----------------------------------------------------------------------

def compute_trade_costs(weights, returns, dollar_volume, config):
    """
    Cost per name per day = spread cost + square-root impact cost, applied to
    the *change* in position (turnover), not the position itself.

    spread_cost = half_spread_bps * |delta_weight|
    impact_cost = impact_coeff * sigma * sqrt(participation) * |delta_weight|
    """
    delta_w = weights.diff().abs().fillna(0)

    sigma = returns.rolling(config["vol_lookback"]).std()
    sigma = sigma.reindex(columns=weights.columns).ffill()

    spread_cost = (config["half_spread_bps"] / 1e4) * delta_w
    impact_cost = config["impact_coeff"] * sigma * np.sqrt(config["adv_participation"]) * delta_w

    total_cost = (spread_cost + impact_cost).sum(axis=1)
    turnover = delta_w.sum(axis=1)
    return total_cost, turnover


# ----------------------------------------------------------------------
# 6. BACKTEST LOOP
# ----------------------------------------------------------------------

def run_backtest(close, volume, config, vix=None, use_regime_overlay=False):
    returns = close.pct_change()

    signal = build_signal(returns, config)
    signal_z = zscore_cross_section(signal)

    weights = build_weights(signal_z, config["target_gross_leverage"])
    weights = apply_rebalance_schedule(weights, config.get("rebalance_days", 1))
    weights = apply_rebalance_band(weights, config.get("rebalance_band", 0.0))

    regime, ann_vol = compute_regime(returns, config, vix=vix)
    if use_regime_overlay:
        weights, regime_scalar = apply_regime_scaling(weights, regime, config)

    # IMPORTANT: shift by 1 day so info known through day t only affects t+1's return.
    weights_traded = weights.shift(1)

    gross_ret = (weights_traded * returns).sum(axis=1)

    dollar_volume = (close * volume)
    costs, turnover = compute_trade_costs(weights_traded, returns, dollar_volume, config)

    net_ret = gross_ret - costs

    results = pd.DataFrame({
        "gross_ret": gross_ret,
        "net_ret": net_ret,
        "cost": costs,
        "turnover": turnover,
        "regime": regime.reindex(gross_ret.index),
        "ann_vol": ann_vol.reindex(gross_ret.index),
    }).dropna(subset=["gross_ret", "net_ret"], how="all")

    return results, weights_traded, signal_z


# ----------------------------------------------------------------------
# 7. PERFORMANCE METRICS
# ----------------------------------------------------------------------

def annualized_sharpe(ret, periods_per_year=252):
    mu = ret.mean() * periods_per_year
    sd = ret.std() * np.sqrt(periods_per_year)
    return mu / sd if sd > 0 else np.nan


def max_drawdown(cum_ret):
    running_max = cum_ret.cummax()
    dd = cum_ret / running_max - 1
    return dd.min()


def performance_summary(results):
    cum_gross = (1 + results["gross_ret"]).cumprod()
    cum_net = (1 + results["net_ret"]).cumprod()

    summary = {
        "Gross Ann. Return": results["gross_ret"].mean() * 252,
        "Net Ann. Return": results["net_ret"].mean() * 252,
        "Gross Sharpe": annualized_sharpe(results["gross_ret"]),
        "Net Sharpe": annualized_sharpe(results["net_ret"]),
        "Avg Daily Turnover": results["turnover"].mean(),
        "Avg Daily Cost (bps)": results["cost"].mean() * 1e4,
        "Max Drawdown (net)": max_drawdown(cum_net),
        "Max Drawdown (gross)": max_drawdown(cum_gross),
    }
    return pd.Series(summary)


def performance_by_regime(results):
    """Break gross/net Sharpe, turnover, and cost down by volatility regime,
    so you can see directly whether the strategy is helped or hurt by
    high-vol/event-driven periods."""
    rows = {}
    order = ["low", "medium", "high", "extreme"]
    for regime_name in order:
        sub = results[results["regime"] == regime_name]
        if len(sub) < 5:
            continue
        rows[regime_name] = {
            "n_days": len(sub),
            "Gross Sharpe": annualized_sharpe(sub["gross_ret"]),
            "Net Sharpe": annualized_sharpe(sub["net_ret"]),
            "Gross Ann. Return": sub["gross_ret"].mean() * 252,
            "Net Ann. Return": sub["net_ret"].mean() * 252,
            "Avg Daily Turnover": sub["turnover"].mean(),
            "Avg Daily Cost (bps)": sub["cost"].mean() * 1e4,
        }
    return pd.DataFrame(rows).T


def diagnose_sharpe_gap(results, label=""):
    """
    Quick localization check: is a low net Sharpe a signal problem or a cost
    problem? Print gross vs. net side by side with a plain-language read.
    """
    gross_sharpe = annualized_sharpe(results["gross_ret"])
    net_sharpe = annualized_sharpe(results["net_ret"])
    avg_cost_bps = results["cost"].mean() * 1e4
    avg_turnover = results["turnover"].mean()

    print(f"--- {label} diagnostic ---")
    print(f"Gross Sharpe: {gross_sharpe:.3f}   Net Sharpe: {net_sharpe:.3f}")
    print(f"Avg daily turnover: {avg_turnover:.3f}   Avg daily cost: {avg_cost_bps:.2f} bps")

    if gross_sharpe < 1.0:
        print("-> Gross Sharpe itself is weak. This looks like a SIGNAL problem: "
              "more likely fixed by widening the universe (breadth) or a different "
              "period/effect than by tuning costs.")
    elif net_sharpe < 0.5 * gross_sharpe:
        print("-> Gross Sharpe is solid but net collapses. This looks like a COST/"
              "TURNOVER problem: try rebalance_band, fewer names traded (deciles "
              "only), or recalibrating half_spread_bps/impact_coeff to your "
              "actual universe's liquidity.")
    else:
        print("-> Gross and net are reasonably close; costs aren't the main issue here.")
    print()
    return {"gross_sharpe": gross_sharpe, "net_sharpe": net_sharpe,
            "avg_cost_bps": avg_cost_bps, "avg_turnover": avg_turnover}


def factor_decomposition(results, market_ret):
    """Regress net strategy returns on the (universe) market return to check
    the alpha isn't just a leveraged market bet. Uses simple OLS via numpy."""
    df = pd.concat([results["net_ret"], market_ret], axis=1).dropna()
    df.columns = ["strategy", "market"]
    X = np.vstack([np.ones(len(df)), df["market"].values]).T
    y = df["strategy"].values
    beta, resid, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    alpha_annualized = beta[0] * 252
    return {"alpha_daily": beta[0], "alpha_annualized": alpha_annualized, "beta_mkt": beta[1]}


# ----------------------------------------------------------------------
# 7b. COMBINING MULTIPLE STRATEGIES
# ----------------------------------------------------------------------

def combine_returns(strategy_returns, weights=None):
    """
    Combine multiple strategy net-return series into a single blended return
    stream. `strategy_returns` is a dict {name: return Series}. If `weights`
    (dict {name: float}) is None, uses equal weight.

    Each strategy return series is treated as if run at its OWN target gross
    leverage already (i.e., each is a fully-formed, tradable sleeve); `weights`
    here represent how much capital/risk you allocate to each sleeve, not a
    re-blend of the underlying signals.
    """
    df = pd.DataFrame(strategy_returns).dropna()
    if weights is None:
        weights = {k: 1.0 / len(strategy_returns) for k in strategy_returns}
    w = pd.Series(weights)
    combined = (df * w).sum(axis=1)
    return combined, df


def strategy_correlation(strategy_returns):
    """Correlation matrix between strategies' net daily returns."""
    df = pd.DataFrame(strategy_returns).dropna()
    return df.corr()


def max_sharpe_weights(strategy_returns, long_only=True):
    """
    Solve for the (tangency-portfolio-style) weights that maximize the
    Sharpe ratio of a blend of strategies, given their historical mean and
    covariance of daily returns. Closed-form: w ∝ Σ^-1 * mu, then normalized
    to sum to 1 (weights represent capital allocation across sleeves, not
    leverage). If long_only, negative weights are clipped to zero and
    renormalized (a simple, common practical constraint).
    """
    df = pd.DataFrame(strategy_returns).dropna()
    mu = df.mean().values * 252
    cov = df.cov().values * 252
    try:
        raw_w = np.linalg.solve(cov, mu)
    except np.linalg.LinAlgError:
        raw_w = np.linalg.lstsq(cov, mu, rcond=None)[0]

    if long_only:
        raw_w = np.clip(raw_w, 0, None)

    if raw_w.sum() == 0:
        raw_w = np.ones(len(raw_w))
    weights = raw_w / raw_w.sum()
    return dict(zip(df.columns, weights))


def sweep_blend_weights(strategy_returns, n_points=21):
    """
    For exactly two strategies, sweep the blend weight from 0 to 1 and report
    the combined Sharpe at each point -- useful for plotting a simple
    'efficient frontier' style chart and sanity-checking the closed-form
    max_sharpe_weights result against a brute-force grid.
    """
    names = list(strategy_returns.keys())
    if len(names) != 2:
        raise ValueError("sweep_blend_weights only supports exactly 2 strategies")
    df = pd.DataFrame(strategy_returns).dropna()

    rows = []
    for w in np.linspace(0, 1, n_points):
        blend = w * df[names[0]] + (1 - w) * df[names[1]]
        rows.append({
            "weight_" + names[0]: w,
            "weight_" + names[1]: 1 - w,
            "Sharpe": annualized_sharpe(blend),
            "Ann. Return": blend.mean() * 252,
            "Ann. Vol": blend.std() * np.sqrt(252),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 8. MAIN
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Example universe -- swap in your real S&P 500 / Russell 1000 list.
    tickers = [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "V", "UNH",
        "HD", "PG", "MA", "DIS", "BAC", "XOM", "PFE", "KO", "PEP", "CSCO",
        "INTC", "T", "VZ", "ABT", "CRM", "NKE", "MRK", "WMT", "MCD", "ADBE",
    ]

    print("Downloading data...")
    close, volume = download_universe(tickers, CONFIG["start_date"], CONFIG["end_date"])
    print(f"Universe after download: {close.shape[1]} names, {close.shape[0]} days")

    print("\n--- Reversal strategy ---")
    results_rev, _, _ = run_backtest(close, volume, CONFIG)
    print(performance_summary(results_rev).round(4))

    print("\n--- Momentum strategy ---")
    results_mom, _, _ = run_backtest(close, volume, MOMENTUM_CONFIG)
    print(performance_summary(results_mom).round(4))

    strategy_returns = {
        "reversal": results_rev["net_ret"],
        "momentum": results_mom["net_ret"],
    }

    print("\n--- Correlation between strategies (net daily returns) ---")
    print(strategy_correlation(strategy_returns).round(3))

    print("\n--- Equal-weight blend ---")
    combined_eq, _ = combine_returns(strategy_returns)  # default: equal weight
    print(f"Combined Sharpe (equal weight): {annualized_sharpe(combined_eq):.3f}")

    print("\n--- Sharpe-optimal blend ---")
    opt_weights = max_sharpe_weights(strategy_returns)
    print(f"Optimal weights: {opt_weights}")
    combined_opt, _ = combine_returns(strategy_returns, weights=opt_weights)
    print(f"Combined Sharpe (optimal weight): {annualized_sharpe(combined_opt):.3f}")

    print("\n--- Blend weight sweep ---")
    sweep = sweep_blend_weights(strategy_returns)
    print(sweep.round(4))

    print("\nSaved daily results to CSV.")