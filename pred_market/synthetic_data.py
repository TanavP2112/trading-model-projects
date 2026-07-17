"""
Synthetic resolved-market generator.

WHY THIS FILE EXISTS: this sandbox's network egress does not include
polymarket.com, so I cannot pull real historical data to demo the
pipeline. This module generates SYNTHETIC-BUT-STRUCTURED price paths so
run_demo.py can prove the data model / signals / backtest engine all
fit together correctly and produce sane output.

THIS IS NOT REAL DATA. Any Sharpe ratio produced against this synthetic
generator tells you the CODE WORKS, not that the STRATEGY WORKS on real
Polymarket markets. See README.md for how to swap in real data via
data_fetcher.py.

The generator deliberately embeds two textbook microstructure effects
that academic prediction-market literature hypothesizes exist to some
degree in real (especially thin/retail-dominated) markets, so the demo
has *something* for momentum and reversal signals to find:

  1. Underreaction to news ("momentum" fuel): when a shock hits, only
     part of the adjustment happens immediately; the rest bleeds in
     over the next few periods.
  2. Overreaction / overshoot ("reversal" fuel): some shocks overshoot
     and partially revert over the following periods.

Whether real Polymarket/Kalshi markets actually exhibit these effects,
and at what magnitude net of costs, is an empirical question -- that's
exactly what you'd go test with data_fetcher.py + this same signals.py
and backtest.py once you swap in real history.
"""

import numpy as np
import pandas as pd

from signals import logit

CATEGORIES = ["crypto", "sports", "finance", "politics", "economics", "culture"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate_one_market(market_id: int, rng: np.random.Generator) -> pd.DataFrame:
    duration_days = int(rng.uniform(5, 45))
    category = rng.choice(CATEGORIES)
    volume = float(np.exp(rng.uniform(np.log(5_000), np.log(2_000_000))))

    true_p = rng.beta(1.5, 1.5)               # the "true" resolution probability
    outcome = int(rng.random() < true_p)
    target_logit = logit(np.array([0.985 if outcome else 0.015]))[0]

    p0 = np.clip(rng.beta(2, 2), 0.05, 0.95)
    x = logit(np.array([p0]))[0]               # current logit-price, evolves below

    n_steps = duration_days                     # 1 bar/day for this demo
    path = np.zeros(n_steps)
    pending_adjustment = 0.0                    # underreaction "carry-over" from last shock

    # Poisson-ish news arrival
    shock_prob_per_day = 0.25

    for t in range(n_steps):
        # 1. Bleed in any pending underreaction from a prior shock
        bleed = pending_adjustment * 0.35
        x += bleed
        pending_adjustment -= bleed

        # 2. Possibly a fresh news shock today
        if rng.random() < shock_prob_per_day:
            # direction biased toward the true outcome (informative shock),
            # but noisy -- this is what makes it a *statistical* edge, not arbitrage
            remaining = target_logit - x
            informative = rng.random() < 0.53
            direction = np.sign(remaining) if informative else rng.choice([-1, 1])
            magnitude = rng.exponential(0.32)

            is_overreaction = rng.random() < 0.30
            if is_overreaction:
                # overshoot now, partially revert later (reversal fuel)
                overshoot = direction * magnitude * 1.3
                x += overshoot
                pending_adjustment += -overshoot * 0.40   # scheduled partial reversion
            else:
                # underreact now, drift the rest in over coming days (momentum fuel)
                immediate = direction * magnitude * 0.5
                x += immediate
                pending_adjustment += direction * magnitude * 0.5

        # 3. Microstructure noise
        x += rng.normal(0, 0.24)

        # 4. Gentle pull toward eventual truth as resolution approaches (info accretion)
        days_left = n_steps - t
        pull_strength = 0.05 * (1.0 / max(days_left, 1))
        x += (target_logit - x) * pull_strength

        path[t] = x

    # Force convergence in the final 1-2 bars (market "knows" by settlement)
    path[-1] = target_logit + rng.normal(0, 0.05)
    if n_steps > 1:
        path[-2] = 0.6 * path[-2] + 0.4 * target_logit

    prices = _sigmoid(path)
    days_to_res = np.arange(n_steps - 1, -1, -1).astype(float)
    ts = pd.date_range("2024-01-01", periods=n_steps, freq="D") + pd.Timedelta(days=market_id % 365)

    # Synthetic bid-ask spread: follows the Glosten-Milgrom shape derived in
    # the DR-AS paper (spread ~ proportional to p(1-p) for a fixed informed-
    # trading share alpha), scaled down by liquidity (more volume -> tighter
    # spread) plus a little multiplicative noise. This exists purely so the
    # demo can exercise the FULL DR-AS model (both channels) rather than
    # only the DR-only fallback -- real Polymarket spread data would come
    # from the order book, not this synthetic approximation.
    alpha_informed = rng.uniform(0.05, 0.25)   # informed-trader share for this market
    liquidity_scale = 1.0 / np.sqrt(max(volume, 1.0) / 10_000.0)
    gm_shape = 4 * prices * (1 - prices)  # peaks at 1.0 when p=0.5, matches paper's derivation
    spread_noise = rng.uniform(0.7, 1.3, size=n_steps)
    spreads = np.clip(alpha_informed * gm_shape * liquidity_scale * spread_noise * 0.5, 0.001, 0.5)

    return pd.DataFrame({
        "market_id": market_id,
        "category": category,
        "timestamp": ts,
        "price": prices,
        "volume": volume,
        "days_to_resolution": days_to_res,
        "outcome": outcome,
        "spread": spreads,
    })


def simulate_market_panel(n_markets: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = [simulate_one_market(i, rng) for i in range(n_markets)]
    panel = pd.concat(frames, ignore_index=True)
    return panel
