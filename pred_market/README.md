# Prediction-Market Momentum/Reversal Research Project

A backtesting framework for statistical momentum/reversal strategies on
Polymarket, built to target a Sharpe ratio ≥ 2.0. Read the whole "Critical
caveats" section before you believe any number this code prints.

## Terminology note (read this first)

**Momentum/reversal is not arbitrage.** True arbitrage in a prediction market
is risk-free: buying YES + NO for a combined price under $1 (after fees) is
a guaranteed profit regardless of outcome, and so is a cross-platform price
gap for the same event on Kalshi vs. Polymarket. Momentum and reversal are
**statistical, directional strategies** — you're betting that recent price
behavior predicts future price behavior, and you can be wrong on any given
trade. This project builds the momentum/reversal engine you asked for, plus
a small bonus true-arbitrage scanner (`signals.complementary_mispricing_scan`)
since it's cheap to add and worth running continuously alongside anything
else you build in this space.

## Why Polymarket over Kalshi

For a **backtesting-first** project specifically, Polymarket wins on pure
convenience:

|                                      | Polymarket               | Kalshi                                                    |
| ------------------------------------ | ------------------------ | --------------------------------------------------------- |
| Market discovery (Gamma API)         | Fully public, no auth    | Public, no auth                                           |
| Price history / orderbook (CLOB API) | Fully public, no auth    | Public for recent data                                    |
| Trade/position history (Data API)    | Fully public, no auth    | Split into live/historical tiers (since Feb 2026)         |
| Auth needed just to backtest         | **None**                 | None for basics, but full historical depth is less mature |
| Auth needed to place live orders     | Wallet EIP-712 signature | Per-request RSA-PSS signing                               |

Kalshi is CFTC-regulated and has a cleaner single-base-URL REST design once
you're set up, which matters more once you're live-trading. But Polymarket's
three fully-public read APIs (Gamma, CLOB, Data) mean you can pull years of
resolved-market history with zero account, zero key, zero signing — which
is exactly what a research-first workflow wants.

## Project structure

```
config.py          API endpoints, Polymarket's real fee schedule, strategy constants
fees.py            Implements Polymarket's actual taker-fee formula
data_fetcher.py     REAL Polymarket data pulls (Gamma + CLOB), zero-auth
synthetic_data.py   Synthetic market-path generator (demo/pipeline-validation only)
signals.py          Logit-scale momentum & reversal signals + true-arb scanner
backtest.py         Train/test calibration, Kelly-capped sizing, trade simulation, metrics
run_demo.py         Runs the backtest
volatility_model.py Incorporates a volatility model
```

## Running it

```bash
pip install pandas numpy matplotlib requests
python run_demo.py
```

Polymarket ships an **official unified Python SDK** (`polymarket-client`,
currently in beta) that's a meaningfully better foundation than raw REST
calls — typed models, real confirmed parameter names, built-in pagination,
and an optional `to_pandas()` flattening helper. `data_fetcher.py` uses it
as the preferred path, with a raw-`requests` fallback if you'd rather skip
the dependency.

```bash
pip install polymarket-client pyarrow
```

## Methodology

**Logit-scale signals.** Everything is computed on `logit(p) = ln(p/(1-p))`,
not raw price. A move from 0.95→0.99 is a much bigger shift in implied
confidence than 0.50→0.54, even though it's numerically smaller — the logit
transform makes these comparable and avoids signal artifacts near 0/1.

- **Momentum**: `logit(p_t) - logit(p_{t-lookback})`. Bet _with_ the recent
  move (underreaction hypothesis).
- **Reversal**: z-score of current logit-price vs. its own rolling
  mean/std. Bet _against_ an overextended move (overreaction hypothesis).

**Time-to-resolution filter.** Trades are only considered when at least
`MIN_DAYS_TO_RESOLUTION` remain. Right before a market resolves, price
legitimately races to 0 or 1 as real uncertainty resolves — that's not
exploitable momentum, that's the market being correct. Trading that window
is front-running convergence, not finding an edge, and it's one of the
highest-value filters in this codebase.

**Train/test split by market start date** (65/35, chronological — never
shuffled, since shuffling would leak future information backward).

**Calibration, frozen before touching test data.** On the training set,
candidate trades are bucketed into signal deciles. Each bucket's empirical
win rate and average entry price become a Kelly-fraction position size,
run through:

1. A **statistical-significance filter** (z-test on edge vs. standard
   error, `z > 2.0`, plus a minimum sample size) — a decile that "looks"
   profitable with 25 noisy trades and no real edge is exactly what blows
   up a live account. This threshold is deliberately stricter than a naive
   95% single-test cutoff because we're testing 10 buckets _simultaneously_
   — under pure noise you'd expect ~1 in 10 to clear a lenient bar by
   chance alone (see the multiple-comparisons note in `backtest.py`).
2. **Fractional Kelly** (1/4 Kelly, a standard risk-of-ruin haircut) and a
   **hard cap** (5% of bankroll per trade), regardless of what the Kelly
   formula says.

**Fees.** Implements Polymarket's real Fee Structure V2 formula:
`fee = shares × category_fee_rate × p × (1-p)`, charged on takers only
(this assumes you cross the spread — the realistic case for a
signal-driven strategy that needs to act at a specific time, not patiently
sit as a maker). An additional flat spread-cost assumption is layered on
top (`ASSUMED_SPREAD_COST` in `config.py`) since the demo has no real
level-2 book.

## The synthetic demo — what it does and does NOT show

`synthetic_data.py` generates fake resolved markets whose price paths embed
two textbook microstructure effects (partial adjustment / underreaction,
and occasional overreaction-then-revert) so the pipeline has _something_ to
find. **This proves the code works. It proves nothing about whether this
edge exists on real Polymarket markets, at what magnitude, or whether it
survives real transaction costs and real order-book depth.**

## Structural volatility signals (DR-AS model)

`volatility_model.py` implements the DR-AS structural volatility model from
Xi, Moallemi, Pai & Wang, _"Volatility in Prediction Markets: A Structural
Approach"_ (arXiv:2607.08199, 2026). One-step conditional variance is
decomposed into two additive channels:

    h^2 = p(1-p)/tau                <- Wright-Fisher deadline-resolution (DR) channel
        + K * nu(volume) * spread^2/4  <- Glosten-Milgrom adverse-selection (AS) channel

**DR channel**: zero free parameters. `p(1-p)` is remaining binary
uncertainty, `tau` is time-to-resolution. This term alone peaks at p=0.5 and
mechanically explodes as `tau -> 0` — both are real structural features of a
market approaching settlement, confirmed as model-free stylized facts in the
paper's own data, not artifacts.

**AS channel**: one free parameter `K`, fit via OLS on train-market data only
(frozen before touching test data, same discipline as the Kelly calibration
elsewhere in this project) — see `signals.add_structural_signals()`.

**What's exact vs. approximated**: the DR term and the AS term's functional
_form_ are exact per the paper's derivation. The specific concave volume-
scaling function they found strongest wasn't fully specified in the portion
of the paper reviewed while building this, so `volatility_model.py` uses
`log1p(volume)` as a defensible placeholder — flagged clearly in that file.
Real bid-ask spread isn't in `data_fetcher.py`'s default pull (Polymarket's
`get_spread()`/`get_order_book()` would need one extra API call per market
to add it); without a spread column, the model cleanly falls back to
DR-only, which the paper itself reports "already improves substantially" on
generic GARCH benchmarks.

**Two signals built on top of it** (`signals.py`):

- `structural_momentum_signal`: raw price move over a lookback window,
  divided by sqrt(cumulative structural variance over that same window) —
  a properly vol-normalized momentum z-score, replacing the naive unitless
  logit-diff. This is the discrete-time analogue of the paper's own
  calendar-time variance-budget identity (summing per-step variances to get
  a window's variance).
- `structural_reversal_signal`: deviation from a rolling mean, divided by
  the same cumulative-variance denominator, replacing the naive empirical
  rolling-std z-score.

Because these are now theoretically-grounded z-scores rather than ad-hoc
units, their entry thresholds (`CANDIDATE_MIN_STRUCT_MOM`/`_REV` in
`backtest.py`) are ordinary z-critical-values — **but check each signal's own
empirical quantiles before trusting a fixed threshold**: momentum (a
cumulative window move) and reversal (a point deviation from a local mean)
have genuinely different typical scales, and a threshold well-calibrated for
one can be wildly mis-selective for the other. This project's defaults were
set by checking `panel['struct_*_signal'].abs().quantile([0.9,0.95,0.99])`
on the demo panel, not assumed.

### What the demo run actually showed (and a bug caught along the way)

## Critical caveats before you trust ANY of this on real money

1. **One train/test split is not validation.** Do proper walk-forward
   validation across many rolling windows before trusting a calibration.
   A single split can get lucky (or unlucky).
2. **Multiple-comparisons risk is real and this demo only partially corrects for it.**
   `z > 2.0` is a pragmatic middle ground, not a
   rigorous Bonferroni correction. If you search over many lookback
   windows, thresholds, and categories in addition to 10 deciles, your
   effective number of "tests" explodes and you need a much stricter bar
   (or a proper correction) or you will overfit.
3. **Backtest-to-live gap.** Real fills depend on actual order-book depth,
   which this demo doesn't model (it assumes a flat spread cost). Thin
   Polymarket markets can have much wider effective slippage than
   `ASSUMED_SPREAD_COST` accounts for, especially at size.
4. **Daily-aggregated Sharpe can look much better than trade-level Sharpe**
   for a strategy that doesn't trade every day, because zero-return days
   pull down the volatility estimate without reflecting real idle-capital
   risk. This code reports both on purpose — if they diverge a lot, trust
   the trade-level number more, or build a proper capital-utilization-
   adjusted metric.
5. **Sharpe ≥ 2.0 is a genuinely high bar.** It's achievable in niche,
   less-efficient corners of a market (which prediction markets, being
   retail-heavy and less liquid than equities, plausibly are), but it
   should come from a real, mechanistically-understood effect confirmed
   out-of-sample across multiple periods — not from the first backtest
   that clears the number.

Note: I am aiming for around a 1.2-1.5 Sharpe, and only have a 0.6 Sharpe so far with this hypothesis

## Suggested next steps
