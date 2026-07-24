# Structural Volatility Modeling & Vol-Normalized Signal Backtests on Kalshi

---

## Repository Structure

```
├── data_fetcher.py       # HF -> DuckDB -> hourly panel, with spread reconstruction
├── volatility_model.py   # DR / DR-AS / GARCH / GARCH+DR-AS, Numba-JIT'd recursion + QMLE fit
├── test1.py              # Phase 1 walk-forward runner (Winkler evaluation)
├── signals.py            # Momentum & reversal signal construction (naive + vol-normalized)
├── backtest.py           # Phase 2 fixed-horizon trade generation + risk suite
├── run_demo.py           # Phase 2 walk-forward runner
├── fees.py                # Kalshi taker fee schedule
├── config.py              # Backtest constants (position sizing, filters)
├── data/                  # Cached panel + walk-forward outputs (parquet/csv)
└── prediction_market_pipeline.ipynb   # End-to-end reproducible notebook with charts
```

---

## Data

The data was constructed from `TrevorJS/kalshi-trades` (HuggingFace), Kalshi's public trade-level feed, aggregated to hourly bars via DuckDB. The dataset contains **4.4M hourly bars across 26,258 markets, from August 2021 – January 2026**, spanning 5 main contract categories (Sports, Economics, Crypto, Entertainment, Politics).

---

## Summary of Results

**Phase 1 — Volatility forecasting:** The paper's joint structural + GARCH specification (`GARCH+DR-AS`) reproduces cleanly: it achieves the lowest volume-weighted Winkler interval score of any model tested, reducing Winkler by **37.8% vs. a plain GARCH(1,1) baseline** and winning **3 of 5 contract categories**, with empirical coverage of ~98% (nominal 95%). Evaluated on 1.73M active out-of-sample contract-hours — roughly double the paper's reported 880k.

**Phase 2 — Trading signal tests:** Vol-normalized momentum shows Sharpe ratios of −8.1 to −3.6 across four fixed holding periods (1/6/12/24h) on ~43k out-of-sample trades. Vol-normalized reversal shows an apparent positive Sharpe initially that a pooled-vs-fold-averaged diagnostic traces to single-event concentration (an early version of the panel showed a pooled Sharpe of +15 at H=6h driven almost entirely by November 2024 election contracts; after tightening the panel to match the paper's filtering conventions, the same signal's Sharpe collapsed to about [-2.53, -1.38]). ~349k out-of-sample trades total across four strategies and four horizons.

---

## Methodology

### Phase 1: Volatility model

Four nested specifications, following the paper's structural decomposition:

| Model         | Description                                                                                           |
| ------------- | ----------------------------------------------------------------------------------------------------- |
| `DR`          | Deadline-resolution baseline: `h² = p(1-p)/τ · Δ` (Wright-Fisher). No fitted parameters.              |
| `DR-AS`       | DR + Glosten-Milgrom adverse-selection term: `h² = DR + K·ν(V)·s²/4`. Fits scalar `K` via OLS.        |
| `GARCH`       | Plain GARCH(1,1), no structural terms. Isolates whether structure adds value over generic clustering. |
| `GARCH+DR-AS` | Full joint specification: structural terms + GARCH residual dynamics, fit jointly via QMLE.           |

**Estimation:** All models fit via **Gaussian quasi-maximum likelihood (QMLE)**, matching the original paper (Appendix B, eq. 17) and Bollerslev & Wooldridge (1992), which is also used in the volatility paper.

**Interval construction:** Rather than parametric Gaussian intervals, we build **asymmetric empirical intervals** from the 2.5%/97.5% quantiles of standardized training residuals per model per fold. This is a legitimate extension the paper's own framing allows — since Appendix B states that "the interval-score evaluation does not require Gaussian standardized innovations" — and is motivated by measured heavy tails in Kalshi returns (active-bar |Δp| has a very VERY large p99/median ratio (~18x) compared to the expectations of Gaussian (which is ~3.8x)).

**Evaluation:** Monthly expanding walk-forward. Following the paper's "Analysis filtering" convention, retain contracts with ≥48 hourly observations and filter to active bars (`|ε| > 1e-10`) at evaluation time.

### Phase 2: Trading signals

Two signals were used: Momentum and Reversal

- `mom_naive` / `mom_vn`: `p_t − p_{t−5}` (5-bar lookback)
- `rev_naive` / `rev_vn`: `p_t − rollmean(p, 24)` (24-bar lookback)

Fixed holding periods (1/6/12/24h), entry threshold at around the 93rd percentile of each signal's own magnitude distribution (equivalent to ~|z|=1.5 for vol-normalized signals), Kalshi taker fees (3.5% each side) — as well as slippage costs — deducted from every trade. A pooled-vs-fold-averaged Sharpe diagnostic is used to detect event-concentration: a large gap between the two indicates the pooled result is driven by a small number of periods rather than persistent edge.

---

## Known Limitations

**Spread reconstruction is relatively weak, and no third-party order-book provider can backfill the full panel:** The public trade feed has no order-book/quote data, so the paper's adverse-selection channel (which depends on time-varying quoted spreads) can't be identified directly. An _effective_ spread is reconstructed from trade aggressor flow (`taker_side`: `min(price | taker=yes) − max(price | taker=no)` per contract-hour, with rolling-median fallback and EWMA smoothing), but ~79% of bars fall through to the minimum-tick floor due to one-sided hourly flow, and the reconstructed spread shows negligible correlation with volume (−0.009, vs. the negative correlation microstructure theory predicts). As a result, `DR-AS` and plain `GARCH` perform almost identically in this reproduction — the AS channel is not sharply identified.

I surveyed five third-party Kalshi order-book providers (Allium, Lychee Data, Dome API, Oddpool, Predexon) as a possible source of real quoted spreads. **None of these API's can substitute for the missing data across this panel's 4.5-year span.** Kalshi's public API has never exposed historical order-book snapshots, so every provider's archive only covers the window they personally started polling in real time — there is no backfill:

| Provider    | Order-book coverage starts                                                                                                                                                                                                                                                                        | Access                               |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| Dome API    | Oct 29, 2025 (~9 months of panel)                                                                                                                                                                                                                                                                 | Free REST API (Limited Data However) |
| Predexon    | Jan 7, 2026 (~6 months of panel)                                                                                                                                                                                                                                                                  | Free REST API (Limited Data However) |
| Oddpool     | Undated ("since we started subscribing")                                                                                                                                                                                                                                                          | Enterprise only                      |
| Allium      | Undated, SQL warehouse                                                                                                                                                                                                                                                                            | Enterprise only                      |
| Lychee Data | Marketed as "since July 2021," but every order-book-specific claim is hedged ("where available") while trade/market claims aren't — the coverage is almost certainly recent-only despite the framing. Also has no API (browser/CSV export only), so it can't feed a scripted pipeline regardless. | Paid, no-code only                   |

A lot of the most complete API's (or databases) are locked behind expensive enterprise-level paywalls that do not make sense for the scope of this project. As such, using the huggingface data was the most appropriate option here.

**Follow-up on API's Explored:** Dome's and Predexon's free tiers do overlap the most recent few months of this panel (Specifically around the end of 2025). A natural next step is pulling real quoted spreads for that overlapping window and directly measuring the correlation with our aggressor-reconstructed spread — turning the current indirect proxies (floor rate, volume correlation) into a quantified validation against ground truth. This wouldn't extend AS identification to the full panel, but it would tell us how much to trust the reconstruction where it's used. Again, **This is not a substitute for the analysis already performed, and the essence of the conclusion still remains consistent regardless**.

**Some niche markets merged with the major markets for the sake of generality:** While Kalshi does contain more niche and less liquid markets, the names of the categories are either very similar or on topic with the main 5 categories. As such, for the sake of generality in our analysis we have merged specific categories together (Elections markets now coincide with the Politics markets). This does not dilute the conclusions of what the paper has produced, and remains consistent with its findings.

**Limitations of Forward-Filling:** The paper's panel is gap-free because Kalshi's continuously-quoted order book gives a mid-quote every hour regardless of trading activity; ours is trade-only, so hours with zero trades have no observation at all. We flag gap-preceded rows (`is_clean_bar`) -- about 53% of consecutive observations are truly one hour apart. In order to combat this, a reindex was performed via forward-filling. This allows us to construct a proper hourly panel. However, this can be complicated when it comes to Sports specifcally, as according to Kalshi themselves, Sports markets officially opened in 2025.

**Last-trade Prices, Not Mid-quotes:** The paper primarily uses close-of-hour mid-quotes; I use last-trade prices from necessity. The paper's own Appendix E confirms the model ranking is preserved under this variant, so this reproduction aligns with that robustness check rather than the primary specification.

**DR Baseline is Pathological in Sports and Economics:** The deadline-resolution variance formula produces very large h² for some near-boundary, short-τ contracts in these categories (Winkler 4.88 and 2.22 respectively), inflating empirical interval widths. This doesn't affect the DR-AS/GARCH/GARCH+DR-AS comparisons, which are the ones that matter f the paper's central claim, but it means DR-relative improvement percentages should be read cautiously.

**Phase 2 is signal-quality diagnostic, not a realizable portfolio backtest:** Trades are sized at a fixed 1% of a $100k bankroll ($1,000 notional each) with no cap on concurrent positions, so at points cumulative exposure exceeds the bankroll many times over. This means the Sharpe / P&L / win-rate metrics are properly interpreted as per-trade signal-quality statistics rather than realizable portfolio returns. An earlier commit of this code also reported max_drawdown computed on an unconstrained equity curve, which produced misleading four-digit negative numbers for the unconditional benchmark; that metric has been dropped and replaced with worst_day_pnl (dollar loss on the worst single day), and turnover is now reported as turnover_pct_per_day (notional traded per day / bankroll) rather than raw shares/day. Converting to a genuine portfolio backtest (concurrent-exposure cap, equity-adjusted sizing, other optimizations) is a separate exercise and hasn't been done here, as the results here show that it is not worth testing.

## References

Xi, W., Moallemi, C. C., Pai, M., & Wang, S. (2026). _Volatility in Prediction Markets: A Structural Approach_. arXiv:2607.08199.
