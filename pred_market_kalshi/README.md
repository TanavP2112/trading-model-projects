Kalshi Database Credits (June 2021 – January 2026): https://huggingface.co/datasets/TrevorJS/kalshi-trades

Notes:
The paper's specification assumes Gaussian innovations. On Kalshi hourly data we observe realized 1-hour moves with a p99/median ratio of ~24× (compared to ~3.3 under Gaussian), suggesting substantially fatter tails than the paper's specification handles. We therefore also fit the joint model under Student-t innovations with degrees-of-freedom ν estimated per model (fitted jointly for GARCH variants, method-of-moments from training residuals for DR and DR-AS). On the full 4.5-year Kalshi panel, this specification maintains coverage near the nominal 95% while sharpening intervals by ~32% for GARCH+DR-AS, yielding a further ~10-16% Winkler improvement beyond the Gaussian baseline. This is not a criticism of the original paper's specification but an adaptation to the microstructure of the target venue.

Following Xi et al. (2026), we implement the joint structural + GARCH specification with Gaussian innovations on Kalshi hourly data, reproducing their central claim that combining the structural DR-AS model with residual GARCH dynamics dominates plain ARCH/GARCH benchmarks (GARCH+DR-AS volume-weighted Winkler score = 1.132 vs 1.213 for the DR-only baseline). Given that Kalshi's realized 1-hour returns exhibit a p99/median ratio of approximately 24× — substantially fatter tails than a Gaussian innovation can accommodate — we additionally test a Student-t specification, fitting the degrees-of-freedom parameter ν jointly with the GARCH parameters (or empirically from training residuals for the DR/DR-AS variants where no MLE step exists). The Student-t specification reduces the joint model's Winkler score by an additional 27% (0.824 vs 1.132), maintains empirical coverage within 3% of the nominal 95%, and yields uniform wins across all five contract categories (versus the Gaussian specification's wins in only three of five). This suggests the paper's structural insight combines productively with a more realistic innovation distribution.

Statistics of the Dataset:
n bars total: 4,381,176
spread median: 0.0100
spread mean: 0.0121
spread p10 / p25 / p75 / p90: 0.0100 / 0.0100 / 0.0100 / 0.0141
spread == 0.01 (floor): 3,463,597 (79.1%)
spread > 0.05: 43,489 (1.0%)
Correlation(spread, volume): -0.009

## References:

- Weiye Xi, Ciamac C. Moallemi, Mallesh Pai, Shouqiao Wang (2026). Volatility in Prediction Markets: A Structural Approach.
  arXiv, 2607.08199, q-fin.TR, https://arxiv.org/abs/2607.08199
