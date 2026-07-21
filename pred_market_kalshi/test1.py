"""
test1.py — Walk-Forward Benchmark for Structural Volatility Models
Fulfills Xi et al. (2026) paper evaluation across all 4 models:
  1. DR          (Wright-Fisher baseline)
  2. DR-AS       (Wright-Fisher + Glosten-Milgrom adverse selection)
  3. GARCH       (Pure GARCH(1,1) joint QMLE, c=0, K=0)
  4. GARCH+DR-AS (Full structural GARCH joint QMLE, c>0, K>0)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import volatility_model as vm


def compute_winkler_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.05
) -> np.ndarray:
    """Computes bar-by-bar Winkler score for (1 - alpha) prediction intervals."""
    width = upper - lower
    below = (2.0 / alpha) * (lower - y_true) * (y_true < lower)
    above = (2.0 / alpha) * (y_true - upper) * (y_true > upper)
    return width + below + above


def predict_h2_all_models(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    spread_col: str = "spread"
) -> Dict[str, np.ndarray]:
    """
    Computes 1-step ahead conditional variance (h^2) for test_df across all 4 models.
    Preserves time-series lag continuity by running GARCH recursions on concatenated
    (train + test) market sequences before slicing out test predictions.
    """
    # Combine train + test to ensure continuous time-series recursion across market boundaries
    full_df = (
        pd.concat([train_df, test_df], ignore_index=True)
        .sort_values(["market_id", "timestamp"])
        .reset_index(drop=True)
    )
    
    # Track original test indices within full_df
    test_market_timestamps = set(zip(test_df["market_id"], test_df["timestamp"]))
    test_mask = [
        (m, t) in test_market_timestamps 
        for m, t in zip(full_df["market_id"], full_df["timestamp"])
    ]

    p_test = test_df["price"].to_numpy()
    tau_test = test_df["days_to_resolution"].to_numpy()
    vol_test = test_df["volume"].to_numpy()
    spr_test = test_df[spread_col].to_numpy() if spread_col in test_df.columns else np.full_like(p_test, 0.01)

    # ------------------------------------------------------------------
    # MODEL 1: DR (Wright-Fisher Baseline)
    # ------------------------------------------------------------------
    h2_DR = vm.structural_h2(p=p_test, tau=tau_test, volume=None, spread=None, K=0.0)

    # ------------------------------------------------------------------
    # MODEL 2: DR-AS (Structural Model)
    # ------------------------------------------------------------------
    # Fit K via OLS on active updates in training data
    train_eps = train_df.groupby("market_id")["price"].diff().to_numpy()
    active_mask = np.isfinite(train_eps) & (train_eps != 0)

    K_hat = vm.fit_K(
        realized_moves=np.nan_to_num(train_eps, nan=0.0),
        p=train_df["price"].to_numpy(),
        tau=train_df["days_to_resolution"].to_numpy(),
        volume=train_df["volume"].to_numpy(),
        spread=train_df[spread_col].to_numpy() if spread_col in train_df.columns else np.full_like(train_eps, 0.01),
        active_mask=active_mask,
    )

    h2_DR_AS = vm.structural_h2(
        p=p_test, 
        tau=tau_test, 
        volume=vol_test, 
        spread=spr_test, 
        K=K_hat
    )

    # ------------------------------------------------------------------
    # MODEL 3: Plain GARCH(1,1) (Constrained c=0, K=0)
    # ------------------------------------------------------------------
    garch_params = vm.fit_garch_dr_as_joint(
        train_df, spread_col=spread_col, constrain_c_zero=True
    )
    full_h2_garch = vm.garch_dr_as_h2(full_df, garch_params, spread_col=spread_col)
    h2_GARCH = full_h2_garch.to_numpy()[test_mask]

    # ------------------------------------------------------------------
    # MODEL 4: GARCH + DR-AS (Full Joint Structural Model)
    # ------------------------------------------------------------------
    garch_as_params = vm.fit_garch_dr_as_joint(
        train_df, spread_col=spread_col, constrain_c_zero=False
    )
    full_h2_garch_as = vm.garch_dr_as_h2(full_df, garch_as_params, spread_col=spread_col)
    h2_GARCH_AS = full_h2_garch_as.to_numpy()[test_mask]
    # print(f"[Debug] Estimated K = {K_hat:.6f} | Spread mean = {test_df['spread'].mean():.4f}") -> debug for K values

    # Apply global numerical floor safeguard
    return {
        "DR": np.clip(h2_DR, 1e-8, None),
        "DR-AS": np.clip(h2_DR_AS, 1e-8, None),
        "GARCH": np.clip(h2_GARCH, 1e-8, None),
        "GARCH+DR-AS": np.clip(h2_GARCH_AS, 1e-8, None),
    }


def evaluate_walk_forward(
    df: pd.DataFrame, 
    spread_col: str = "spread", 
    z_95: float = 1.96
) -> pd.DataFrame:
    """Runs expanding-window walk-forward benchmark across available markets."""
    df = df.copy()
    
    # 1. Parse timestamps safely (handling Unix epoch seconds vs ms vs string datetimes)
    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        # Check if timestamps are in milliseconds (> 1e11) vs seconds
        unit = "ms" if df["timestamp"].iloc[0] > 1e11 else "s"
        df["datetime"] = pd.to_datetime(df["timestamp"], unit=unit, errors="coerce")
    else:
        df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Drop any unparseable rows
    df = df.dropna(subset=["datetime"]).sort_values(["market_id", "datetime"]).reset_index(drop=True)

    # 2. Try Monthly splits first; fall back to Weekly or 14-day splits if span is short
    df["period"] = df["datetime"].dt.to_period("M")
    unique_periods = sorted(df["period"].unique())

    # Fallback to Weekly windows if dataset spans less than 2 distinct months
    if len(unique_periods) < 2:
        print("[Notice] Dataset spans less than 2 full calendar months. Falling back to weekly evaluation splits...")
        df["period"] = df["datetime"].dt.to_period("W")
        unique_periods = sorted(df["period"].unique())

    if len(unique_periods) < 2:
        raise ValueError(
            f"Dataset span is too short for walk-forward evaluation. "
            f"Found only {len(unique_periods)} period(s) from {df['datetime'].min()} to {df['datetime'].max()}."
        )

    eval_records = []

    # Expanding window: train on < T, evaluate on period T
    for i in range(1, len(unique_periods)):
        eval_period = unique_periods[i]
        train_df = df[df["period"] < eval_period].copy()
        test_df = df[df["period"] == eval_period].copy()

        if len(train_df) < 50 or len(test_df) == 0:
            continue

        print(
            f"[Walk-Forward] Evaluating period {eval_period} | "
            f"Train: {len(train_df):,} bars | Test: {len(test_df):,} bars..."
        )

        # Get actual 1-bar price changes
        test_df["actual_dp"] = test_df.groupby("market_id")["price"].diff().fillna(0.0)
        actual_dp = test_df["actual_dp"].to_numpy()

        # Compute predictions across all 4 models
        h2_preds = predict_h2_all_models(train_df, test_df, spread_col=spread_col)

        for model_name, h2_arr in h2_preds.items():
            sigma = np.sqrt(h2_arr)
            lower = -z_95 * sigma
            upper = z_95 * sigma
            winkler = compute_winkler_score(actual_dp, lower, upper)

            for idx, (_, row) in enumerate(test_df.iterrows()):
                eval_records.append({
                    "timestamp": row["datetime"],
                    "market_id": row["market_id"],
                    "category": row.get("category", "Uncategorized"),
                    "volume": row.get("volume", 0.0),
                    "model": model_name,
                    "actual_dp": actual_dp[idx],
                    "lower_95": lower[idx],
                    "upper_95": upper[idx],
                    "winkler_score": winkler[idx],
                })

    return pd.DataFrame(eval_records)


def summarize_results(eval_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates Overall and Category-Level summary metrics cleanly handling 0-volume edge cases."""
    
    def compute_metrics(group: pd.DataFrame) -> pd.Series:
        vol_sum = group["volume"].sum()
        winkler = group["winkler_score"].to_numpy()
        
        # Volume-Weighted Winkler Score fallback
        if vol_sum > 0 and not np.isnan(vol_sum):
            vw_winkler = float(np.average(winkler, weights=group["volume"]))
        else:
            vw_winkler = float(np.mean(winkler))

        in_bounds = (group["actual_dp"] >= group["lower_95"]) & (group["actual_dp"] <= group["upper_95"])
        coverage = float(in_bounds.mean())
        avg_width = float((group["upper_95"] - group["lower_95"]).mean())

        return pd.Series({
            "Volume-Weighted Winkler Score": vw_winkler,
            "Empirical Coverage Rate": coverage,
            "Average Interval Width": avg_width,
            "Forecast Bars": len(group)
        })

    # Overall Summary
    overall_summary = (
        eval_df.groupby("model", group_keys=False)
        .apply(compute_metrics)
        .reindex(["DR", "DR-AS", "GARCH", "GARCH+DR-AS"])
    )

    # Category Breakdown Summary
    category_summary = (
        eval_df.groupby(["category", "model"], group_keys=False)
        .apply(compute_metrics)
    )

    return overall_summary, category_summary


if __name__ == "__main__":
    # Load standardized panel data
    # Ensure 'data.parquet' or 'data.csv' exists with required columns: 
    # ['market_id', 'timestamp', 'price', 'days_to_resolution', 'volume', 'spread', 'category']
    data_path = "kalshi_hf_panel.parquet"
    
    try:
        df = pd.read_parquet(data_path)
    except Exception:
        print(f"Loading CSV fallback for {data_path}...")
        df = pd.read_csv("panel_data.csv")

    print(f"Successfully loaded dataset with {len(df):,} total bars.")

    # Execute benchmark
    eval_df = evaluate_walk_forward(df, spread_col="spread")

    # Display Summaries
    overall_sum, category_sum = summarize_results(eval_df)

    print("\n" + "=" * 70)
    print("OVERALL BENCHMARK SUMMARY")
    print("=" * 70)
    print(overall_sum.to_string())

    print("\n" + "=" * 70)
    print("CATEGORY-LEVEL BREAKDOWN")
    print("=" * 70)
    print(category_sum.to_string())


# import numpy as np
# import pandas as pd
# from typing import Dict, List, Any

# from volatility_model import (
#     structural_h2, fit_K, fit_garch_dr_as_joint, garch_dr_as_h2,
#     DEFAULT_BAR_LENGTH,
# )
# from forecasting.metrics import evaluate_prediction_intervals
# from data_fetcher import build_market_panel

# BAR_LENGTH = DEFAULT_BAR_LENGTH  # 1 hour in days
# Z_95 = 1.96
# MODELS = ["DR", "DR-AS", "GARCH", "GARCH+DR-AS"]


# def _predict_h2(model_type: str, train_df: pd.DataFrame, test_df: pd.DataFrame,
#                 spread_col: str | None) -> tuple[np.ndarray, Dict[str, float]]:
#     """
#     Fit `model_type` on train_df, return (h2 forecast on test_df, fitted-param
#     dict for reporting). All fitting uses TRAIN only; test_df is untouched
#     except to read its p/tau/volume/spread features for the forecast.
#     """
#     if model_type == "DR":
#         h2 = structural_h2(
#             test_df["price"].values, test_df["days_to_resolution"].values,
#             K=0.0, bar_length=BAR_LENGTH,
#         )
#         return h2, {"K": 0.0}

#     if model_type == "DR-AS":
#         active = (train_df.groupby("market_id")["price"].diff() != 0).to_numpy()
#         realized_sq = (train_df.groupby("market_id")["price"].diff() ** 2).to_numpy()
#         K = fit_K(
#             realized_sq_moves=realized_sq,
#             p=train_df["price"].to_numpy(),
#             tau=train_df["days_to_resolution"].to_numpy(),
#             volume=train_df["volume"].to_numpy(),
#             spread=train_df[spread_col].to_numpy() if spread_col else np.full(len(train_df), 0.01),
#             active_mask=active,
#             bar_length=BAR_LENGTH,
#         )
#         h2 = structural_h2(
#             test_df["price"].values, test_df["days_to_resolution"].values,
#             volume=test_df["volume"].values,
#             spread=test_df[spread_col].values if spread_col else np.full(len(test_df), 0.01),
#             K=K, bar_length=BAR_LENGTH,
#         )
#         return h2, {"K": K}

#     if model_type == "GARCH":
#         params = fit_garch_dr_as_joint(train_df, spread_col=spread_col,
#                                        bar_length=BAR_LENGTH, constrain_c_zero=True)
#         h2 = garch_dr_as_h2(test_df, params=params, spread_col=spread_col,
#                             bar_length=BAR_LENGTH).values
#         return h2, params

#     if model_type == "GARCH+DR-AS":
#         params = fit_garch_dr_as_joint(train_df, spread_col=spread_col,
#                                        bar_length=BAR_LENGTH, constrain_c_zero=False)
#         h2 = garch_dr_as_h2(test_df, params=params, spread_col=spread_col,
#                             bar_length=BAR_LENGTH).values
#         return h2, params

#     raise ValueError(f"Unknown model_type: {model_type}")


# def run_walk_forward(df: pd.DataFrame, model_type: str,
#                      spread_col: str | None = "spread", verbose: bool = False) -> pd.DataFrame:
#     """
#     Expanding-window monthly walk-forward. For each month T (from the 2nd on),
#     fit on all months < T, forecast month T. Returns concatenated per-bar
#     forecasts with actual next-hour move, 95% interval bounds, and volume.
#     """
#     df = df.sort_values(["market_id", "timestamp"]).copy()
#     _ts = df["timestamp"]
#     if getattr(_ts.dt, "tz", None) is not None:
#         _ts = _ts.dt.tz_localize(None)
#     df["month"] = _ts.dt.to_period("M")
#     months = sorted(df["month"].unique())
#     if len(months) < 2:
#         raise ValueError(f"Need >=2 distinct months; got {len(months)}. "
#                          f"Fetch a longer history (build_market_panel pulls ~90d/market).")

#     out_frames = []
#     for i in range(1, len(months)):
#         train_df = df[df["month"].isin(months[:i])].copy()
#         test_df = df[df["month"] == months[i]].copy()
#         if train_df.empty or test_df.empty:
#             continue

#         # Realized next-hour move is the forecast TARGET.
#         test_df["actual_dp"] = test_df.groupby("market_id")["price"].diff().shift(-1)

#         h2_pred, params = _predict_h2(model_type, train_df, test_df, spread_col)
#         test_df["pred_h"] = np.sqrt(np.maximum(h2_pred, 1e-12))
#         test_df["lower_95"] = -Z_95 * test_df["pred_h"]
#         test_df["upper_95"] = Z_95 * test_df["pred_h"]

#         # Active, non-null evaluation bars only (a flat bar carries no vol info).
#         eval_df = test_df.dropna(subset=["actual_dp", "pred_h"]).copy()
#         eval_df = eval_df[eval_df["actual_dp"] != 0]
#         if verbose:
#             kbits = " ".join(f"{k}={v:.4f}" for k, v in params.items()
#                              if isinstance(v, (int, float)))
#             print(f"    [{model_type:11s}] test {months[i]}  n={len(eval_df):5d}  {kbits}")
#         out_frames.append(eval_df)

#     if not out_frames:
#         return pd.DataFrame()
#     return pd.concat(out_frames, ignore_index=True)


# def report_spread_quality(df: pd.DataFrame) -> None:
#     # """Print how much of the panel carries a REAL reconstructed spread and its
#     # variation -- the context needed to interpret a DR vs DR-AS result."""
#     # print("\n--- Spread quality (context for interpreting DR vs DR-AS) ---")
#     # if "spread_is_real" in df.columns:
#     #     frac_real = df["spread_is_real"].mean()
#     #     print(f"  Bars with a REAL reconstructed spread: {frac_real:.1%} "
#     #           f"(rest use the one-tick fallback)")
#     # if "spread" in df.columns:
#     #     s = df["spread"]
#     #     print(f"  Spread: median={s.median():.4f}  std={s.std():.4f}  "
#     #           f"p25={s.quantile(.25):.4f}  p75={s.quantile(.75):.4f}")
#     #     if s.std() < 1e-3:
#     #         print("  !! Spread variation is near-zero -- DR-AS cannot meaningfully differ "
#     #               "from DR here (no spread signal to drive K). A DR~=DR-AS result would "
#     #               "reflect the DATA, not a failure of the AS term.")
#     pass


# def main():
#     print("=" * 70)
#     print("VOLATILITY-FORECASTING BENCHMARK")
#     print("=" * 70)

#     # Real-spread panel from liquid markets. build_market_panel now attaches
#     # reconstructed effective spreads (use_real_spreads=True) by default.
#     print("[1/3] Building real-spread panel from resolved Kalshi markets...")
#     df = build_market_panel(limit_markets=90, use_real_spreads=True)
#     if df.empty:
#         print("[Error] Empty panel; cannot run benchmark.")
#         return
#     df.to_parquet("data/kalshi_phase1_panel.parquet")
#     print(f"      {len(df)} bars across {df['market_id'].nunique()} markets, "
#           f"{df['timestamp'].dt.to_period('M').nunique()} months.")

#     report_spread_quality(df)
#     spread_col = "spread" if "spread" in df.columns else None

#     print("\n[2/3] Running expanding monthly walk-forward for all four models...")
#     summary = []
#     for model in MODELS:
#         forecasts = run_walk_forward(df, model, spread_col=spread_col, verbose=True)
#         if forecasts.empty:
#             print(f"  [{model}] produced no evaluable forecasts; skipping.")
#             continue
#         m = evaluate_prediction_intervals(
#             y_true=forecasts["actual_dp"].values,
#             lower_bounds=forecasts["lower_95"].values,
#             upper_bounds=forecasts["upper_95"].values,
#             volume_weights=forecasts["volume"].values,
#             alpha=0.05,
#         )
#         m["Model"] = model
#         m["Forecast Bars"] = len(forecasts)
#         summary.append(m)

#     print("\n[3/3] Results (lower Winkler = better, 95% Confidence):")
#     print("=" * 70)
#     if not summary:
#         print("No model produced forecasts -- likely too few months. Fetch more history.")
#         return
#     res = pd.DataFrame(summary).set_index("Model")
#     cols = ["Volume-Weighted Winkler Score", "Empirical Coverage Rate",
#             "Average Interval Width", "Forecast Bars"]
#     print(res[cols].to_string())


# if __name__ == "__main__":
#     main()