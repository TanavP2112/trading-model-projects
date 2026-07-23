import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
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
    spread_col: str = "spread",
    dist: str = "empirical",
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
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
    full_df["_id_ts"] = full_df["market_id"].astype(str) + "_" + full_df["timestamp"].astype(str)
    test_set = set(test_df["market_id"].astype(str) + "_" + test_df["timestamp"].astype(str))
    test_mask = full_df["_id_ts"].isin(test_set).to_numpy()

    p_test = test_df["price"].to_numpy()
    tau_test = test_df["days_to_resolution"].to_numpy()
    vol_test = test_df["volume"].to_numpy()
    spr_test = test_df[spread_col].to_numpy() if spread_col in test_df.columns else np.full_like(p_test, 0.01)

    # MODEL 1: DR (Wright-Fisher Baseline)
    h2_DR = vm.structural_h2(p=p_test, tau=tau_test, volume=None, spread=None, K=0.0)

    # MODEL 2: DR-AS (Structural Model)
    train_eps = train_df.groupby("market_id")["price"].diff().to_numpy()
    if "is_clean_bar" in train_df.columns:
        clean = train_df["is_clean_bar"].fillna(False).to_numpy()
    else:
        clean = np.ones(len(train_df), dtype=bool)
    active_mask = np.isfinite(train_eps) & (np.abs(train_eps) > 1e-10) & clean

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

    # MODEL 3: Plain GARCH(1,1) (Constrained c=0, K=0)
    garch_params = vm.fit_garch_dr_as_joint(
        train_df, spread_col=spread_col, constrain_c_zero=True,
    )
    full_h2_garch = vm.garch_dr_as_h2(full_df, garch_params, spread_col=spread_col)
    h2_GARCH = full_h2_garch.to_numpy()[test_mask]

    # MODEL 4: GARCH + DR-AS (Full Joint Structural Model)
    garch_as_params = vm.fit_garch_dr_as_joint(
        train_df, spread_col=spread_col, constrain_c_zero=False,
    )
    full_h2_garch_as = vm.garch_dr_as_h2(full_df, garch_as_params, spread_col=spread_col)
    h2_GARCH_AS = full_h2_garch_as.to_numpy()[test_mask]

    # Apply global numerical floor safeguard
    h2_dict = {
        "DR": np.clip(h2_DR, 1e-8, None),
        "DR-AS": np.clip(h2_DR_AS, 1e-8, None),
        "GARCH": np.clip(h2_GARCH, 1e-8, None),
        "GARCH+DR-AS": np.clip(h2_GARCH_AS, 1e-8, None),
    }

    p_train = train_df["price"].to_numpy()
    tau_train = train_df["days_to_resolution"].to_numpy()
    vol_train = train_df["volume"].to_numpy()
    spr_train = (train_df[spread_col].to_numpy()
                 if spread_col in train_df.columns
                 else np.full_like(p_train, 0.01))
    train_eps_full = train_df.groupby("market_id")["price"].diff().to_numpy()
    valid_train = np.isfinite(train_eps_full) & (train_eps_full != 0)

    # Structural h^2 on train for DR / DR-AS
    h2_train_DR = vm.structural_h2(p=p_train, tau=tau_train, volume=None, spread=None, K=0.0)
    h2_train_DR_AS = vm.structural_h2(
        p=p_train, tau=tau_train, volume=vol_train, spread=spr_train, K=K_hat
    )

    # Joint h^2 on train for GARCH / GARCH+DR-AS (uses the fitted joint params)
    h2_train_GARCH = vm.garch_dr_as_h2(train_df, garch_params, spread_col=spread_col).to_numpy()
    h2_train_GARCH_AS = vm.garch_dr_as_h2(train_df, garch_as_params, spread_col=spread_col).to_numpy()

    def _empirical_quantiles(train_eps, train_h2, valid, q_clip=20.0):
        """Return (q_lo, q_hi) at 2.5% and 97.5% of z = eps/sqrt(h^2) on active train bars.
        """
        # Extend the valid mask to also require h2 to be finite and positive
        valid_h2 = valid & np.isfinite(train_h2) & (train_h2 > 0)
        if valid_h2.sum() < 40:  # Below this the quantiles are too noisy
            return -1.96, 1.96
        z = train_eps[valid_h2] / np.sqrt(train_h2[valid_h2])
        z = z[np.isfinite(z)]  # Belt and suspenders
        if len(z) < 40:
            return -1.96, 1.96
        q_lo = float(np.quantile(z, 0.025))
        q_hi = float(np.quantile(z, 0.975))
        # Clip to sane range; a single outlier training residual could otherwise produce absurd intervals
        q_lo = max(q_lo, -q_clip)
        q_hi = min(q_hi, q_clip)
        return q_lo, q_hi

    if dist == "empirical":
        method = "empirical"
        q_DR = _empirical_quantiles(train_eps_full, h2_train_DR, valid_train)
        q_DR_AS = _empirical_quantiles(train_eps_full, h2_train_DR_AS, valid_train)
        q_GARCH = _empirical_quantiles(train_eps_full, h2_train_GARCH, valid_train)
        q_GARCH_AS = _empirical_quantiles(train_eps_full, h2_train_GARCH_AS, valid_train)
        interval_params = {
            "DR":          {"q_lo": q_DR[0],       "q_hi": q_DR[1],       "method": method},
            "DR-AS":       {"q_lo": q_DR_AS[0],    "q_hi": q_DR_AS[1],    "method": method},
            "GARCH":       {"q_lo": q_GARCH[0],    "q_hi": q_GARCH[1],    "method": method},
            "GARCH+DR-AS": {"q_lo": q_GARCH_AS[0], "q_hi": q_GARCH_AS[1], "method": method},
        }
    else:
        interval_params = {name: {"method": "gaussian"} for name in h2_dict}

    return h2_dict, interval_params


def evaluate_walk_forward(
    df: pd.DataFrame,
    spread_col: str = "spread",
    z_95: float = 1.96,
    lookback_periods: Optional[int] = None,
    dist: str = "empirical",
) -> pd.DataFrame:
    """
    Runs rolling-window walk-forward benchmark across available markets.

    dist : {"gaussian", "empirical"}
    """
    df = df.copy()
    
    # 1. Parse timestamps safely
    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        unit = "ms" if df["timestamp"].iloc[0] > 1e11 else "s"
        df["datetime"] = pd.to_datetime(df["timestamp"], unit=unit, errors="coerce")
    else:
        df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Drop any unparseable rows and sort
    df = df.dropna(subset=["datetime"]).sort_values(["market_id", "datetime"]).reset_index(drop=True)

    # 2. Pre-calculate 1-bar price changes (actual_dp) vector-wide to avoid loop overhead
    df["actual_dp"] = df.groupby("market_id")["price"].diff().shift(-1).fillna(0.0)

    # 3. Try Monthly splits first; fall back to Weekly if span is short
    df["period"] = df["datetime"].dt.to_period("M")
    unique_periods = sorted(df["period"].unique())

    if len(unique_periods) < 2:
        print("[Notice] Dataset spans less than 2 full calendar months. Falling back to weekly evaluation splits...")
        df["period"] = df["datetime"].dt.to_period("W")
        unique_periods = sorted(df["period"].unique())

    if len(unique_periods) < 2:
        raise ValueError(
            f"Dataset span is too short for walk-forward evaluation. "
            f"Found only {len(unique_periods)} period(s) from {df['datetime'].min()} to {df['datetime'].max()}."
        )

    # 4. Optimization: Filter to only keep markets with >= 24 bars total across the panel
    # to avoid trivial cross-sections that get filtered out anyway.
    market_counts = df.groupby("market_id").size()
    valid_markets = market_counts[market_counts >= 24].index
    df = df[df["market_id"].isin(valid_markets)].copy()

    eval_records = []
    for i in range(1, len(unique_periods)):
        eval_period = unique_periods[i]
        if lookback_periods is None:
            start_idx = 0
        else:
            start_idx = max(0, i - lookback_periods)
        valid_training_periods = unique_periods[start_idx:i]
        
        train_df = df[df["period"].isin(valid_training_periods)]
        test_df = df[df["period"] == eval_period]

        if len(train_df) < 50 or len(test_df) == 0:
            continue

        print(
            f"[Walk-Forward] Evaluating period {eval_period} | "
            f"Train: {len(train_df):,} bars | Test: {len(test_df):,} bars..."
        )

        actual_dp = test_df["actual_dp"].to_numpy()
        h2_preds, interval_params = predict_h2_all_models(train_df, test_df,
                                                          spread_col=spread_col, dist=dist)

        timestamps = test_df["datetime"].to_numpy()
        market_ids = test_df["market_id"].to_numpy()
        categories = test_df["category"].to_numpy() if "category" in test_df.columns else np.array(["Uncategorized"] * len(test_df))
        volumes = test_df["volume"].to_numpy() if "volume" in test_df.columns else np.zeros(len(test_df))

        if "is_clean_bar" in test_df.columns:
            is_clean = test_df["is_clean_bar"].fillna(False).to_numpy()
        else:
            is_clean = np.ones(len(test_df), dtype=bool)

        for model_name, h2_arr in h2_preds.items():
            sigma = np.sqrt(h2_arr)
            params = interval_params[model_name]
            method = params.get("method", "gaussian")

            if method == "empirical":
                lower = params["q_lo"] * sigma
                upper = params["q_hi"] * sigma
            else:
                lower = -z_95 * sigma
                upper = z_95 * sigma

            winkler = compute_winkler_score(actual_dp, lower, upper)
            for j in range(len(test_df)):
                eval_records.append({
                    "timestamp": timestamps[j],
                    "market_id": market_ids[j],
                    "category": categories[j],
                    "volume": volumes[j],
                    "model": model_name,
                    "actual_dp": actual_dp[j],
                    "lower_95": lower[j],
                    "upper_95": upper[j],
                    "winkler_score": winkler[j],
                    "is_clean_bar": bool(is_clean[j]),
                })

    return pd.DataFrame(eval_records)


def summarize_results(
    eval_df: pd.DataFrame,
    active_only: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates Overall and Category-Level summary metrics.
    """
    if active_only:
        clean = eval_df.get("is_clean_bar", pd.Series(True, index=eval_df.index))
        active = clean.astype(bool) & (np.abs(eval_df["actual_dp"]) > 1e-10)
        n_dropped = int((~active).sum())
        if n_dropped > 0:
            print(f"[summarize_results] active_only=True -> "
                  f"dropping {n_dropped:,} inactive/gap-preceded bars "
                  f"({100*n_dropped/len(eval_df):.1f}% of eval rows).")
        eval_df = eval_df[active].copy()

    def compute_metrics(group: pd.DataFrame) -> pd.Series:
        vol_sum = group["volume"].sum()
        winkler = group["winkler_score"].to_numpy()

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
    data_path = "data/kalshi_hf_panel.parquet"
    
    try:
        df = pd.read_parquet(data_path)
    except Exception:
        print(f"Loading CSV fallback for {data_path}...")
        df = pd.read_csv("panel_data.csv")

    print(f"Successfully loaded dataset with {len(df):,} total bars.")
    import sys
    dist_arg = sys.argv[1] if len(sys.argv) > 1 else "empirical"
    print(f"[Info] Running with dist={dist_arg!r}")
    eval_df = evaluate_walk_forward(df, spread_col="spread", dist=dist_arg)

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