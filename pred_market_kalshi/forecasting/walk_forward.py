"""
Expanding-Window Monthly Walk-Forward Engine for Volatility Forecasts
"""
import pandas as pd
import numpy as np
from typing import Dict, Any

def monthly_expanding_walk_forward(df: pd.DataFrame, model_class) -> pd.DataFrame:
    """
    Evaluates forecasting models using an expanding window by month.
    Fits parameters on all data prior to month T, predicts on month T.
    """
    df = df.sort_values("timestamp").copy()
    df["month"] = df["timestamp"].dt.to_period("M")
    months = df["month"].unique()

    forecast_results = []

    # Requires at least 2 prior months to establish an initial training window
    for i in range(2, len(months)):
        train_months = months[:i]
        test_month = months[i]

        train_df = df[df["month"].isin(train_months)]
        test_df = df[df["month"] == test_month]

        if test_df.empty or train_df.empty:
            continue

        # Fit model parameters strictly on past observations
        model = model_class()
        model.fit(train_df)

        # Generate 1-hour probability volatility forecasts
        preds = model.predict_volatility(test_df)
        
        # Calculate 95% Prediction Intervals (z = 1.96 for normal errors)
        z = 1.96
        test_df = test_df.copy()
        test_df["pred_vol"] = preds
        test_df["lower_95"] = -z * test_df["pred_vol"]
        test_df["upper_95"] = z * test_df["pred_vol"]
        test_df["actual_dp"] = test_df.groupby("market_id")["price"].diff().shift(-1)

        forecast_results.append(test_df.dropna(subset=["actual_dp", "pred_vol"]))

    return pd.concat(forecast_results, ignore_index=True)