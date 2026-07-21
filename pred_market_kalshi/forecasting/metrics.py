import numpy as np
import pandas as pd
from typing import Dict

def evaluate_prediction_intervals(
    y_true: np.ndarray, 
    lower_bounds: np.ndarray, 
    upper_bounds: np.ndarray, 
    volume_weights: np.ndarray = None,
    alpha: float = 0.05
) -> Dict[str, float]:
    
    # 1. Handle empty inputs gracefully
    if len(y_true) == 0:
        return {
            "Volume-Weighted Winkler Score": np.nan,
            "Empirical Coverage Rate": np.nan,
            "Average Interval Width": np.nan,
        }

    # Convert to numpy arrays if passed as pandas Series
    y_true = np.asarray(y_true, dtype=float)
    lower_bounds = np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)

    if volume_weights is None:
        volume_weights = np.ones_like(y_true, dtype=float)
    else:
        volume_weights = np.asarray(volume_weights, dtype=float)

    # 2. Compute interval widths and penalties
    width = upper_bounds - lower_bounds
    
    # Penalize realizations outside the interval
    below_penalty = (2.0 / alpha) * (lower_bounds - y_true) * (y_true < lower_bounds)
    above_penalty = (2.0 / alpha) * (y_true - upper_bounds) * (y_true > upper_bounds)
    
    winkler_scores = width + below_penalty + above_penalty

    # 3. Safe Volume-Weighted Average Calculation
    total_volume = np.sum(volume_weights)

    if total_volume > 0 and not np.isnan(total_volume):
        # Normal volume-weighted average
        vw_winkler = np.average(winkler_scores, weights=volume_weights)
    else:
        # Fallback to simple unweighted mean if total volume is 0 or NaN
        vw_winkler = np.mean(winkler_scores)

    # 4. Standard metrics
    coverage = np.mean((y_true >= lower_bounds) & (y_true <= upper_bounds))
    avg_width = np.mean(width)

    return {
        "Volume-Weighted Winkler Score": float(vw_winkler),
        "Empirical Coverage Rate": float(coverage),
        "Average Interval Width": float(avg_width),
    }