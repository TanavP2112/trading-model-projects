import numpy as np
import pandas as pd
from typing import Dict

from volatility_model import (
    structural_h2, fit_K, fit_garch_dr_as_joint, garch_dr_as_h2,
    DEFAULT_BAR_LENGTH,
)
DEFAULT_MOM_LOOKBACK = 5   # (5 hours on the hourly panel)
DEFAULT_REV_LOOKBACK = 24  # (1 day on the hourly panel)

# The `_BOOTSTRAP_WINNERS` map below is used ONLY when no saved Phase 1
# output exists, this is from MY CURRENT AND MOST RECENT RUN. if this is
# fallen back to, something is wrong with the code
_BOOTSTRAP_WINNERS: Dict[str, str] = {
    "Crypto": "GARCH+DR-AS",
    "Sports": "GARCH+DR-AS",
    "Economics": "GARCH+DR-AS",
    "Politics": "GARCH+DR-AS",
    "Entertainment": "GARCH+DR-AS",
}


def get_phase1_winners(
    by_category_csv_path: str = "data/walk_forward/wf_by_category.csv",
    score_col: str = "Volume-Weighted Winkler Score",
    verbose: bool = True,
) -> Dict[str, str]:
    import os
    if not os.path.exists(by_category_csv_path):
        if verbose:
            print(f"[signals] Phase 1 output not found at {by_category_csv_path}; "
                  f"using bootstrap fallback (rerun Phase 1 for authoritative winners).")
        return dict(_BOOTSTRAP_WINNERS)

    df = pd.read_csv(by_category_csv_path)
    if "category" not in df.columns:
        # Saved with (category, model) as index; reload with index_col
        df = pd.read_csv(by_category_csv_path, index_col=[0, 1]).reset_index()
    winners = (
        df.loc[df.groupby("category")[score_col].idxmin()]
          .set_index("category")["model"]
          .to_dict()
    )
    if verbose:
        print(f"[signals] Loaded Phase 1 winners from {by_category_csv_path}: {winners}")
    return winners
PHASE1_WINNER_BY_CATEGORY: Dict[str, str] = _BOOTSTRAP_WINNERS


def load_phase1_winners_from_csv(*args, **kwargs) -> Dict[str, str]:
    """Deprecated alias for get_phase1_winners(). Kept for backwards compat."""
    return get_phase1_winners(*args, **kwargs)
def attach_h2(
    df: pd.DataFrame,
    train_mask: pd.Series,
    model: str | Dict[str, str] | None = None,
    spread_col: str = "spread",
) -> pd.DataFrame:
    """
    model options:
        "DR"           -- deadline-resolution only (no fit needed)
        "DR-AS"        -- fits K on train via OLS
        "GARCH"        -- joint plain GARCH (c=0, K=0)
        "GARCH+DR-AS"  -- full joint model
    """
    # Resolve default: dynamically load Phase 1 winners
    if model is None:
        model = get_phase1_winners()

    if isinstance(model, dict):
        return _attach_h2_by_category(df, train_mask, model_map=model, spread_col=spread_col)

    df = df.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    train_df = df[train_mask.values].copy()

    if model == "DR":
        h2 = structural_h2(df["price"].values, df["days_to_resolution"].values,
                            K=0.0, bar_length=DEFAULT_BAR_LENGTH)

    elif model == "DR-AS":
        train_eps = train_df.groupby("market_id")["price"].diff().to_numpy()
        active = np.isfinite(train_eps) & (train_eps != 0)
        K_hat = fit_K(
            realized_moves=np.nan_to_num(train_eps, nan=0.0),
            p=train_df["price"].to_numpy(),
            tau=train_df["days_to_resolution"].to_numpy(),
            volume=train_df["volume"].to_numpy(),
            spread=train_df[spread_col].to_numpy(),
            active_mask=active,
        )
        h2 = structural_h2(
            p=df["price"].values, tau=df["days_to_resolution"].values,
            volume=df["volume"].values, spread=df[spread_col].values,
            K=K_hat, bar_length=DEFAULT_BAR_LENGTH,
        )

    elif model in ("GARCH", "GARCH+DR-AS"):
        params = fit_garch_dr_as_joint(
            train_df, spread_col=spread_col,
            constrain_c_zero=(model == "GARCH"),
        )
        h2 = garch_dr_as_h2(df, params=params, spread_col=spread_col).to_numpy()

    else:
        raise ValueError(f"unknown model: {model!r}")

    df["h2"] = np.clip(h2, 1e-12, None)
    df["h"] = np.sqrt(df["h2"])
    return df


def _attach_h2_by_category(
    df: pd.DataFrame,
    train_mask: pd.Series,
    model_map: Dict[str, str],
    spread_col: str = "spread",
    default_model: str = "GARCH+DR-AS",
) -> pd.DataFrame:
    if "category" not in df.columns:
        raise ValueError("model dict passed but df has no 'category' column")

    MIN_TRAIN_BARS_FOR_GARCH = 500  # below this, GARCH fits are unreliable
    frames = []
    for cat, sub in df.groupby("category", sort=False):
        chosen = model_map.get(cat, default_model)
        cat_train_mask = train_mask.loc[sub.index]
        n_train = int(cat_train_mask.sum())
        if "GARCH" in chosen and n_train < MIN_TRAIN_BARS_FOR_GARCH:
            print(f"[attach_h2/{cat}] only {n_train} train bars -- "
                  f"falling back from {chosen} to DR-AS (GARCH needs more data).")
            chosen = "DR-AS"
        sub_out = attach_h2(sub, train_mask=cat_train_mask, model=chosen, spread_col=spread_col)
        sub_out["_h2_model"] = chosen  # traceable which model produced each row's h2
        frames.append(sub_out)

    return pd.concat(frames).sort_values(["market_id", "timestamp"]).reset_index(drop=True)

def momentum_naive(df: pd.DataFrame, lookback: int = DEFAULT_MOM_LOOKBACK) -> pd.Series:
    """
    Raw momentum: p_t - p_{t-lookback}, per market. Positive -> recent up-move.
    Uses only past bars <= t, so it's leakage-safe.
    """
    return df.groupby("market_id")["price"].diff(lookback)


def momentum_vol_normalized(df: pd.DataFrame, lookback: int = DEFAULT_MOM_LOOKBACK) -> pd.Series:
    """
    Vol-normalized momentum: (p_t - p_{t-lookback}) / sqrt(sum_{s=t-lookback+1..t} h_s^2).
    """
    raw = momentum_naive(df, lookback)
    var_lb = (
        df.groupby("market_id")["h2"]
        .transform(lambda s: s.rolling(lookback, min_periods=lookback).sum())
    )
    return raw / np.sqrt(np.clip(var_lb.values, 1e-12, None))


def reversal_naive(df: pd.DataFrame, lookback: int = DEFAULT_REV_LOOKBACK) -> pd.Series:
    """
    Raw reversal: p_t - rolling_mean(p, lookback). Positive -> above recent
    average. The trading interpretation FLIPS the sign (bet AGAINST the deviation).
    """
    roll_mean = (
        df.groupby("market_id")["price"]
        .transform(lambda s: s.rolling(lookback, min_periods=lookback).mean())
    )
    return df["price"] - roll_mean


def reversal_vol_normalized(df: pd.DataFrame, lookback: int = DEFAULT_REV_LOOKBACK) -> pd.Series:
    """
    Vol-normalized reversal: (p_t - rolling_mean) / sqrt(sum h^2 over lookback).
    Same z-score interpretation as the momentum version.
    """
    raw = reversal_naive(df, lookback)
    var_lb = (
        df.groupby("market_id")["h2"]
        .transform(lambda s: s.rolling(lookback, min_periods=lookback).sum())
    )
    return raw / np.sqrt(np.clip(var_lb.values, 1e-12, None))

def add_all_signals(
    df: pd.DataFrame,
    train_mask: pd.Series,
    model: str | Dict[str, str] | None = None,
    mom_lookback: int = DEFAULT_MOM_LOOKBACK,
    rev_lookback: int = DEFAULT_REV_LOOKBACK,
    spread_col: str = "spread",
) -> pd.DataFrame:
    df = attach_h2(df, train_mask=train_mask, model=model, spread_col=spread_col)
    df["mom_naive"] = momentum_naive(df, mom_lookback).values
    df["mom_vn"] = momentum_vol_normalized(df, mom_lookback).values
    df["rev_naive"] = reversal_naive(df, rev_lookback).values
    df["rev_vn"] = reversal_vol_normalized(df, rev_lookback).values
    return df

def signal_correlation_and_turnover(
    df: pd.DataFrame,
    mom_col: str = "mom_vn",
    rev_col: str = "rev_vn",
) -> Dict[str, float]:
    valid = df[[mom_col, rev_col]].dropna()
    corr = float(valid[mom_col].corr(valid[rev_col])) if len(valid) > 10 else float("nan")

    def _mean_abs_diff(s):
        d = s.diff().abs()
        return float(d.mean()) if d.notna().any() else float("nan")

    mom_turnover = float(df.groupby("market_id")[mom_col].apply(_mean_abs_diff).mean())
    rev_turnover = float(df.groupby("market_id")[rev_col].apply(_mean_abs_diff).mean())
    return {
        "correlation": corr,
        "momentum_turnover": mom_turnover,
        "reversal_turnover": rev_turnover,
    }