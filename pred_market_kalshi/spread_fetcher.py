import sys
import time
from typing import Dict, List, Optional, Any, Union

import numpy as np
import pandas as pd
from datasets import load_dataset

DEFAULT_HF_DATASET_REPO = "thomaswmitch/kalshi-prediction-markets-betting"
DEFAULT_BUCKET = "1h"


def _to_float(x) -> float:
    """Safe numeric converter for price/volume values."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def normalize_trade_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes raw trades loaded from Hugging Face / Parquet into the standard
    schema expected by effective_spread_by_bucket.
    
    Expected output columns:
        ['trade_id', 'ticker', 'timestamp', 'yes_price', 'no_price', 'count', 'taker_side']
    """
    if df.empty:
        return pd.DataFrame(columns=["trade_id", "ticker", "timestamp", "yes_price", 
                                     "no_price", "count", "taker_side"])

    out = df.copy()

    # 1. Map ticker / market ID
    ticker_col = next((c for c in ["ticker", "market_ticker", "market_id"] if c in out.columns), None)
    if ticker_col:
        out["ticker"] = out[ticker_col]
    else:
        out["ticker"] = "UNKNOWN"

    # 2. Map trade ID
    if "trade_id" not in out.columns:
        out["trade_id"] = out.index.astype(str)

    # 3. Map Timestamp to UTC Datetime
    time_col = next((c for c in ["created_time", "timestamp", "created_ts", "ts"] if c in out.columns), None)
    if time_col:
        out["timestamp"] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    else:
        out["timestamp"] = pd.NaT

    # 4. Normalize Yes / No Prices (Convert integer cents to dollar probabilities if needed)
    if "yes_price_dollars" in out.columns:
        out["yes_price"] = out["yes_price_dollars"].apply(_to_float)
    elif "yes_price" in out.columns:
        out["yes_price"] = out["yes_price"].apply(_to_float)
        # If prices are in cents (e.g. 56 instead of 0.56), scale to dollars
        if out["yes_price"].median() > 1.0:
            out["yes_price"] = out["yes_price"] / 100.0
    elif "price" in out.columns:
        out["yes_price"] = out["price"].apply(_to_float)
        if out["yes_price"].median() > 1.0:
            out["yes_price"] = out["yes_price"] / 100.0
    else:
        out["yes_price"] = np.nan

    if "no_price_dollars" in out.columns:
        out["no_price"] = out["no_price_dollars"].apply(_to_float)
    elif "no_price" in out.columns:
        out["no_price"] = out["no_price"].apply(_to_float)
        if out["no_price"].median() > 1.0:
            out["no_price"] = out["no_price"] / 100.0
    else:
        out["no_price"] = 1.0 - out["yes_price"]

    # 5. Normalize Volume / Count
    count_col = next((c for c in ["count_fp", "count", "volume", "size"] if c in out.columns), None)
    if count_col:
        out["count"] = out[count_col].apply(_to_float)
    else:
        out["count"] = 1.0

    # 6. Normalize Taker Side ("yes" vs "no")
    taker_col = next((c for c in ["taker_side", "taker_outcome_side", "side"] if c in out.columns), None)
    if taker_col:
        out["taker_side"] = out[taker_col].astype(str).str.lower().str.strip()
    else:
        out["taker_side"] = np.nan

    cols = ["trade_id", "ticker", "timestamp", "yes_price", "no_price", "count", "taker_side"]
    return out[cols].drop_duplicates(subset=["trade_id"]).reset_index(drop=True)


def load_trades_dataset(
    repo_id: str = DEFAULT_HF_DATASET_REPO,
    split: str = "train",
    data_files: Optional[Union[str, List[str]]] = None,
) -> pd.DataFrame:
    """
    Loads trade data from Hugging Face datasets into a normalized pandas DataFrame.
    """
    print(f"[spread_fetcher] Loading trades dataset from Hugging Face: {repo_id}...")
    try:
        ds = load_dataset(repo_id, data_files=data_files, split=split)
        df_raw = ds.to_pandas() if hasattr(ds, "to_pandas") else pd.DataFrame(ds)
        df_norm = normalize_trade_schema(df_raw)
        print(f"[spread_fetcher] Successfully loaded {len(df_norm):,} trades across "
              f"{df_norm['ticker'].nunique():,} unique market tickers.")
        return df_norm
    except Exception as e:
        print(f"[spread_fetcher] ERROR loading Hugging Face dataset ({e}). Returning empty DataFrame.")
        return pd.DataFrame(columns=["trade_id", "ticker", "timestamp", "yes_price", 
                                     "no_price", "count", "taker_side"])


def fetch_trades_for_market(
    ticker: str,
    trades_df: Optional[pd.DataFrame] = None,
    min_ts: Optional[int] = None,
    max_ts: Optional[int] = None,
) -> pd.DataFrame:
    """
    Filters trade data for a single market ticker from the pre-loaded dataset.
    """
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(columns=["trade_id", "ticker", "timestamp", "yes_price", 
                                     "no_price", "count", "taker_side"])

    sub = trades_df[trades_df["ticker"] == ticker].copy()
    if sub.empty:
        return sub

    if min_ts is not None:
        min_dt = pd.to_datetime(min_ts, unit="s", utc=True)
        sub = sub[sub["timestamp"] >= min_dt]
    if max_ts is not None:
        max_dt = pd.to_datetime(max_ts, unit="s", utc=True)
        sub = sub[sub["timestamp"] <= max_dt]

    return sub.reset_index(drop=True)


def effective_spread_by_bucket(trades: pd.DataFrame, bucket: str = DEFAULT_BUCKET) -> pd.DataFrame:
    """
    Computes the effective spread per time bucket from trade data:
        spread = mean(yes_price | taker=yes) - mean(yes_price | taker=no)
    """
    if trades.empty:
        return pd.DataFrame(columns=["timestamp", "market_id", "spread", "n_yes_taker",
                                     "n_no_taker", "n_trades", "vwap"])

    df = trades.dropna(subset=["timestamp", "yes_price", "taker_side"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "market_id", "spread", "n_yes_taker",
                                     "n_no_taker", "n_trades", "vwap"])

    df["bucket"] = df["timestamp"].dt.floor(bucket)

    out = []
    for (mkt, b), g in df.groupby(["ticker", "bucket"]):
        yes_takers = g[g["taker_side"] == "yes"]
        no_takers = g[g["taker_side"] == "no"]
        n_yes, n_no = len(yes_takers), len(no_takers)

        if n_yes > 0 and n_no > 0:
            spread = yes_takers["yes_price"].mean() - no_takers["yes_price"].mean()
            spread = max(float(spread), 0.0)
        else:
            spread = np.nan  # Requires two-sided flow

        w = g["count"].fillna(0).to_numpy()
        vwap = (float(np.average(g["yes_price"], weights=w))
                if w.sum() > 0 else float(g["yes_price"].mean()))

        out.append({
            "timestamp": b, "market_id": mkt, "spread": spread,
            "n_yes_taker": n_yes, "n_no_taker": n_no,
            "n_trades": len(g), "vwap": vwap,
        })

    return pd.DataFrame(out).sort_values(["market_id", "timestamp"]).reset_index(drop=True)


def build_spread_panel(
    tickers: List[str],
    trades_df: pd.DataFrame,
    bucket: str = DEFAULT_BUCKET,
    min_tick: float = 0.01,
) -> pd.DataFrame:
    """
    Builds a joinable effective-spread panel for a list of market tickers from 
    the Hugging Face dataset.
    """
    if trades_df.empty:
        print("[build_spread_panel] Provided trades DataFrame is empty.")
        return pd.DataFrame(columns=["market_id", "timestamp", "spread"])

    frames = []
    for tk in tickers:
        sub_trades = fetch_trades_for_market(tk, trades_df=trades_df)
        if sub_trades.empty:
            continue
        buckets = effective_spread_by_bucket(sub_trades, bucket=bucket)
        if buckets.empty:
            continue

        # Floor raw estimates at one tick (0.01), then forward/back fill within market
        buckets["spread"] = buckets["spread"].clip(lower=min_tick)
        buckets = buckets.sort_values("timestamp")
        buckets["spread"] = buckets["spread"].ffill().bfill()
        
        if buckets["spread"].isna().all():
            buckets["spread"] = min_tick
            
        frames.append(buckets[["market_id", "timestamp", "spread"]])

    if not frames:
        print("[build_spread_panel] No spread data reconstructable; downstream will fallback.")
        return pd.DataFrame(columns=["market_id", "timestamp", "spread"])

    return pd.concat(frames, ignore_index=True)


def attach_real_spreads(
    candle_panel: pd.DataFrame,
    trades_df: Optional[pd.DataFrame] = None,
    bucket: str = DEFAULT_BUCKET,
    min_tick: float = 0.01,
    fallback_spread: float = 0.01,
) -> pd.DataFrame:
    """
    Attaches real reconstructed effective spreads to an existing candlestick panel.
    """
    if trades_df is None or trades_df.empty:
        print("[attach_real_spreads] No dataset provided; using nominal fallback spread.")
        out = candle_panel.copy()
        out["spread"] = fallback_spread
        out["spread_is_real"] = False
        return out

    tickers = candle_panel["market_id"].unique().tolist()
    spread_panel = build_spread_panel(tickers, trades_df=trades_df, bucket=bucket, min_tick=min_tick)

    out = candle_panel.copy()
    out["_bucket"] = out["timestamp"].dt.floor(bucket)

    if not spread_panel.empty:
        spread_panel = spread_panel.rename(columns={"timestamp": "_bucket", "spread": "_real_spread"})
        out = out.merge(spread_panel, on=["market_id", "_bucket"], how="left")
    else:
        out["_real_spread"] = np.nan

    out["spread_is_real"] = out["_real_spread"].notna()
    out["spread"] = out["_real_spread"].fillna(fallback_spread).clip(lower=min_tick)
    out = out.drop(columns=["_bucket", "_real_spread"])

    frac_real = out["spread_is_real"].mean() if len(out) else 0.0
    print(f"[attach_real_spreads] {frac_real:.1%} of panel rows carry a real reconstructed "
          f"effective spread; remaining use the {fallback_spread} tick fallback.")
    return out


def diagnose_spread_coverage(
    trades_df: pd.DataFrame,
    tickers: Optional[List[str]] = None,
    bucket: str = DEFAULT_BUCKET,
) -> Dict[str, Any]:
    """
    Evaluates spread coverage across specified tickers using dataset trades.
    """
    if trades_df.empty:
        print("[diagnose] Provided dataset is empty.")
        return {}

    if tickers is None:
        # Sample top 15 most active tickers in dataset
        tickers = trades_df["ticker"].value_counts().head(15).index.tolist()

    per_market = []
    all_bucket_frames = []

    print(f"[diagnose] Assessing spread coverage across {len(tickers)} dataset markets...\n")
    for tk in tickers:
        trades = fetch_trades_for_market(tk, trades_df=trades_df)
        if trades.empty or len(trades) < 20:
            per_market.append({"ticker": tk, "n_trades": len(trades), "n_buckets": 0,
                               "n_two_sided": 0, "coverage": np.nan, "thin": True})
            continue

        buckets = effective_spread_by_bucket(trades, bucket=bucket)
        n_buckets = len(buckets)
        n_two_sided = int(buckets["spread"].notna().sum())
        coverage = n_two_sided / n_buckets if n_buckets else np.nan
        med_spread = float(buckets["spread"].median()) if n_two_sided else np.nan

        print(f"  {tk:40s}  {len(trades):6d} trades  {n_buckets:4d} buckets  "
              f"{coverage:5.1%} two-sided  median spread={med_spread:.3f}")
        per_market.append({"ticker": tk, "n_trades": len(trades), "n_buckets": n_buckets,
                           "n_two_sided": n_two_sided, "coverage": coverage, "thin": False})
        all_bucket_frames.append(buckets)

    summary_df = pd.DataFrame(per_market)
    combined = pd.concat(all_bucket_frames, ignore_index=True) if all_bucket_frames else pd.DataFrame()

    total_buckets = int(summary_df["n_buckets"].sum())
    total_two_sided = int(summary_df["n_two_sided"].sum())
    overall_coverage = total_two_sided / total_buckets if total_buckets else np.nan

    print("\n" + "=" * 70)
    print("DATASET SPREAD COVERAGE DIAGNOSIS")
    print("=" * 70)
    print(f"Total buckets evaluated:    {total_buckets}")
    print(f"Buckets w/ two-sided flow:  {total_two_sided} ({overall_coverage:.1%})")
    print("=" * 70)

    return {
        "per_market": summary_df,
        "buckets": combined,
        "overall_coverage": overall_coverage,
    }


if __name__ == "__main__":
    # Test execution using Hugging Face datasets
    df_trades = load_trades_dataset()
    if not df_trades.empty:
        diagnose_spread_coverage(df_trades)