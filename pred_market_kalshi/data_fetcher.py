import os
import numpy as np
import pandas as pd
from datasets import load_dataset


def assign_paper_category(ticker: str) -> str:
    """Classifies Kalshi tickers into paper categories based on explicit prefixes."""
    t = str(ticker).upper()

    # 1. Sports (Check explicitly BEFORE general keyword matches)
    sports_prefixes = [
        "KXNBA",
        "KXMLB",
        "KXNFL",
        "KXNHL",
        "KXMLS",
        "KXWNBA",
        "KXSOCCER",
        "KXATPMATCH",
        "KXWTAMATCH",
        "KXPGATOUR",
        "KXTHEOPEN",
        "KXF1RACE",
        "KXF1",
    ]
    if any(t.startswith(p) for p in sports_prefixes):
        return "Sports"

    # 2. Crypto
    crypto_prefixes = ["KXBTCD", "KXETHD", "KXBTC", "KXETH", "KXSOL", "KXCRYPTO"]
    if any(t.startswith(p) for p in crypto_prefixes):
        return "Crypto"

    # 3. Entertainment & Culture
    ent_prefixes = [
        "KXRT",
        "KXNETFLIX",
        "KXBOXOFFICE",
        "KXOSCARS",
        "KXGRAMMY",
        "KXEMMY",
        "KXTOPALBUM",
    ]
    if any(t.startswith(p) for p in ent_prefixes):
        return "Entertainment"

    # 4. Politics & Elections (Check prefix to avoid matching inside player names)
    pol_prefixes = [
        "KXPOL",
        "KXCONGRESS",
        "KXGOV",
        "KXSENATE",
        "KXELECTION",
        "KXPRES",
        "KXPRIMARY",
        "KXSTATEMENT",
    ]
    if any(t.startswith(p) for p in pol_prefixes):
        return "Politics"

    # 5. Economics & Macro (Default fallback)
    return "Economics"


def resample_and_ffill_markets(df: pd.DataFrame) -> pd.DataFrame:
    """Groups data by market_id, creates an unbroken hourly grid for each market,

    and forward-fills prices and metadata to eliminate time gaps.
    """
    print("[Info] Resampling and forward-filling hourly grid per market...")

    filled_dfs = []
    time_col = "datetime" if "datetime" in df.columns else "timestamp"

    for market_id, group in df.groupby("market_id"):
        # Set datetime/timestamp index and sort
        group = group.set_index(time_col).sort_index()

        # 1. Aggregate duplicate trades within the same hour
        hourly = group.resample("1h").agg({"price": "last", "volume": "sum"})

        # 2. Forward-fill price across quiet hours (last known price carries forward)
        hourly["price"] = hourly["price"].ffill().bfill()

        # 3. Fill missing volume with 0 (no trades occurred in gap hours)
        hourly["volume"] = hourly["volume"].fillna(0.0)

        # 4. Re-attach market_id
        hourly["market_id"] = market_id

        filled_dfs.append(hourly.reset_index())

    # Combine back into a single panel DataFrame
    panel_df = pd.concat(filled_dfs, ignore_index=True)

    # Standardize output timestamp column name
    if "index" in panel_df.columns:
        panel_df = panel_df.rename(columns={"index": "timestamp"})
    elif "datetime" in panel_df.columns:
        panel_df = panel_df.rename(columns={"datetime": "timestamp"})

    return panel_df


def build_panel_from_hf_dataset(
    repo_id: str = "thomaswmitch/kalshi-prediction-markets-betting",
    min_hourly_bars: int = 24,
    max_rows: int = 3_000_000,
) -> pd.DataFrame:
    print(f"[Info] Loading Hugging Face dataset '{repo_id}'...")
    ds = load_dataset(repo_id, split="train")

    if max_rows and len(ds) > max_rows:
        print(
            f"[Info] Sampling top {max_rows:,} rows for rapid prototyping..."
        )
        df = ds.select(range(max_rows)).to_pandas()
    else:
        df = ds.to_pandas()

    print(f"[Info] Loaded {len(df):,} raw trades. Formatting schema...")

    # 1. Safely locate ticker column without creating duplicated columns
    ticker_col = None
    for candidate in ["market_ticker", "ticker", "market_id"]:
        if candidate in df.columns:
            ticker_col = candidate
            break

    if not ticker_col:
        raise KeyError("Could not find a valid ticker column in dataset.")

    df["market_id"] = df[ticker_col].astype(str)

    # 2. Standardize timestamp and volume columns
    time_col = "created_time" if "created_time" in df.columns else "datetime"
    vol_col = "count" if "count" in df.columns else "volume"

    df["datetime"] = pd.to_datetime(df[time_col], utc=True)
    df["volume"] = df[vol_col].astype(float) if vol_col in df.columns else 1.0

    # 3. Convert integer cents (1-99) to probabilities (0.01-0.99)
    if "yes_price" in df.columns:
        df["price"] = df["yes_price"].astype(float) / 100.0
    elif "price" in df.columns:
        df["price"] = df["price"].astype(float)
        if df["price"].max() > 1.0:
            df["price"] = df["price"] / 100.0

    # Clean up duplicate columns to guarantee 1D Series for GroupBy
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # 4. Resample and forward-fill on continuous 1-hour grids per market
    df_hourly = resample_and_ffill_markets(df)

    # 5. Calculate time to resolution proxy
    max_ts = df_hourly.groupby("market_id")["timestamp"].transform("max")
    tau_days = (max_ts - df_hourly["timestamp"]).dt.total_seconds() / 86400.0
    df_hourly["days_to_resolution"] = np.maximum(tau_days, 1.0 / 24.0)

    # 6. Add category mapping & default spread
    df_hourly["category"] = df_hourly["market_id"].apply(assign_paper_category)
    df_hourly["spread"] = 0.01

    # 7. Filter out illiquid markets with too few bars
    counts = df_hourly.groupby("market_id").size()
    valid_markets = counts[counts >= min_hourly_bars].index
    df_hourly = df_hourly[df_hourly["market_id"].isin(valid_markets)].copy()

    df_hourly.sort_values(["market_id", "timestamp"], inplace=True)
    df_hourly.reset_index(drop=True, inplace=True)

    # Save to local cache
    df_hourly.to_parquet("data/kalshi_hf_panel.parquet")
    print(
        f"[Success] Built panel with {len(df_hourly):,} hourly bars across"
        f" {df_hourly['market_id'].nunique()} markets!"
    )
    return df_hourly


if __name__ == "__main__":
    # Test execution
    build_panel_from_hf_dataset(min_hourly_bars=24)
