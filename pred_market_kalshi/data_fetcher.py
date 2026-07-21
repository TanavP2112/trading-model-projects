import os
import re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
import duckdb


def assign_paper_category(ticker: str) -> str:
    """Classifies Kalshi market tickers into 5 core academic categories.
    Matches standard series prefixes (with optional 'KX' prefix).
    """
    t = str(ticker).upper()

    # 1. Sports (Leagues, Majors, Tournaments, Player Props)
    sports_pattern = (
        r"^(KX)?(NBA|MLB|NFL|NHL|MLS|WNBA|SOCCER|ATP|WTA|PGA|THEOPEN|"
        r"GOLF|TENNIS|EPL|UFC|BOXING|NASCAR|NCAAB|NCAAF|MASTERS|"
        r"WIMBLEDON|USOPEN|HIGHLAX|WWOMENSINGLES|WMENSINGLES)"
    )
    if re.search(sports_pattern, t):
        return "Sports"

    # 2. Crypto (Tokens, Industry, ETF approvals)
    crypto_pattern = r"^(KX)?(BTC|ETH|SOL|DOGE|XRP|AVAX|CRYPTO|BITCOIN)"
    if re.search(crypto_pattern, t):
        return "Crypto"

    # 3. Entertainment & Culture
    ent_pattern = (
        r"^(KX)?(RT|NETFLIX|BOXOFFICE|OSCARS|GRAMMY|EMMY|TOPALBUM|TONY|"
        r"GOLDENGLOBE|SPOTIFY|CONNSMYTHE|SQUIDGAMES|1SONG)"
    )
    if re.search(ent_pattern, t):
        return "Entertainment"

    # 4. Politics & Government (Elections, Appointments, Congress)
    pol_pattern = (
        r"^(KX)?(POL|CONGRESS|GOV|SENATE|ELECTION|PRES|PRIMARY|STATEMENT|"
        r"SCOTUS|CABINET|APPROVAL|MAYOR|HOUSE|NYCMAYOR|NYCBOROUGH|TRUMPMENTION)"
    )
    if re.search(pol_pattern, t):
        return "Politics"

    # 5. Economics & Macro (Default fallback: CPI, FED, GDP, Jobless Claims, Rates)
    return "Economics"


def resample_and_ffill_markets(df: pd.DataFrame) -> pd.DataFrame:
    """Groups data by market_id and creates an unbroken hourly grid per market

    using vectorized pandas resampling to improve performance on large datasets.
    """
    print("[Info] Resampling and forward-filling hourly grid per market...")

    time_col = "datetime" if "datetime" in df.columns else "timestamp"

    # Vectorized GroupBy + Resample across markets
    resampled = (
        df.set_index(time_col)
        .groupby("market_id")
        .resample("1h")
        .agg({"price": "last", "volume": "sum"})
    )

    # Forward-fill price per market to bridge inactive/quiet hours
    resampled["price"] = (
        resampled.groupby("market_id")["price"].ffill().bfill()
    )

    # Fill empty volume bars with 0.0 (no trades occurred)
    resampled["volume"] = resampled["volume"].fillna(0.0)

    panel_df = resampled.reset_index()
    panel_df.rename(columns={time_col: "timestamp"}, inplace=True)

    return panel_df


def build_panel_from_hf_dataset(
    repo_id: str = "TrevorJS/kalshi-trades",
    min_hourly_bars: int = 24
) -> pd.DataFrame:
    print(f"[Info] Downloading/verifying dataset files locally for '{repo_id}'...")
    
    # 1. Download parquet files to HF local cache with retry handling
    # snapshot_download handles rate-limits, resume-on-failure, and local caching automatically
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns="trades-*.parquet",
        max_workers=4  # Limit concurrent workers to prevent triggering HF rate limits
    )

    local_pattern = os.path.join(local_dir, "trades-*.parquet").replace("\\", "/")
    print(f"[Info] Streaming and aggregating local Parquet files via DuckDB: {local_pattern}")

    # 2. Execute DuckDB aggregation over local disk files (100% immune to HTTP 429)
    query = f"""
    WITH raw_trades AS (
        SELECT 
            ticker AS market_id,
            created_time AS timestamp,
            yes_price / 100.0 AS price,
            CAST(count AS DOUBLE) AS volume
        FROM '{local_pattern}'
    ),
    hourly_grid AS (
        SELECT 
            market_id,
            time_bucket(INTERVAL '1 hour', timestamp) AS timestamp,
            LAST(price) AS price,
            SUM(volume) AS volume
        FROM raw_trades
        GROUP BY market_id, time_bucket(INTERVAL '1 hour', timestamp)
    )
    SELECT * FROM hourly_grid
    """

    df_hourly = duckdb.query(query).df()
    print(f"[Info] Aggregated into {len(df_hourly):,} hourly bars! Cleaning panel...")

    # 3. Sort and Forward-Fill Prices per Market
    df_hourly.sort_values(["market_id", "timestamp"], inplace=True)
    df_hourly["price"] = (
        df_hourly.groupby("market_id")["price"].ffill().bfill()
    )
    df_hourly["volume"] = df_hourly["volume"].fillna(0.0)

    # 4. Time to Resolution Proxy (tau in days)
    max_ts = df_hourly.groupby("market_id")["timestamp"].transform("max")
    tau_days = (max_ts - df_hourly["timestamp"]).dt.total_seconds() / 86400.0
    df_hourly["days_to_resolution"] = np.maximum(tau_days, 1.0 / 24.0)

    # 5. Category & Spread
    df_hourly["category"] = df_hourly["market_id"].apply(assign_paper_category)
    df_hourly["spread"] = 0.01

    # 6. Filter Illiquid Markets
    counts = df_hourly.groupby("market_id").size()
    valid_markets = counts[counts >= min_hourly_bars].index
    df_hourly = df_hourly[df_hourly["market_id"].isin(valid_markets)].copy()

    df_hourly.reset_index(drop=True, inplace=True)

    # Cache local panel
    os.makedirs("data", exist_ok=True)
    df_hourly.to_parquet("data/kalshi_hf_panel.parquet")

    print(
        f"[Success] Built panel with {len(df_hourly):,} hourly bars across"
        f" {df_hourly['market_id'].nunique()} markets!"
    )
    return df_hourly


if __name__ == "__main__":
    # Test execution across full dataset
    build_panel_from_hf_dataset(min_hourly_bars=24)