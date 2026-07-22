import os
import re
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
import duckdb


def assign_paper_category(ticker: str) -> str:
    """Classifies Kalshi market tickers into 5 core academic categories.
    Note: These are based on what I found from the dataset, not an official Kalshi taxonomy.
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


def build_panel_from_hf_dataset(
    repo_id: str = "TrevorJS/kalshi-trades",
    min_hourly_bars: int = 48,
    reconstruct_spread: bool = True,
    ewma_alpha: float = 0.3,
    output_path: str = "data/kalshi_hf_panel.parquet",
) -> pd.DataFrame:
    """Build hourly panel from Kalshi trade data with optional per-hour
    spread reconstruction from trade aggressor pattern.

    reconstruct_spread : bool
        If True (default), compute per-hour effective spread from taker_side
        aggressor pattern:
            spread = min(price where taker='yes') - max(price where taker='no')
        With rolling-24h market-specific median fallback for hours with
        insufficient two-sided flow, EWMA smoothing (alpha=ewma_alpha),
        and a floor at 0.01 (Kalshi's minimum tick).

        If False, uses a constant 0.01 spread placeholder (original behavior).
    """
    print(f"[Info] Downloading/verifying dataset files locally for '{repo_id}'...")

    # Download parquet files to HF local cache
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns="trades-*.parquet",
        max_workers=4,
    )
    local_pattern = os.path.join(local_dir, "trades-*.parquet").replace("\\", "/")

    if reconstruct_spread:
        print(f"[Info] Aggregating with per-hour spread reconstruction via DuckDB: {local_pattern}")
        query = f"""
        WITH raw_trades AS (
            SELECT
                ticker AS market_id,
                created_time AS timestamp,
                yes_price / 100.0 AS price,
                CAST(count AS DOUBLE) AS volume,
                taker_side
            FROM '{local_pattern}'
        ),
        hourly_grid AS (
            SELECT
                market_id,
                time_bucket(INTERVAL '1 hour', timestamp) AS timestamp,
                SUM(price * volume) / NULLIF(SUM(volume), 0) AS price,
                SUM(volume) AS volume,
                MIN(CASE WHEN taker_side = 'yes' THEN price END) AS min_ask_touch,
                MAX(CASE WHEN taker_side = 'no'  THEN price END) AS max_bid_touch,
                COUNT(CASE WHEN taker_side = 'yes' THEN 1 END)   AS n_yes,
                COUNT(CASE WHEN taker_side = 'no'  THEN 1 END)   AS n_no
            FROM raw_trades
            GROUP BY market_id, time_bucket(INTERVAL '1 hour', timestamp)
        )
        SELECT
            market_id,
            timestamp,
            price,
            volume,
            CASE
                WHEN n_yes >= 2 AND n_no >= 2
                     AND (min_ask_touch - max_bid_touch) > 0
                THEN (min_ask_touch - max_bid_touch)
                ELSE NULL
            END AS spread_raw
        FROM hourly_grid
        """
    else:
        print(f"[Info] Aggregating with constant 0.01 spread placeholder via DuckDB")
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

    # Sort by market_id and timestamp
    df_hourly["timestamp"] = pd.to_datetime(df_hourly["timestamp"], utc=True)
    df_hourly.sort_values(["market_id", "timestamp"], inplace=True)
    df_hourly.reset_index(drop=True, inplace=True)

    # Gap-aware bar validity
    df_hourly["gap_hours"] = (
        df_hourly.groupby("market_id")["timestamp"]
                 .diff()
                 .dt.total_seconds() / 3600.0
    )
    # tag clean bars
    df_hourly["is_clean_bar"] = (
        df_hourly["gap_hours"].between(0.9, 1.1)
    )
    n_total = len(df_hourly)
    n_clean = int(df_hourly["is_clean_bar"].sum())
    n_gapped = n_total - n_clean
    print(f"[Info] Bar validity: {n_clean:,} clean 1-hour bars "
          f"({100*n_clean/n_total:.1f}%), {n_gapped:,} gap-preceded or "
          f"first-of-market bars.")
    df_hourly["volume"] = df_hourly["volume"].fillna(0.0)

    if reconstruct_spread:
        print("[Info] Filling spread NULLs with per-market rolling 24h medians...")
        df_hourly["spread_rolling"] = (
            df_hourly.groupby("market_id")["spread_raw"]
                     .transform(lambda s: s.rolling(24, min_periods=3).median())
        )

        df_hourly["spread_filled"] = (
            df_hourly["spread_raw"]
                     .fillna(df_hourly["spread_rolling"])
                     .fillna(0.01)
        )
        print(f"[Info] Applying EWMA smoothing (alpha={ewma_alpha}) per market...")
        df_hourly["spread"] = (
            df_hourly.groupby("market_id")["spread_filled"]
                     .transform(lambda s: s.ewm(alpha=ewma_alpha, adjust=False).mean())
                     .clip(lower=0.01)  # Floor at Kalshi's minimum tick
        )
        df_hourly = df_hourly.drop(columns=["spread_raw", "spread_rolling", "spread_filled"])
    else:
        df_hourly["spread"] = 0.01

    max_ts = df_hourly.groupby("market_id")["timestamp"].transform("max")
    tau_days = (max_ts - df_hourly["timestamp"]).dt.total_seconds() / 86400.0
    df_hourly["days_to_resolution"] = np.maximum(tau_days, 1.0 / 24.0)
    df_hourly["category"] = df_hourly["market_id"].apply(assign_paper_category)

    counts = df_hourly.groupby("market_id").size()
    valid_markets = counts[counts >= min_hourly_bars].index
    df_hourly = df_hourly[df_hourly["market_id"].isin(valid_markets)].copy()
    df_hourly.reset_index(drop=True, inplace=True)

    os.makedirs("data", exist_ok=True)
    df_hourly.to_parquet(output_path)
    print(
        f"[Success] Built panel with {len(df_hourly):,} hourly bars across"
        f" {df_hourly['market_id'].nunique()} markets at {output_path}"
    )

    if reconstruct_spread:
        print()
        print("=" * 60)
        print("Spread reconstruction diagnostics")
        print("=" * 60)
        print(f"  n bars total:                     {len(df_hourly):,}")
        print(f"  spread median:                    {df_hourly.spread.median():.4f}")
        print(f"  spread mean:                      {df_hourly.spread.mean():.4f}")
        p10, p25, p75, p90 = df_hourly.spread.quantile([0.10, 0.25, 0.75, 0.90])
        print(f"  spread p10 / p25 / p75 / p90:     {p10:.4f} / {p25:.4f} / {p75:.4f} / {p90:.4f}")
        floor_hits = (df_hourly.spread <= 0.0101).sum()
        print(f"  spread == 0.01 (floor):           {floor_hits:,} "
              f"({floor_hits/len(df_hourly)*100:.1f}%)")
        print(f"  spread > 0.05:                    {(df_hourly.spread > 0.05).sum():,} "
              f"({(df_hourly.spread > 0.05).mean()*100:.1f}%)")
        corr = df_hourly[["spread", "volume"]].corr().iloc[0, 1]
        print(f"  Correlation(spread, volume):      {corr:+.3f}")
        print("=" * 60)

    return df_hourly


if __name__ == "__main__":
    build_panel_from_hf_dataset(min_hourly_bars=48, reconstruct_spread=True)