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
    max_rows: int = 1_000_000,
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
    df_hourly.to_parquet("kalshi_hf_panel.parquet")
    print(
        f"[Success] Built panel with {len(df_hourly):,} hourly bars across"
        f" {df_hourly['market_id'].nunique()} markets!"
    )
    return df_hourly


if __name__ == "__main__":
    # Test execution
    build_panel_from_hf_dataset(max_rows=1_000_000, min_hourly_bars=24)

# # Paper-aligned series mapping across Kalshi's 6 core categories
# CATEGORY_SERIES_MAP = {
#     "Economics": ["KXFED", "KXCPI", "KXJOBS", "KXGDP", "KXINFLATION"],
#     "Crypto": ["KXBTCD", "KXETHD", "KXBTC", "KXETH", "KXSOL", "KXCRYPTO"],
#     "Sports": ["KXNBA", "KXNFLGAME", "KXMLB", "KXNHL", "KXSOCCER"],
#     "Politics": ["KXPOL", "KXCONGRESS", "KXGOV", "KXCABINET", "KXSENATE"],
#     "Elections": ["KXELECTION", "KXPRES", "KXPRIMARY", "KXVOTE"],
#     "Entertainment": ["KXBOXOFFICE", "KXOSCARS", "KXGRAMMY", "KXEMMY", "KXMOVIES"]
# }


# class KalshiSyncIngestor:
#     """Synchronous REST client optimized for historical Kalshi markets."""

#     def __init__(self, session: Optional[requests.Session] = None):
#         self.session = session or requests.Session()
#         self._cutoffs: Optional[Dict[str, int]] = None

#     def fetch_cutoffs(self) -> Dict[str, int]:
#         """Fetches market settlement cutoff timestamp."""
#         try:
#             resp = self.session.get(f"{BASE_URL}/historical/cutoff", timeout=10)
#             resp.raise_for_status()
#             raw = resp.json()
#             self._cutoffs = {}
#             for k, v in raw.items():
#                 try:
#                     self._cutoffs[k] = int(v)
#                 except (ValueError, TypeError):
#                     self._cutoffs[k] = int(pd.to_datetime(v).timestamp()) if v else 0
#         except Exception as e:
#             print(f"[Warning] Failed to fetch historical cutoff ({e}). Defaulting to 0.")
#             self._cutoffs = {"market_settled_ts": 0}

#         return self._cutoffs

#     def discover_resolved_markets(
#         self, limit: int = 60, markets_per_category: Optional[int] = None
#     ) -> List[Dict[str, Any]]:
#         """
#         Discovers liquid settled markets balanced across the paper's 6 categories,
#         with a global fallback if series endpoints return empty.
#         """
#         selected_markets = []
#         quota = markets_per_category or max(1, limit // len(CATEGORY_SERIES_MAP))

#         for category, series_list in CATEGORY_SERIES_MAP.items():
#             cat_markets = []
#             for series in series_list:
#                 try:
#                     time.sleep(0.05)
#                     resp = self.session.get(
#                         f"{BASE_URL}/markets",
#                         params={"series_ticker": series, "status": "settled", "limit": 50},
#                         timeout=10,
#                     )
#                     if resp.status_code == 200:
#                         mkts = resp.json().get("markets", [])
#                         for m in mkts:
#                             m["paper_category"] = category
#                         cat_markets.extend(mkts)
#                 except Exception:
#                     continue

#             def _get_vol(m: Dict[str, Any]) -> float:
#                 for k in ("volume_fp", "volume_24h_fp", "volume", "volume_24h"):
#                     v = m.get(k)
#                     if v is not None:
#                         try:
#                             return float(v)
#                         except (ValueError, TypeError):
#                             pass
#                 return 0.0

#             ranked = sorted([m for m in cat_markets if _get_vol(m) > 0], key=_get_vol, reverse=True)
#             seen = set()
#             count = 0
#             for m in ranked:
#                 t = m.get("ticker")
#                 if t and t not in seen:
#                     seen.add(t)
#                     selected_markets.append(m)
#                     count += 1
#                     if count >= quota:
#                         break

#         # Global Fallback if specific series return fewer markets than needed
#         if len(selected_markets) < limit:
#             try:
#                 resp = self.session.get(
#                     f"{BASE_URL}/markets",
#                     params={"status": "settled", "limit": 100},
#                     timeout=10,
#                 )
#                 if resp.status_code == 200:
#                     extra_mkts = resp.json().get("markets", [])
#                     seen_tickers = {m["ticker"] for m in selected_markets if "ticker" in m}
#                     for m in extra_mkts:
#                         t = m.get("ticker", "")
#                         if t and t not in seen_tickers and _get_vol(m) > 0:
#                             from test1 import assign_paper_category
#                             m["paper_category"] = assign_paper_category(t)
#                             selected_markets.append(m)
#                             seen_tickers.add(t)
#                         if len(selected_markets) >= limit:
#                             break
#             except Exception as e:
#                 print(f"[Warning] Fallback market discovery skipped: {e}")

#         return selected_markets[:limit]

#     def fetch_historical_candlesticks(
#         self, series_ticker: str, ticker: str, start_ts: int, end_ts: int, period_minutes: int = 60
#     ) -> pd.DataFrame:
#         """
#         Fetches hourly candlesticks. If native endpoints return empty,
#         falls back to aggregating hourly bars directly from raw trade history.
#         """
#         params = {
#             "start_ts": int(start_ts),
#             "end_ts": int(end_ts),
#             "period_interval": int(period_minutes)
#         }

#         routes = [f"/historical/markets/{ticker}/candlesticks"]
#         if series_ticker:
#             routes.append(f"/series/{series_ticker}/markets/{ticker}/candlesticks")

#         for route in routes:
#             for attempt in range(2):
#                 try:
#                     resp = self.session.get(f"{BASE_URL}{route}", params=params, timeout=10)
#                     if resp.status_code == 200:
#                         df = self._parse_candlesticks(resp.json())
#                         if not df.empty:
#                             return df
#                     elif resp.status_code == 429:
#                         time.sleep(1.0)
#                 except Exception:
#                     pass

#         # Fallback: Reconstruct hourly candles directly from raw trade history
#         return self._reconstruct_candles_from_trades(ticker, start_ts, end_ts)

#     def _reconstruct_candles_from_trades(self, ticker: str, start_ts: int, end_ts: int) -> pd.DataFrame:
#         trades = []
#         cursor = None

#         for _ in range(10):  # Fetch up to ~1,000 trades
#             params = {"ticker": ticker, "limit": 100}
#             if cursor:
#                 params["cursor"] = cursor

#             try:
#                 resp = self.session.get(f"{BASE_URL}/historical/trades", params=params, timeout=10)
#                 if resp.status_code != 200:
#                     resp = self.session.get(f"{BASE_URL}/markets/trades", params=params, timeout=10)

#                 if resp.status_code != 200:
#                     break

#                 data = resp.json()
#                 raw_t = data.get("trades", [])
#                 if not raw_t:
#                     break

#                 trades.extend(raw_t)
#                 cursor = data.get("cursor")
#                 if not cursor:
#                     break
#             except Exception:
#                 break

#         if not trades:
#             return pd.DataFrame()

#         records = []
#         for tr in trades:
#             ts = tr.get("created_time") or tr.get("ts")
#             price = tr.get("yes_price_dollars") or tr.get("yes_price") or tr.get("price")
#             count = tr.get("count_fp") or tr.get("count") or 1.0

#             if ts and price is not None:
#                 dt = pd.to_datetime(ts, utc=True)
#                 p = float(price) if float(price) <= 1.0 else float(price) / 100.0
#                 records.append({"datetime": dt, "price": p, "volume": float(count)})

#         if not records:
#             return pd.DataFrame()

#         df_tr = pd.DataFrame(records).sort_values("datetime").set_index("datetime")
#         df_hourly = df_tr["price"].resample("1h").last().dropna().to_frame("close_price")
#         df_hourly["volume"] = df_tr["volume"].resample("1h").sum().fillna(0.0)

#         return df_hourly

#     def _parse_candlesticks(self, data: Dict[str, Any]) -> pd.DataFrame:
#         raw_candles = data.get("candlesticks", [])
#         if not raw_candles:
#             return pd.DataFrame()

#         records = []
#         for c in raw_candles:
#             end_ts = c.get("end_period_ts") or c.get("ts")
#             price_val = None

#             if "price" in c:
#                 p = c["price"]
#                 price_val = p.get("close") if isinstance(p, dict) else p
#             elif "yes_price" in c:
#                 price_val = c["yes_price"]
#             elif "close" in c:
#                 price_val = c["close"]

#             volume_val = c.get("volume", 0)

#             if end_ts and price_val is not None:
#                 p_float = float(price_val)
#                 records.append({
#                     "datetime": pd.to_datetime(end_ts, unit="s", utc=True),
#                     "close_price": p_float if p_float <= 1.0 else p_float / 100.0,
#                     "volume": float(volume_val),
#                 })

#         if not records:
#             return pd.DataFrame()

#         df = pd.DataFrame(records)
#         df.set_index("datetime", inplace=True)
#         df.sort_index(inplace=True)
#         return df


# def build_market_panel(
#     limit_markets: int = 60,
#     ingestor: Optional[KalshiSyncIngestor] = None,
#     use_real_spreads: bool = True,
# ) -> pd.DataFrame:
#     """Constructs an hourly panel of resolved contracts across all 6 paper categories."""
#     if ingestor is None:
#         ingestor = KalshiSyncIngestor()

#     print("[Info] Fetching category-balanced resolved markets from Kalshi API...")
#     markets = ingestor.discover_resolved_markets(limit=limit_markets)

#     if not markets:
#         print("[Error] No resolved markets found.")
#         return pd.DataFrame()

#     print(f"[Info] Found {len(markets)} liquid resolved markets. Fetching hourly candlestick series...")
#     panel_records = []

#     for mkt in markets:
#         ticker = mkt.get("ticker")
#         series_ticker = mkt.get("series_ticker") or (ticker.split("-")[0] if ticker else "")
#         paper_cat = mkt.get("paper_category", "Economics")

#         expiration_str = mkt.get("expiration_time") or mkt.get("close_time") or mkt.get("settlement_timer_started_at")
#         open_str = mkt.get("open_time") or mkt.get("created_time")

#         if not ticker or not expiration_str:
#             continue

#         expiration_dt = pd.to_datetime(expiration_str, utc=True)
#         end_ts = int(expiration_dt.timestamp())

#         if open_str:
#             start_ts = int(pd.to_datetime(open_str, utc=True).timestamp())
#         else:
#             start_ts = int(end_ts - (30 * 24 * 3600))

#         if start_ts >= end_ts:
#             start_ts = int(end_ts - (7 * 24 * 3600))

#         time.sleep(0.1)

#         df_candles = ingestor.fetch_historical_candlesticks(
#             series_ticker=series_ticker,
#             ticker=ticker,
#             start_ts=start_ts,
#             end_ts=end_ts,
#             period_minutes=60,
#         )

#         if df_candles.empty or len(df_candles) < 3:
#             continue

#         df_candles = df_candles.reset_index()

#         df_candles["spread"] = np.nan
#         df_candles["market_id"] = ticker
#         df_candles["category"] = paper_cat
#         df_candles["timestamp"] = df_candles["datetime"]
#         df_candles["price"] = df_candles["close_price"]

#         tau_days = (expiration_dt - df_candles["timestamp"]).dt.total_seconds() / 86400.0
#         df_candles["days_to_resolution"] = np.maximum(tau_days, 1.0 / 24.0)

#         cols = ["market_id", "category", "timestamp", "price", "volume", "spread", "days_to_resolution"]
#         panel_records.append(df_candles[cols])

#     if not panel_records:
#         print("[Error] Could not retrieve candlestick history for any resolved market.")
#         return pd.DataFrame()

#     full_panel = pd.concat(panel_records, ignore_index=True)
#     full_panel.sort_values(["market_id", "timestamp"], inplace=True)
#     full_panel.reset_index(drop=True, inplace=True)

#     if use_real_spreads:
#         from spread_fetcher import attach_real_spreads
#         print("[Info] Reconstructing real effective spreads from public trade feed...")
#         full_panel = attach_real_spreads(full_panel, session=ingestor.session)
#     else:
#         full_panel["spread"] = 0.01
#         full_panel["spread_is_real"] = False

#     full_panel.to_parquet("kalshi_resolved_panel.parquet")
#     print(f"[Success] Loaded {len(full_panel)} hourly bars across {full_panel['market_id'].nunique()} resolved markets!")

#     return full_panel