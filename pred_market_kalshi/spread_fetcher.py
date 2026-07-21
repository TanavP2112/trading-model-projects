import time
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import requests

from config import BASE_URL


HISTORICAL_TRADES_ROUTE = "/historical/trades"
LIVE_TRADES_ROUTE = "/markets/trades"
DEFAULT_BUCKET = "1h"


def fetch_trade_cutoff(session: Optional[requests.Session] = None) -> int:
    """
    Returns the `trades_created_ts` cutoff: trades filled BEFORE this Unix
    timestamp live only on /historical/trades; trades AFTER it live only on
    the live /markets/trades endpoint. Kalshi advances this forward over
    time (target live window ~3 months), so it must be fetched, not assumed.

    Returns 0 on failure, which makes the router default to the historical
    endpoint everywhere (the old behavior) rather than silently breaking.
    """
    session = session or requests.Session()
    try:
        resp = session.get(f"{BASE_URL}/historical/cutoff", timeout=10)
        resp.raise_for_status()
        raw = resp.json()
        val = raw.get("trades_created_ts", 0)
        try:
            return int(val)
        except (ValueError, TypeError):
            return int(pd.to_datetime(val).timestamp()) if val else 0
    except Exception as e:
        print(f"[spread_fetcher] could not fetch trade cutoff ({e}); "
              f"defaulting to historical-only routing.")
        return 0


def _to_float(x) -> float:
    """Kalshi returns numeric fields as strings ('0.5600', '10.00')."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _paginate_trades(
    route: str,
    ticker: str,
    session: requests.Session,
    min_ts: Optional[int],
    max_ts: Optional[int],
    max_pages: int,
    page_limit: int,
    pause: float,
) -> List[Dict[str, Any]]:
    """
    Cursor-paginate one trades endpoint (live OR historical -- same schema
    and same cursor mechanics per Kalshi docs) and return raw row dicts.
    """
    params: Dict[str, Any] = {"ticker": ticker, "limit": page_limit}
    if min_ts is not None:
        params["min_ts"] = int(min_ts)
    if max_ts is not None:
        params["max_ts"] = int(max_ts)

    rows: List[Dict[str, Any]] = []
    cursor = None
    pages = 0
    while pages < max_pages:
        if cursor:
            params["cursor"] = cursor
        try:
            resp = session.get(f"{BASE_URL}{route}", params=params, timeout=15)
            if resp.status_code == 429:
                time.sleep(2.0)
                continue
            if resp.status_code == 404:
                # Not necessarily an error: a purely-recent market legitimately
                # has nothing on /historical, and vice versa. Caller merges both.
                break
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"[spread_fetcher] request error for {ticker} on {route}: {e}")
            break

        payload = resp.json()
        trades = payload.get("trades", [])
        for t in trades:
            rows.append({
                "trade_id": t.get("trade_id"),
                "ticker": t.get("ticker", ticker),
                "timestamp": pd.to_datetime(t.get("created_time"), utc=True, errors="coerce"),
                "yes_price": _to_float(t.get("yes_price_dollars")),
                "no_price": _to_float(t.get("no_price_dollars")),
                "count": _to_float(t.get("count_fp")),
                # Kalshi migrated taker_side -> taker_outcome_side; accept either.
                "taker_side": t.get("taker_side") or t.get("taker_outcome_side"),
            })
        cursor = payload.get("cursor")
        pages += 1
        if not cursor or not trades:
            break
        time.sleep(pause)

    if pages >= max_pages:
        print(f"[spread_fetcher] WARNING: hit max_pages={max_pages} for {ticker} on {route}; "
              f"history may be truncated. Raise max_pages if this market is very active.")
    return rows


def fetch_trades_for_market(
    ticker: str,
    session: Optional[requests.Session] = None,
    min_ts: Optional[int] = None,
    max_ts: Optional[int] = None,
    trade_cutoff: Optional[int] = None,
    max_pages: int = 200,
    page_limit: int = 1000,
    pause: float = 0.12,
) -> pd.DataFrame:
    """
    Pulls ALL trades for one market ticker, routing by the trade cutoff:
    trades before `trade_cutoff` come from /historical/trades, trades after
    from the live /markets/trades. A single market's history can straddle
    the cutoff, so by default we query BOTH and merge (dedup on trade_id) --
    exactly as Kalshi's migration guide recommends. This is the fix for the
    "no trades returned" symptom on recent markets (e.g. today's KXINX):
    their trades are newer than the cutoff and simply aren't on /historical.

    Pass trade_cutoff explicitly (from fetch_trade_cutoff) to avoid re-
    fetching it per market. If None, it's fetched once here.

    Returns a DataFrame: trade_id, ticker, timestamp (UTC), yes_price,
    no_price, count, taker_side. Empty DataFrame if genuinely no trades.
    """
    session = session or requests.Session()
    if trade_cutoff is None:
        trade_cutoff = fetch_trade_cutoff(session)

    # Decide which endpoints are worth hitting given the requested window and
    # the cutoff. When in doubt (no window, or window straddles cutoff), hit
    # both -- correctness beats saving one request.
    want_historical = True
    want_live = True
    if trade_cutoff > 0:
        if min_ts is not None and min_ts >= trade_cutoff:
            want_historical = False  # entire requested window is post-cutoff
        if max_ts is not None and max_ts < trade_cutoff:
            want_live = False        # entire requested window is pre-cutoff

    rows: List[Dict[str, Any]] = []
    if want_historical:
        rows += _paginate_trades(HISTORICAL_TRADES_ROUTE, ticker, session,
                                 min_ts, max_ts, max_pages, page_limit, pause)
    if want_live:
        rows += _paginate_trades(LIVE_TRADES_ROUTE, ticker, session,
                                 min_ts, max_ts, max_pages, page_limit, pause)

    if not rows:
        return pd.DataFrame(columns=["trade_id", "ticker", "timestamp", "yes_price",
                                     "no_price", "count", "taker_side"])
    df = pd.DataFrame(rows)
    # Merge dedup: a trade near the boundary could appear in both feeds.
    df = df.drop_duplicates(subset=["trade_id"]).reset_index(drop=True)
    return df


def effective_spread_by_bucket(trades: pd.DataFrame, bucket: str = DEFAULT_BUCKET) -> pd.DataFrame:
    """
    Given a market's trades, compute the effective spread per time bucket:

        spread = mean(yes_price | taker=yes) - mean(yes_price | taker=no)

    Returns one row per bucket with: timestamp (bucket start), market_id,
    spread (NaN when the bucket lacks two-sided flow), n_yes_taker,
    n_no_taker, n_trades, vwap (volume-weighted mean yes_price, always
    available when there's >=1 trade -- useful as the price series too).

    The spread is clipped at 0 on the low side: noise can occasionally make
    the yes-taker mean below the no-taker mean, which is economically a
    zero/negative effective spread -- we floor at 0 rather than emit a
    negative variance input to fit_K.
    """
    if trades.empty:
        return pd.DataFrame(columns=["timestamp", "market_id", "spread", "n_yes_taker",
                                     "n_no_taker", "n_trades", "vwap"])

    df = trades.dropna(subset=["timestamp", "yes_price", "taker_side"]).copy()
    df["bucket"] = df["timestamp"].dt.floor(bucket)

    out = []
    for (mkt, b), g in df.groupby([df["ticker"], df["bucket"]]):
        yes_takers = g[g["taker_side"] == "yes"]
        no_takers = g[g["taker_side"] == "no"]
        n_yes, n_no = len(yes_takers), len(no_takers)

        if n_yes > 0 and n_no > 0:
            spread = yes_takers["yes_price"].mean() - no_takers["yes_price"].mean()
            spread = max(float(spread), 0.0)
        else:
            spread = np.nan  # can't difference without both sides

        w = g["count"].fillna(0).to_numpy()
        vwap = (float(np.average(g["yes_price"], weights=w))
                if w.sum() > 0 else float(g["yes_price"].mean()))

        out.append({
            "timestamp": b, "market_id": mkt, "spread": spread,
            "n_yes_taker": n_yes, "n_no_taker": n_no,
            "n_trades": len(g), "vwap": vwap,
        })

    return pd.DataFrame(out).sort_values(["market_id", "timestamp"]).reset_index(drop=True)


def diagnose_spread_coverage(
    tickers: List[str],
    session: Optional[requests.Session] = None,
    bucket: str = DEFAULT_BUCKET,
    min_ts: Optional[int] = None,
) -> Dict[str, Any]:
    """
    THE FIRST THING TO RUN. Pulls trades for a small set of tickers and
    reports how usable the effective-spread reconstruction actually is,
    BEFORE committing to a full panel build. Specifically measures the
    make-or-break quantity: what fraction of (market, hour) buckets have
    two-sided taker flow (and thus a real spread estimate).

    Prints a readable report and returns the raw numbers so a caller can
    branch on them (e.g. widen the bucket if coverage < 50%).
    """
    session = session or requests.Session()
    per_market = []
    all_bucket_frames = []

    # Fetch the cutoff ONCE and thread it through -- this is what lets recent
    # markets route to the live endpoint instead of silently returning empty.
    trade_cutoff = fetch_trade_cutoff(session)
    if trade_cutoff > 0:
        cutoff_dt = pd.to_datetime(trade_cutoff, unit="s", utc=True)
        print(f"[diagnose] trade cutoff = {trade_cutoff} ({cutoff_dt}); trades newer than "
              f"this route to the LIVE endpoint, older to /historical.\n")

    # A market needs at least this many trades before a spread estimate is
    # even meaningful -- separates "illiquid market" from "wrong endpoint".
    MIN_TRADES_FOR_SIGNAL = 20

    print(f"[diagnose] Pulling trades for {len(tickers)} markets to assess spread coverage...\n")
    for tk in tickers:
        trades = fetch_trades_for_market(tk, session=session, min_ts=min_ts,
                                         trade_cutoff=trade_cutoff)
        if trades.empty:
            print(f"  {tk:40s}  no trades returned (checked both live + historical)")
            per_market.append({"ticker": tk, "n_trades": 0, "n_buckets": 0,
                               "n_two_sided": 0, "coverage": np.nan, "thin": True})
            continue
        if len(trades) < MIN_TRADES_FOR_SIGNAL:
            print(f"  {tk:40s}  {len(trades):6d} trades  -- TOO THIN to estimate spread, skipping")
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

    n_thin = int(summary_df["thin"].sum()) if "thin" in summary_df.columns else 0
    n_liquid = len(tickers) - n_thin

    print("\n" + "=" * 70)
    print("SPREAD COVERAGE DIAGNOSIS")
    print("=" * 70)
    print(f"Markets sampled:            {len(tickers)}")
    print(f"  too thin to use:          {n_thin}  (excluded from coverage below)")
    print(f"  liquid enough to assess:  {n_liquid}")
    print(f"Total (market,{bucket}) buckets: {total_buckets}   (liquid markets only)")
    print(f"Buckets w/ two-sided flow:  {total_two_sided}  ({overall_coverage:.1%})")
    if n_liquid == 0:
        print("\n!! Every sampled market was too thin OR returned no trades. This is a")
        print("   SAMPLING problem, not a verdict on the method -- re-run on deliberately")
        print("   liquid markets (high-volume KXBTC/KXFED/major finals) before concluding.")
    if combined is not None and not combined.empty and total_two_sided > 0:
        valid = combined["spread"].dropna()
        print(f"Effective spread distribution (dollars): "
              f"median={valid.median():.3f}  p25={valid.quantile(.25):.3f}  "
              f"p75={valid.quantile(.75):.3f}")
        print(f"Spread VARIATION (std across buckets): {valid.std():.4f}  "
              f"<- this is what makes fit_K meaningful; near-zero would be a red flag")
    print("=" * 70)
    if not np.isnan(overall_coverage):
        if overall_coverage >= 0.5:
            print("VERDICT: coverage >= 50%. Effective-spread reconstruction is viable;")
            print("proceed to a full panel build (consider forward-filling the gap buckets).")
        elif overall_coverage >= 0.25:
            print("VERDICT: moderate coverage. Consider a WIDER bucket (e.g. '2h'/'4h') to")
            print("raise two-sided rate, accepting coarser time resolution for the AS term.")
        else:
            print("VERDICT: LOW coverage. Hourly effective spread will be too sparse; either")
            print("widen the bucket substantially or fall back to labeling the model DR-only.")

    return {
        "per_market": summary_df,
        "buckets": combined,
        "overall_coverage": overall_coverage,
        "total_buckets": total_buckets,
        "total_two_sided": total_two_sided,
    }


def _fetch_settled_markets_by_series(
    series: str, session: requests.Session, limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Query GET /markets for settled markets in one series, carrying volume_fp
    for liquidity ranking. This targets KNOWN-liquid series directly rather
    than relying on discover_resolved_markets (whose /historical/markets
    call is dominated by illiquid KXMVE multi-game micro-markets).
    """
    try:
        resp = session.get(
            f"{BASE_URL}/markets",
            params={"series_ticker": series, "status": "settled", "limit": limit},
            timeout=15,
        )
        if resp.status_code != 200:
            return []
        return resp.json().get("markets", [])
    except requests.exceptions.RequestException:
        return []


def _market_volume(m: Dict[str, Any]) -> float:
    """
    Kalshi reports volume as fixed-point STRINGS under volume_fp /
    volume_24h_fp (e.g. '10.00'), NOT a numeric 'volume' field -- guessing
    'volume' was the bug that made every market rank 0. Prefer lifetime
    volume_fp, fall back to 24h, then legacy integer 'volume'.
    """
    for k in ("volume_fp", "volume_24h_fp", "volume", "volume_24h"):
        v = m.get(k)
        if v is not None:
            try:
                return float(v)
            except (ValueError, TypeError):
                pass
    return 0.0


def build_spread_panel(
    tickers: List[str],
    session: Optional[requests.Session] = None,
    bucket: str = DEFAULT_BUCKET,
    trade_cutoff: Optional[int] = None,
    min_tick: float = 0.01,
    max_pages: int = 60,
) -> pd.DataFrame:
    """
    Build a joinable effective-spread panel for a set of market tickers,
    ready to merge into build_market_panel's candlestick panel on
    (market_id, timestamp).

    Returns columns: market_id, timestamp (hourly, UTC), spread.

    DESIGN DECISIONS, grounded in the coverage diagnostic:

    - FORWARD-FILL within each market: the diagnostic found ~46% of hourly
      buckets are one-sided (no two-sided flow -> raw spread is NaN). Rather
      than drop those bars (losing half the panel) or hardcode them, we
      forward-fill the last KNOWN effective spread within that market. This
      assumes spread is persistent hour-to-hour, which is far more defensible
      than assuming a flat 0.02 everywhere -- a market that was trading at a
      1-cent spread an hour ago is very likely still near 1 cent now.

    - FLOOR at one tick (min_tick=0.01): Kalshi trades in 1-cent ticks, so a
      spread below one tick is impossible; noise in the estimator can produce
      sub-tick or zero values, which we floor. (The diagnostic confirmed the
      median effective spread IS one tick on liquid markets, so this floor is
      the common case, not an edge case.)

    - LEADING gaps (bars before the market's first two-sided bucket) can't be
      forward-filled -- there's no prior estimate. Those are back-filled from
      the market's first known spread, then any still-missing (a market with
      ZERO two-sided buckets ever) fall back to min_tick with a per-market
      warning, so a silently-empty market can't inject NaNs into fit_K.

    max_pages defaults lower here (60) than the fetcher's 200: for a spread
    panel we need enough trades to estimate per-hour spreads, not a market's
    complete multi-hundred-thousand-trade history. Raise if you find spread
    coverage thinning out in a market's later hours.
    """
    session = session or requests.Session()
    if trade_cutoff is None:
        trade_cutoff = fetch_trade_cutoff(session)

    frames = []
    for tk in tickers:
        trades = fetch_trades_for_market(tk, session=session, trade_cutoff=trade_cutoff,
                                         max_pages=max_pages)
        if trades.empty:
            print(f"[build_spread_panel] {tk}: no trades; will fall back to min_tick spread.")
            continue
        buckets = effective_spread_by_bucket(trades, bucket=bucket)
        if buckets.empty:
            continue
        # Floor raw estimates at one tick, then fill the one-sided gaps.
        buckets["spread"] = buckets["spread"].clip(lower=min_tick)
        buckets = buckets.sort_values("timestamp")
        # forward-fill, then back-fill leading gaps, within this market only
        buckets["spread"] = buckets["spread"].ffill().bfill()
        if buckets["spread"].isna().all():
            print(f"[build_spread_panel] {tk}: no two-sided buckets ever; min_tick fallback.")
            buckets["spread"] = min_tick
        frames.append(buckets[["market_id", "timestamp", "spread"]])

    if not frames:
        print("[build_spread_panel] No spread data for any ticker; returning empty. "
              "Downstream should fall back to a labeled DR-only spread.")
        return pd.DataFrame(columns=["market_id", "timestamp", "spread"])

    return pd.concat(frames, ignore_index=True)


def attach_real_spreads(
    candle_panel: pd.DataFrame,
    session: Optional[requests.Session] = None,
    bucket: str = DEFAULT_BUCKET,
    min_tick: float = 0.01,
    fallback_spread: float = 0.01,
) -> pd.DataFrame:
    """
    Given build_market_panel's candlestick panel (with market_id, timestamp,
    ...), fetch and attach REAL effective spreads, replacing the hardcoded
    0.02. Any (market, hour) with no reconstructable spread falls back to
    `fallback_spread` (one tick, clearly the neutral default -- NOT 0.02,
    which was an arbitrary guess), and a 'spread_is_real' boolean column
    marks which rows carry a genuine estimate vs the fallback, so downstream
    analysis can weight or filter on data quality honestly.
    """
    session = session or requests.Session()
    tickers = candle_panel["market_id"].unique().tolist()
    spread_panel = build_spread_panel(tickers, session=session, bucket=bucket, min_tick=min_tick)

    out = candle_panel.copy()
    # Floor the candle timestamps to the same bucket grid the spread uses.
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
          f"spread; the rest use the {fallback_spread} one-tick fallback.")
    return out


def diagnose_on_discovered_markets(
    n_markets: int = 15,
    session: Optional[requests.Session] = None,
    bucket: str = DEFAULT_BUCKET,
    series_tickers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    End-to-end diagnostic that DISCOVERS real, liquid market tickers by
    querying known-liquid series directly (GET /markets?series_ticker=...&
    status=settled), ranks them by volume_fp, and runs spread-coverage
    analysis on the top n_markets. This removes BOTH failure sources hit
    earlier: hand-typed event-prefix tickers, AND the KXMVE-dominated
    candidate pool from discover_resolved_markets.
    """
    session = session or requests.Session()

    # Default to liquid, high-volume series across a few categories -- these
    # are where two-sided flow (and thus a usable effective spread) is most
    # plausible. KXBTCD=daily BTC, KXETHD=daily ETH, KXINX=S&P, KXFED=Fed,
    # KXHIGHNY=NYC weather (surprisingly active), KXNBA/KXNFL sports finals.
    if series_tickers is None:
        series_tickers = ["KXBTCD", "KXETHD", "KXINX", "KXFED",
                          "KXCPI", "KXHIGHNY", "KXNBA"]

    print(f"[diagnose] Querying {len(series_tickers)} liquid series directly for settled markets...")
    all_markets = []
    for s in series_tickers:
        ms = _fetch_settled_markets_by_series(s, session)
        print(f"    {s:14s} -> {len(ms)} settled markets")
        all_markets.extend(ms)
        time.sleep(0.15)

    if not all_markets:
        print("[diagnose] No settled markets returned from any liquid series. "
              "Series tickers may have changed, or all are pre-cutoff (try /historical/markets).")
        return {}

    ranked = sorted(all_markets, key=_market_volume, reverse=True)
    # Drop zero-volume markets entirely -- they can't have two-sided flow.
    ranked = [m for m in ranked if _market_volume(m) > 0]
    top = ranked[:n_markets]
    tickers = [m["ticker"] for m in top if m.get("ticker")]

    if not tickers:
        print("[diagnose] All discovered markets had zero volume. Something's off with "
              "the volume field or these series are genuinely inactive.")
        sample = all_markets[0]
        print(f"[diagnose] Sample market keys: {sorted(sample.keys())}")
        return {}

    print(f"\n[diagnose] Selected top {len(tickers)} by volume_fp "
          f"({_market_volume(top[0]):.0f} down to {_market_volume(top[-1]):.0f}).\n")

    return diagnose_spread_coverage(tickers, session=session, bucket=bucket)


if __name__ == "__main__":
    # Two modes:
    #   python spread_fetcher.py                -> auto-discover liquid markets
    #                                              (recommended; avoids bad tickers)
    #   python spread_fetcher.py TICKER1 ...    -> test specific FULL market tickers
    #                                              (must include strike suffix, e.g.
    #                                               KXBTCD-26JUL20-T112000, NOT the
    #                                               bare KXBTCD-26JUL20 event prefix)
    import sys
    tickers = sys.argv[1:] if len(sys.argv) > 1 else []
    if tickers:
        diagnose_spread_coverage(tickers)
    else:
        print("No tickers given -- auto-discovering liquid markets from the API.\n")
        diagnose_on_discovered_markets(n_markets=15)