"""
Real Polymarket data ingestion.

Two paths, both zero-auth (public reads only):

  1. DEFAULT / build_market_panel(): the official `polymarket-client` SDK
     (currently beta). Typed models, confirmed real parameter names,
     built-in pagination, and an optional to_pandas() flattening helper.
     Verified against polymarket-client==0.1.0b20 by installing it and
     introspecting the actual method signatures / pydantic model fields
     directly -- not guessed from docs prose -- AND confirmed working
     against live data. (An earlier test against this same logic reported
     0 markets; that appears no longer to reproduce, and since this path
     has no artificial pagination ceiling the way the raw-REST fallback
     did, it's now the default rather than a fallback.) Install with:
         pip install polymarket-client pyarrow

  2. build_market_panel_raw_rest(): raw `requests` calls against the
     public REST endpoints, for when you don't want the extra dependency.
     Fixed here to correctly json.loads() the 'outcomePrices' /
     'clobTokenIds' fields, which Polymarket returns as STRINGIFIED JSON,
     not native lists/arrays -- confirmed against Polymarket's own example
     client (github.com/Polymarket/agents). Has a max_pages parameter
     since its hand-rolled pagination previously had a silent 2,000-market
     scan ceiling regardless of max_markets requested -- fixed, but still
     worth knowing this path needed that fix and the SDK path didn't.

IMPORTANT: this sandbox's network egress does not include polymarket.com,
so neither path has been execution-tested end-to-end against the live
API from THIS environment (the SDK's *installation* and *object model*
were verified locally against the real package; live HTTP calls were not).
The user has independently confirmed the SDK path works against live data.
"""

from __future__ import annotations

import time
import json as _json

import numpy as np
import pandas as pd
import requests

from config import GAMMA_MARKETS_ENDPOINT, CLOB_PRICE_HISTORY_ENDPOINT


# ===========================================================================
# PATH 1 (preferred): official polymarket-client SDK
# ===========================================================================
def _resolve_category(m) -> str:
    """
    m.category is a genuinely optional field (str | None, confirmed via
    direct model introspection) that Polymarket's API frequently leaves
    empty -- confirmed necessary after a real fetch came back with every
    single market categorized as "other". The real categorization lives in
    m.tags (a list of MarketTag(id, slug, label) objects) instead. This
    scans tag labels/slugs for a match against the known fee-schedule
    categories (see config.FEE_RATE_BY_CATEGORY) before giving up.

    Falling back to "other" incorrectly isn't just a labeling nuisance --
    it directly mis-prices every affected trade's fees (a crypto market
    silently charged the "other" rate instead of crypto's, for example),
    so this matters for backtest accuracy, not just readability.
    """
    if m.category:
        return m.category.lower()
    known_categories = {"crypto", "sports", "finance", "politics", "mentions",
                        "tech", "economics", "culture", "weather", "geopolitics"}
    for tag in getattr(m, "tags", ()) or ():
        for candidate in (getattr(tag, "label", None), getattr(tag, "slug", None)):
            if candidate and candidate.lower() in known_categories:
                return candidate.lower()
    return "other"


def build_market_panel(min_volume: float = 50_000, max_markets: int = 300,
                        fidelity_minutes: int = 60, interval: str = "max",
                        checkpoint_path: str | None = "data/_fetch_checkpoint.parquet",
                        checkpoint_every: int = 50, max_retries: int = 3,
                        tag_slug: str | None = None) -> pd.DataFrame:
    """
    Pull resolved markets + full price history using the official SDK.
    Confirmed-real signatures (verified 2026 against polymarket-client==0.1.0b20),
    and confirmed WORKING against live data (returned real markets in testing,
    unlike an earlier unresolved 0-markets report against this same logic --
    whatever caused that appears no longer to apply):

        PublicClient()  -- no credentials needed for public reads
        client.list_markets(closed=True, volume_num_min=..., order="volume",
                             ascending=False, page_size=100) -> Paginator[Market]
        client.get_price_history(token_id=..., fidelity=..., interval=...)
            -> tuple[PriceHistoryPoint(t: int, p: float), ...]

    Market model (confirmed fields): m.id, m.category, m.state.end_date,
    m.metrics.volume, m.outcomes.yes.token_id, m.outcomes.yes.price (final
    settled price for a closed market -- ~1.0 if YES won, ~0.0 if it lost;
    there is no separate boolean "did YES win" field in this SDK version,
    so the settled yes-price IS the resolution signal).
    """
    from polymarket import PublicClient  # local import: optional dependency
    import os as _os
    import time as _time

    print(f"      Starting fetch: target up to {max_markets} markets "
          f"(min_volume={min_volume:,.0f}, fidelity={fidelity_minutes}min)...")
    fetch_start = _time.time()

    # Resume support: if a checkpoint exists, load it and skip any market_id
    # already fetched. This is what turns a connection-drop 87% through a
    # 40+ minute run into "lose a few minutes," not "lose the whole run" --
    # confirmed necessary after exactly that happened (a mid-run
    # ConnectionTerminated error at 1300/1500 markets with nothing saved
    # until the very end lost the entire 43 minutes of progress).
    already_fetched_ids = set()
    frames = []
    if checkpoint_path and _os.path.exists(checkpoint_path):
        try:
            cached = pd.read_parquet(checkpoint_path)
            already_fetched_ids = set(cached["market_id"].unique())
            frames.append(cached)
            print(f"      Resuming from checkpoint: {len(already_fetched_ids)} markets already fetched, "
                  f"will skip these and continue from where the last run stopped.")
        except Exception as e:
            print(f"      Could not load checkpoint ({e}) -- starting fresh.")

    def _new_client():
        return PublicClient()

    client = _new_client()

    # Server-side category filtering: look up the real tag_id at runtime via
    # get_tag(slug=...) rather than hardcoding a guessed numeric ID, which
    # is more robust and self-corrects if Polymarket ever changes tag IDs.
    # This is meaningfully better than fetching broadly and filtering
    # client-side afterward -- your whole max_markets budget goes toward
    # the category you actually want, instead of being diluted across every
    # category and then thrown away.
    resolved_tag_id = None
    if tag_slug is not None:
        try:
            tag = client.get_tag(slug=tag_slug)
            resolved_tag_id = int(tag.id)
            print(f"      Resolved tag_slug='{tag_slug}' -> tag_id={resolved_tag_id} "
                  f"(label='{tag.label}')")
        except Exception as e:
            print(f"      !! Could not resolve tag_slug='{tag_slug}' ({type(e).__name__}: {e}) -- "
                  f"proceeding WITHOUT a category filter. Check the slug is correct "
                  f"(try client.list_tags() to see available slugs) before trusting this run's category label.")

    count = len(already_fetched_ids)
    scanned = 0
    since_checkpoint = 0
    try:
        paginator = client.list_markets(
            closed=True,
            volume_num_min=min_volume,
            order="volume",
            ascending=False,
            page_size=100,
            **({"tag_id": resolved_tag_id} if resolved_tag_id is not None else {}),
        )
        for m in paginator.iter_items():
            if count >= max_markets:
                break
            scanned += 1
            if m.id in already_fetched_ids:
                continue
            token_id = m.outcomes.yes.token_id if m.outcomes and m.outcomes.yes else None
            if token_id is None:
                continue

            # Retry with a FRESH client on failure -- a connection-level error
            # (like the ConnectionTerminated seen in practice) likely means the
            # underlying connection pool is dead, so retrying on the SAME
            # client object could just fail again immediately. Recreating the
            # client forces a genuinely fresh connection.
            points = None
            for attempt in range(max_retries):
                try:
                    points = client.get_price_history(token_id=token_id, fidelity=fidelity_minutes,
                                                       interval=interval)
                    break
                except Exception as e:
                    wait = min(2 ** attempt, 30)
                    print(f"      !! network error on market {m.id} (attempt {attempt+1}/{max_retries}): "
                          f"{type(e).__name__}: {e} -- reconnecting and retrying in {wait}s")
                    _time.sleep(wait)
                    try:
                        client.close()
                    except Exception:
                        pass
                    client = _new_client()
            if points is None:
                print(f"      !! giving up on market {m.id} after {max_retries} attempts, skipping it.")
                continue
            if not points:
                continue

            hist = pd.DataFrame({"timestamp": [p.t for p in points], "price": [p.p for p in points]})
            hist["timestamp"] = pd.to_datetime(hist["timestamp"], unit="s", utc=True)
            hist = hist.sort_values("timestamp").reset_index(drop=True)

            hist["market_id"] = m.id
            hist["category"] = tag_slug.lower() if tag_slug else _resolve_category(m)
            hist["volume"] = float(m.metrics.volume) if m.metrics and m.metrics.volume is not None else 0.0

            end_date = m.state.end_date if m.state else None
            if end_date is not None:
                hist["days_to_resolution"] = (end_date - hist["timestamp"]).dt.total_seconds() / 86400.0
            else:
                hist["days_to_resolution"] = np.nan

            yes_final = m.outcomes.yes.price
            outcome = int(float(yes_final) >= 0.5) if yes_final is not None else int(hist["price"].iloc[-1] >= 0.5)
            hist["outcome"] = outcome

            frames.append(hist)
            count += 1
            since_checkpoint += 1

            if checkpoint_path and since_checkpoint >= checkpoint_every:
                pd.concat(frames, ignore_index=True).to_parquet(checkpoint_path, index=False)
                since_checkpoint = 0
                print(f"      [checkpoint saved: {count} markets -- a crash now loses at most "
                      f"{checkpoint_every} markets of progress, not the whole run]")

            if count % 50 == 0 or count == max_markets:
                elapsed = _time.time() - fetch_start
                new_this_run = count - len(already_fetched_ids)
                rate = new_this_run / elapsed if elapsed > 0 and new_this_run > 0 else 0
                remaining = (max_markets - count) / rate if rate > 0 else float("nan")
                print(f"      ...{count}/{max_markets} markets fetched "
                      f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining at current rate)")
            time.sleep(0.1)  # polite pacing even though there's no auth/rate-limit requirement documented
    finally:
        try:
            client.close()
        except Exception:
            pass

    print(f"      Fetch complete: {count} markets with usable price history "
          f"out of {scanned} scanned, in {_time.time() - fetch_start:.0f}s.")

    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    if checkpoint_path:
        result.to_parquet(checkpoint_path, index=False)
    return result

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ===========================================================================
# PATH 2 (fallback): raw REST, no extra dependency
# ===========================================================================
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "pm-quant-research/0.2"})


def fetch_resolved_markets(min_volume: float = 50_000, limit: int = 100,
                            max_pages: int = 20, sleep_s: float = 0.2) -> list[dict]:
    """
    Raw Gamma API pull. NOTE: 'clobTokenIds' and 'outcomePrices' come back
    as JSON-encoded STRINGS (e.g. '["111","222"]'), not native lists --
    confirmed against Polymarket's own example repo (agents/polymarket/gamma.py),
    which explicitly calls json.loads() on both fields before use. This
    function does that parsing for you.
    """
    all_markets = []
    offset = 0
    for _ in range(max_pages):
        params = {"closed": "true", "limit": limit, "offset": offset}
        resp = SESSION.get(GAMMA_MARKETS_ENDPOINT, params=params, timeout=20)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        for m in batch:
            vol = float(m.get("volume", 0) or 0)
            if vol < min_volume:
                continue
            for key in ("clobTokenIds", "outcomePrices"):
                if key in m and isinstance(m[key], str):
                    try:
                        m[key] = _json.loads(m[key])
                    except (TypeError, ValueError):
                        pass
            all_markets.append(m)
        if len(batch) < limit:
            break
        offset += limit
        time.sleep(sleep_s)
    return all_markets


def fetch_price_history(clob_token_id: str, fidelity_minutes: int = 60,
                         interval: str = "max",
                         start_ts: int | None = None, end_ts: int | None = None) -> pd.DataFrame:
    """
    Raw CLOB API pull. `interval` is effectively required by the API
    (confirmed via Polymarket's own SDK type signature, which does not
    default it) -- 'max' pulls the full available history for the token.
    """
    params = {"market": clob_token_id, "fidelity": fidelity_minutes, "interval": interval}
    if start_ts is not None:
        params["startTs"] = start_ts
    if end_ts is not None:
        params["endTs"] = end_ts
    resp = SESSION.get(CLOB_PRICE_HISTORY_ENDPOINT, params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    points = payload.get("history", payload if isinstance(payload, list) else [])
    df = pd.DataFrame(points)
    if df.empty:
        return df
    df = df.rename(columns={"t": "timestamp", "p": "price"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["price"] = df["price"].astype(float)
    return df.sort_values("timestamp").reset_index(drop=True)


def build_market_panel_raw_rest(min_volume: float = 50_000, max_markets: int = 300,
                                 fidelity_minutes: int = 60, max_pages: int = 20) -> pd.DataFrame:
    """
    Fallback (no-SDK) version of build_market_panel_sdk. Prefer the SDK version.

    IMPORTANT: max_pages controls how many raw closed markets get SCANNED
    before filtering by min_volume (max_pages * 100 per page). This is a
    completely separate cap from max_markets (which only limits how many
    markets survive AFTER the volume filter). Raising max_markets alone does
    NOTHING if you're hitting the max_pages ceiling first -- e.g. the
    default max_pages=20 means at most 2,000 raw markets are ever scanned,
    so if your min_volume filter is strict relative to Polymarket's overall
    volume distribution, you can end up with far fewer markets than
    max_markets requested, no matter how high you set it. If you're getting
    suspiciously few markets back, raise max_pages first and check the
    printed scan-funnel numbers below before assuming min_volume is the
    problem.
    """
    raw_markets = fetch_resolved_markets(min_volume=min_volume, limit=100, max_pages=max_pages)
    print(f"      [fetch funnel] scanned up to {max_pages * 100} raw closed markets, "
          f"{len(raw_markets)} cleared min_volume={min_volume:,.0f}"
          + (" -- consider raising max_pages if this feels low" if len(raw_markets) < max_markets else ""))
    markets = raw_markets[:max_markets]
    frames = []
    for m in markets:
        token_ids = m.get("clobTokenIds")
        if not token_ids:
            continue
        token_id = token_ids[0] if isinstance(token_ids, list) else token_ids
        hist = fetch_price_history(token_id, fidelity_minutes=fidelity_minutes)
        if hist.empty:
            continue
        hist["market_id"] = m.get("conditionId", m.get("id"))
        hist["category"] = (m.get("category") or "other").lower()
        hist["volume"] = float(m.get("volume", 0) or 0)
        end_date = pd.to_datetime(m.get("endDate"), utc=True, errors="coerce")
        hist["days_to_resolution"] = (end_date - hist["timestamp"]).dt.total_seconds() / 86400.0
        outcome_prices = m.get("outcomePrices")
        try:
            hist["outcome"] = int(float(outcome_prices[0]) >= 0.5) if outcome_prices else int(hist["price"].iloc[-1] >= 0.5)
        except (TypeError, ValueError, IndexError):
            hist["outcome"] = int(hist["price"].iloc[-1] >= 0.5)
        frames.append(hist)
        time.sleep(0.15)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    import sys
    if "--smoke-test" in sys.argv:
        use_sdk = "--raw" not in sys.argv
        if use_sdk:
            try:
                print("Smoke-testing via the official polymarket-client SDK...")
                panel = build_market_panel(min_volume=100_000, max_markets=3, fidelity_minutes=60)
                print(f"Got {len(panel)} price rows across {panel['market_id'].nunique() if not panel.empty else 0} markets.")
                print(panel.head())
            except ImportError:
                print("polymarket-client not installed. Run: pip install polymarket-client pyarrow")
        else:
            print("Smoke-testing via raw REST fallback...")
            ms = fetch_resolved_markets(min_volume=100_000, limit=5, max_pages=1)
            print(f"Got {len(ms)} markets. First: {ms[0].get('question') if ms else 'N/A'}")
            if ms:
                token_ids = ms[0].get("clobTokenIds")
                token_id = token_ids[0] if isinstance(token_ids, list) else token_ids
                hist = fetch_price_history(token_id)
                print(f"Price history points: {len(hist)}")
                print(hist.head())
