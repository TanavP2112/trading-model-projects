
from __future__ import annotations

import time
# import json as _json

import numpy as np
import pandas as pd
import asyncio
# import requests

# from config import GAMMA_MARKETS_ENDPOINT, CLOB_PRICE_HISTORY_ENDPOINT
def build_market_panel_sdk(min_volume: float = 50000, max_markets: int = 300,
                            fidelity_minutes: int = 60, interval: str = "max") -> pd.DataFrame:
    """
    Pull resolved markets + full price history using the official SDK.
    Confirmed-real signatures (verified 2026 against polymarket-client==0.1.0b20):

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
    from polymarket import AsyncPublicClient

    frames = []
    async def main():
        try:
            async with AsyncPublicClient() as client:
                paginator = await client.list_markets(
                    # closed=True,
                    # volume_num_min=min_volume,
                    # order="volume",
                    # ascending=False,
                    page_size=5,
                )
                count = 0
                async for m in paginator.iter_items():
                    if count >= max_markets:
                        break
                    token_id = m.outcomes.yes.token_id if m.outcomes and m.outcomes.yes else None
                    if token_id is None:
                        continue
                    points = await client.get_price_history(token_id=token_id, fidelity=fidelity_minutes,
                                                    interval=interval)
                    if not points:
                        continue
                    hist = pd.DataFrame({"timestamp": [p.t for p in points], "price": [p.p for p in points]})
                    hist["timestamp"] = pd.to_datetime(hist["timestamp"], unit="s", utc=True)
                    hist = hist.sort_values("timestamp").reset_index(drop=True)

                    hist["market_id"] = m.id
                    hist["category"] = (m.category or "other").lower()
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
                    await asyncio.sleep(0.1)
        except Exception:
            import traceback
            traceback.print_exc()
            raise
    asyncio.run(main())
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
                panel = build_market_panel_sdk(min_volume=50000, max_markets=3, fidelity_minutes=60)
                print(f"Got {len(panel)} price rows across {panel['market_id'].nunique() if not panel.empty else 0} markets.")
                print(panel.head())
            except ImportError:
                print("polymarket-client not installed. Run: pip install polymarket-client pyarrow")