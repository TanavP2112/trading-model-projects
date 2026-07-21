from dataclasses import dataclass
from typing import List, Optional


@dataclass
class KalshiBookLevel:
    price: float
    quantity: int


class KalshiOrderBook:
    """Parses raw order book payloads and constructs synthetic asks."""

    def __init__(self, raw_book: dict):
        self.yes_bids = [
            KalshiBookLevel(p / 100.0, q) for p, q in raw_book.get("yes", [])
        ]
        self.no_bids = [
            KalshiBookLevel(p / 100.0, q) for p, q in raw_book.get("no", [])
        ]

        self.yes_asks = self._derive_asks(self.no_bids)
        self.no_asks = self._derive_asks(self.yes_bids)

    @staticmethod
    def _derive_asks(opposite_bids: List[KalshiBookLevel]) -> List[KalshiBookLevel]:
        asks = [
            KalshiBookLevel(round(1.0 - level.price, 2), level.quantity)
            for level in opposite_bids
        ]
        return sorted(asks, key=lambda x: x.price)

    @property
    def best_yes_bid(self) -> Optional[float]:
        return max([b.price for b in self.yes_bids], default=None)

    @property
    def best_yes_ask(self) -> Optional[float]:
        return min([a.price for a in self.yes_asks], default=None)

    @property
    def yes_spread(self) -> Optional[float]:
        if self.best_yes_bid is not None and self.best_yes_ask is not None:
            return round(self.best_yes_ask - self.best_yes_bid, 2)
        return None