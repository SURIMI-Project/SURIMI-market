"""Thread-safe accumulator for catch quantities routed to a market between
ticks -- the same pattern CombinedMarket and LocalMarket each need for
their own catch bookkeeping."""
from collections import defaultdict
import threading


class CatchAccumulator:
    def __init__(self):
        self._totals: dict[tuple[str, str], float] = defaultdict(float)
        self._lock = threading.Lock()

    def add(self, species_code: str, category_code: str, quantity: float):
        with self._lock:
            self._totals[(species_code, category_code)] += quantity

    def clear(self):
        with self._lock:
            self._totals.clear()

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._totals)
