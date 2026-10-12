import threading

from src.models.market.market_trackers import CatchAccumulator


def test_add_accumulates_quantity_for_the_same_key():
    acc = CatchAccumulator()
    acc.add("ANE", "Fresh - Whole", 100.0)
    acc.add("ANE", "Fresh - Whole", 25.0)

    assert acc.snapshot()[("ANE", "Fresh - Whole")] == 125.0


def test_add_tracks_separate_keys_independently():
    acc = CatchAccumulator()
    acc.add("ANE", "Fresh - Whole", 100.0)
    acc.add("PIL", "Fresh - Whole", 50.0)

    snapshot = acc.snapshot()
    assert snapshot[("ANE", "Fresh - Whole")] == 100.0
    assert snapshot[("PIL", "Fresh - Whole")] == 50.0


def test_clear_empties_the_accumulator():
    acc = CatchAccumulator()
    acc.add("ANE", "Fresh - Whole", 100.0)

    acc.clear()

    assert acc.snapshot() == {}


def test_snapshot_returns_a_copy_not_a_live_reference():
    acc = CatchAccumulator()
    acc.add("ANE", "Fresh - Whole", 100.0)

    snapshot = acc.snapshot()
    acc.add("ANE", "Fresh - Whole", 50.0)

    assert snapshot[("ANE", "Fresh - Whole")] == 100.0
    assert acc.snapshot()[("ANE", "Fresh - Whole")] == 150.0


def test_take_returns_and_clears_atomically():
    acc = CatchAccumulator()
    acc.add("ANE", "Fresh - Whole", 100.0)

    taken = acc.take()

    assert taken[("ANE", "Fresh - Whole")] == 100.0
    assert acc.snapshot() == {}


def test_take_under_concurrent_add_never_loses_or_double_counts_quantity():
    """Many threads call add() concurrently, simulating many separate
    UpdateSales messages landing for the same tick, while the main thread
    repeatedly calls take() -- this is the exact pattern step() relies on
    to avoid the snapshot-then-clear race (a sale arriving in the gap
    between reading the accumulator and clearing it would otherwise be
    silently dropped). Regardless of how add() and take() interleave, the
    sum of everything every take() call returns, plus whatever's left at
    the very end, must equal exactly the total of everything ever added --
    nothing lost, nothing double-counted."""
    acc = CatchAccumulator()
    n_workers = 8
    adds_per_worker = 2000
    increment = 1.0
    expected_total = n_workers * adds_per_worker * increment
    key = ("ANE", "Fresh - Whole")

    def worker():
        for _ in range(adds_per_worker):
            acc.add(key[0], key[1], increment)

    threads = [threading.Thread(target=worker) for _ in range(n_workers)]
    for t in threads:
        t.start()

    collected_total = 0.0
    while any(t.is_alive() for t in threads):
        collected_total += acc.take().get(key, 0.0)

    for t in threads:
        t.join()

    # Catch anything added in the final stretch, after the last in-loop
    # take() but before every worker thread had actually finished.
    collected_total += acc.take().get(key, 0.0)

    assert collected_total == expected_total
