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
