import datetime

from src.models.market.Tick import Tick


def test_tick_advances_by_step_months():
    start = datetime.datetime(2020, 1, 1)
    tick = Tick(start_time=start, step_months=1)

    assert tick.get_tick() == 0
    assert tick.get_datetime() == start

    tick.increase_tick()
    assert tick.get_tick() == 1
    assert tick.get_datetime() == datetime.datetime(2020, 2, 1)

    tick.increase_tick()
    assert tick.get_datetime() == datetime.datetime(2020, 3, 1)


def test_tick_get_date_returns_date_not_datetime():
    start = datetime.datetime(2020, 1, 1)
    tick = Tick(start_time=start)
    assert tick.get_date() == datetime.date(2020, 1, 1)
