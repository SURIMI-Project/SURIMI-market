import datetime
from dateutil.relativedelta import *

class Tick:
    """A per-MarketSystem time tracker.

    Month-stepped only: step_months is hardcoded via MarketSystem's
    Tick(step_months=1, ...) construction, regardless of whatever
    time_step/step_size value the gRPC contract carries -- the same
    pre-existing, established behavior as the sibling elsa plugin's Tick,
    not a new limitation introduced here.
    """
    def __init__(self, start_time: datetime.datetime, step_months=1):
        self.tick = 0
        self.start_time = start_time
        self.step_months = step_months

    def increase_tick(self):
        self.tick += 1

    def get_tick(self):
        return self.tick

    def get_timestamp(self):
        """
        Get the current time in ISO format.

        Examples:
        - Tick 1 corresponds to 2013-01-01T00:00:00Z
        - Tick 2 corresponds to 2013-02-01T00:00:00Z
        """
        current_time = (self.start_time +
                        relativedelta(months=self.tick * self.step_months)).isoformat()
        return current_time

    def get_date(self):
        current_time = self.start_time + \
                        relativedelta(months=self.tick * self.step_months)
        return current_time.date()

    def get_datetime(self):
        return (self.start_time + \
                relativedelta(months=self.tick * self.step_months))
