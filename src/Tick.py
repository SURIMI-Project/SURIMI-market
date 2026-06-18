import datetime
from dateutil.relativedelta import *

class Tick:
    """A per-MarketSystem time tracker."""
    def __init__(self, start_time: datetime.datetime, step_months=1):
        self.tick = 0
        self.start_time = start_time
        self.step_months = step_months

    def increase_tick(self):
        self.tick += 1

    def get_tick(self):
        return self.tick

    def set_timestep(self, days: int):
        self.step_months = days

    def get_timestamp(self):
        # TODO unit test this
        current_time = (self.start_time +
                        relativedelta(months=self.tick * self.step_months)).isoformat() + "Z"
        return current_time

    def reset(self):
        self.tick = 0
        self.start_time = datetime.datetime.utcnow()

