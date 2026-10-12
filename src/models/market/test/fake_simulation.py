"""Shared fake-protobuf test scaffolding: minimal stand-ins for the
surimi.v1 Simulation/Items/Market/Species/PriceCategory protobuf messages,
used by tests and the fidelity backtest harness to construct a
MarketHandler without a real protobuf-populated Simulation object."""


class FakeMarket:
    def __init__(self, market_code):
        self.market_code = market_code


class FakeSpecies:
    def __init__(self, species_code):
        self.species_code = species_code


class FakePriceCategory:
    def __init__(self, category_code):
        self.category_code = category_code


class FakeItems:
    def __init__(self, markets, species, price_categories):
        self.markets = markets
        self.species = species
        self.price_categories = price_categories


class FakeSimulation:
    def __init__(self, items):
        self.items = items
