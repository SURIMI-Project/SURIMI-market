import datetime

import pytest

from src.MarketSystem import MarketSystem
from src.LocalMarket import LocalMarket
from surimi.v1 import sales_pb2, species_pb2, fleet_segment_pb2


@pytest.fixture
def correct_market_sales():
    species = species_pb2.Species(
        species_code="RJC"
    )

    fleet_segment = fleet_segment_pb2.FleetSegment(
        gear_code="OTB"
    )

    sales = sales_pb2.Sale(
        species=species,
        fleet_segment=fleet_segment,
        quantity=3.5054,
        value=8.22
    )

    market_sales = sales_pb2.MarketSales(
        market_code="ESBRX",
        currency="EUR",
        sales=[
            sales
        ]
    )
    return [market_sales]


@pytest.fixture
def negative_market_sales_catch():
    species = species_pb2.Species(
        species_code="RJC"
    )

    fleet_segment = fleet_segment_pb2.FleetSegment(
        gear_code="OTB"
    )

    sales = sales_pb2.Sale(
        species=species,
        fleet_segment=fleet_segment,
        quantity=-3.5054,
        value=8.22
    )

    market_sales = sales_pb2.MarketSales(
        market_code="ESBRX",
        currency="EUR",
        sales=[
            sales
        ]
    )
    return [market_sales]


def test_negative_input_quantity(
        correct_market_sales,
        negative_market_sales_catch
):
    dt = datetime.datetime(2013, 1, 1)
    dt_in_seconds = int(dt.timestamp())
    market_system = MarketSystem("EUR", dt)
    market_system.local_markets["ESBRX"] = LocalMarket("ESBRX")

    err = market_system.update_sales(correct_market_sales, dt_in_seconds)
    assert err is None, err.message

    err = market_system.update_sales(negative_market_sales_catch, dt_in_seconds)
    assert err is not None, "Error should not be None"
