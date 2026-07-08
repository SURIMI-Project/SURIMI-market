import datetime

import pytest

from src.MarketSystem import MarketSystem
from src.LocalMarket import LocalMarket
from surimi.v1 import sales_pb2, species_pb2, price_category_pb2


@pytest.fixture
def correct_market_sales():
    species = species_pb2.Species(
        species_code="RJC"
    )

    price_category = price_category_pb2.PriceCategory(
        category_code="Fresh - Whole"
    )

    sales = sales_pb2.Sale(
        species=species,
        category_code=price_category.category_code,
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

    price_category = price_category_pb2.PriceCategory(
        category_code="Fresh - Whole"
    )

    sales = sales_pb2.Sale(
        species=species,
        category_code=price_category.category_code,
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
    dt = datetime.datetime(2013, 1, 1, tzinfo=datetime.UTC)
    dt_in_seconds = int(dt.timestamp())
    market_system = MarketSystem("EUR", dt)
    market_system.local_markets["ESBRX"] = LocalMarket("ESBRX")

    err = market_system.update_sales(correct_market_sales, dt_in_seconds)
    assert err is None, err.message

    err = market_system.update_sales(negative_market_sales_catch, dt_in_seconds)
    assert err is not None, "Error should not be None"
