import pytest
import pandas as pd

from src.models.elsa import ElsaHandler
from src.models.elsa import LocalMarket

@pytest.fixture
def markets():
    return {
        "ESSCR": LocalMarket.LocalMarket("ESSCR"),
        "ESPAL": LocalMarket.LocalMarket("ESPAL"),
    }

@pytest.fixture
def dataset():
    return pd.DataFrame({
        "date": [
            "2013-01-03", "2015-01-03", "2012-01-03", "2009-01-03",
            "2013-01-03", "2015-01-03", "2012-01-03", "2009-01-03"
        ],
        "market_code": [
            "ESSCR", "ESSCR", "ESSCR", "ESSCR", 
            "ESPAL", "ESPAL", "ESPAL", "ESPAL"
        ],
        "species_code": [
            "ANE", "ANE", "PIL", "PIL",
            "ANE", "ANE", "PIL", "PIL"
        ],
        "category_code": [
            "OTB", "OTB", "PS", "PS",
            "OTB", "OTB", "PS", "PS"
        ],
        "price": [
            2.3, 1.7, 1.8, 0.0,
            0.0, 0.0, 10.1, 13.0
        ],
        "currency": [
            "EUR","EUR","EUR","EUR",
            "EUR","EUR","EUR","EUR"
        ],
        "measurement_unit": [
            "kg","kg","kg","kg",
            "kg","kg","kg","kg"
        ]
    })

# The dataset in CSV:
# date,market_code,species_code,category_code,price,currency,measurement_unit
# 2013-01-03,ESSCR,ANE,OTB,2.3,EUR,kg
# 2015-01-03,ESSCR,ANE,OTB,1.7,EUR,kg
# 2012-01-03,ESSCR,PIL,PS,1.8,EUR,kg
# 2009-01-03,ESSCR,PIL,PS,0.0,EUR,kg
# 2013-01-03,ESPAL,ANE,OTB,0.0,EUR,kg
# 2015-01-03,ESPAL,ANE,OTB,0.0,EUR,kg
# 2012-01-03,ESPAL,PIL,PS,10.0,EUR,kg
# 2009-01-03,ESPAL,PIL,PS,-1,EUR,kg

@pytest.fixture
def species_codes():
    return ["ANE", "HKE", "PIL", "ANK"]

@pytest.fixture
def category_codes():
    return ["OTB", "OTB", "PS", "RECT"]

def test_set_initial_price_to_earliest_available(markets, dataset, species_codes, category_codes):
    sh = ElsaHandler.ElsaHandler("0000", "", "", 0)
    sh.market_system.local_markets = {}
    sh.market_system.local_markets = markets
    sh.market_system.load_local_market_price_history(dataset)

    for market in markets.values():
        market.set_initial_price_to_earliest_available(species_codes, category_codes)
    
    assert markets.get("ESSCR").local_prices.get(("ANE", "OTB")) == 1.7
    assert markets.get("ESPAL").local_prices.get(("ANE", "OTB")) == None

    assert markets.get("ESSCR").local_prices.get(("ANE", "PS")) == None
    assert markets.get("ESPAL").local_prices.get(("ANE", "PS")) == None
     
    assert markets.get("ESSCR").local_prices.get(("PIL", "OTB")) == None
    assert markets.get("ESPAL").local_prices.get(("PIL", "OTB")) == None

    assert markets.get("ESPAL").local_prices.get(("PIL", "PS")) == 13.0
    assert markets.get("ESSCR").local_prices.get(("PIL", "PS")) == 1.8
