from server.model_router import get_model_handler
from models.market.MarketHandler import MarketHandler


def test_get_model_handler_resolves_market():
    assert get_model_handler("market") is MarketHandler
