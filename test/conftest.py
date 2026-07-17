from google.protobuf.json_format import ParseDict
import os
from pathlib import Path
import pytest
from server.MarketService import MarketService
from surimi.v1.initialise_simulation_pb2 import InitialiseSimulationRequest
from util import consts

class DummyServicerContext:
    """
    Replacement context for the grpc.ServicerContext.
    """

    def __init__(self):
        pass

    def abort(self, status_code, message):
        pass

@pytest.fixture
def grpc_context():
    return DummyServicerContext()

@pytest.fixture
def config_dir():
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def contract() -> InitialiseSimulationRequest:
    # request = InitialiseSimulationRequest()
    # data = MessageToDict(request)

    data = {
        "simulation_id": "test_initialise_market_service",
        "scenario_name": "my_scenario_name",
        "simulation": {
            "start_date_time": "2010-01-01T00:00:00Z",
            "items": {
                "markets": [
                    {"market_code": "PORT_A"},
                    {"market_code": "PORT_B"},
                    {"market_code": "PORT_C"},
                    {"market_code": "PORT_D"}
                ],
                "currencies": [
                    {"code": "EUR"}
                ],
                "species": [
                    {"species_code": "Species_A"},
                    {"species_code": "Species_B"},
                    {"species_code": "Species_C"},
                    {"species_code": "Species_D"}
                ],
                "fleet_segments": [
                    {"gear_code": "GearCode_A"},
                    {"gear_code": "GearCode_B"},
                    {"gear_code": "GearCode_C"},
                    {"gear_code": "GearCode_D"}
                ],
                "price_categories": [
                    {"category_code": "PriceCategory_A"},
                    {"category_code": "PriceCategory_B"},
                    {"category_code": "PriceCategory_C"},
                    {"category_code": "PriceCategory_D"}
                ]
            }
        }
    }

    request = InitialiseSimulationRequest()
    request = ParseDict(data, request)

    return request

@pytest.fixture
def market_service(contract, config_dir, grpc_context) -> MarketService:
    os.environ[consts.ROOT_CONFIG_DIR] = str(config_dir)
    market_service = MarketService({}, "1.0.0-test")
    assert market_service.InitialiseSimulation(contract, grpc_context), \
        "Test error! Failed to initialise simulation"
    return market_service