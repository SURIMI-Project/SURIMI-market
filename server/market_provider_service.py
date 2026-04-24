from google.protobuf.timestamp_pb2 import Timestamp
import grpc
from surimi.v1 import market_provider_service_pb2, market_provider_service_pb2_grpc
from common_functions import log_and_abort
from src.simulation_handling import SimulationHandler
from logger import get_logger

logger = get_logger("market_provider_service")


class MarketProviderService(market_provider_service_pb2_grpc.MarketProviderServiceServicer):
    def __init__(self, simulation_dictionary: dict[str, SimulationHandler]):
        self.simulation_dictionary = simulation_dictionary

    def GetSpeciesPrices(self, request: market_provider_service_pb2.GetSpeciesPricesRequest, context: grpc.ServicerContext):
        logger.info(f"GetSpeciesPrices for simulation {request.simulation_id}.")
        if not request.simulation_id in self.simulation_dictionary.keys():
            log_and_abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Simulation {request.simulation_id} does not exist.")

        current_simulation = self.simulation_dictionary[request.simulation_id]
        prices = current_simulation.update_prices()
        prices.simulation_id = current_simulation.simulation_id
        return prices

    def UpdateSales(self, request: market_provider_service_pb2.UpdateSalesRequest, context: grpc.ServicerContext):
        logger.info(f"UpdateSales for simulation {request.simulation_id}.")
        if not request.simulation_id in self.simulation_dictionary.keys():
            log_and_abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Simulation {request.simulation_id} does not exist.")

        current_simulation = self.simulation_dictionary[request.simulation_id]
        return current_simulation.update_sales(request)
