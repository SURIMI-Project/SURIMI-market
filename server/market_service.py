from google.protobuf.timestamp_pb2 import Timestamp
import grpc
from surimi.v1 import market_pb2_grpc
from opentelemetry import trace
from common_functions import log_and_abort

class MarketService(market_pb2_grpc.MarketServiceServicer):
    def __init__(self, tracer, simulation_dictionary):
        self.simulation_dictionary = simulation_dictionary  # Will hold the current simulation instance
        self.tracer = tracer  # Store the tracer instance

    def GetSpeciesPrices(self, request, context):
        trace.get_current_span().set_attribute("simulation_id", request.simulation_id)
        print("[Server] Waiting for price update ...")
        if not request.simulation_id in self.simulation_dictionary.keys():
            log_and_abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Simulation {request.simulation_id} does not exist.")

        current_simulation = self.simulation_dictionary[request.simulation_id]
        prices = current_simulation.update_prices()
        prices.simulation_id = current_simulation.simulation_id
        return prices

    def UpdateSales(self, request, context):
        trace.get_current_span().set_attribute("simulation_id", request.simulation_id)
        print("[Server] Waiting for sales update ...")
        if not request.simulation_id in self.simulation_dictionary.keys():
            log_and_abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Simulation {request.simulation_id} does not exist.")

        current_simulation = self.simulation_dictionary[request.simulation_id]
        return current_simulation.update_sales()

