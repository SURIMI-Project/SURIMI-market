from google.protobuf.timestamp_pb2 import Timestamp
import grpc
from surimi.v1 import market_pb2, market_pb2_grpc, species_price_pb2, workflow_pb2

class MarketService(market_pb2_grpc.MarketServiceServicer):
    def __init__(self, tracer, simulation_dictionary):
        self.simulation_dictionary = simulation_dictionary  # Will hold the current simulation instance
        self.tracer = tracer  # Store the tracer instance

    def GetSpeciesPrices(self, request, context):
        with self.tracer.start_as_current_span("MarketService.GetSpeciesPrices") as span:
            print("[Server] Waiting for price update ...")
            if not request.simulation_id in self.simulation_dictionary.keys():
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "SimulationId does not exist.")

            current_simulation = self.simulation_dictionary[request.simulation_id]
            prices = current_simulation.update_prices()
            prices.simulation_id = current_simulation.simulation_id
            return prices

    def UpdateSales(self, request, context):
        with self.tracer.start_as_current_span("MarketService.UpdateSales") as span:
            print("[Server] Waiting for sales update ...")
            if not request.simulation_id in self.simulation_dictionary.keys():
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "SimulationId does not exist.")

            current_simulation = self.simulation_dictionary[request.simulation_id]
            return current_simulation.update_sales()

