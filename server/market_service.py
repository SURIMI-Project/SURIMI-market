from google.protobuf.timestamp_pb2 import Timestamp
import grpc
from surimi.v1 import market_pb2, market_pb2_grpc, species_price_pb2

class MarketService(market_pb2_grpc.MarketServiceServicer):
    def __init__(self, tracer, simulation_dictionary):
        self.simulation_dictionary = simulation_dictionary  # Will hold the current simulation instance
        self.tracer = tracer  # Store the tracer instance

    def GetSpeciesPrices(self, request, context):
        with self.tracer.start_as_current_span("MarketService.GetSpeciesPrices") as span:
            try:
                print("[Server] Waiting for price update trigger...")
                if not request.simulation_id in self.simulation_dictionary.keys():
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, "SimulationId does not exist.")
                current_simulation = self.simulation_dictionary[request.simulation_id]
                return current_simulation.send_species_prices()

            except Exception as e:
                print(f"[ERROR] gRPC Stream Error: {e}")
                context.abort(grpc.StatusCode.INTERNAL, "Server Error")


