from google.protobuf.timestamp_pb2 import Timestamp
import grpc
from surimi.v1 import market_pb2, market_pb2_grpc, species_price_pb2

class MarketService(market_pb2_grpc.MarketServiceServicer):
    def __init__(self, tracer):
        self.simulation = None  # Will hold the current simulation instance
        self.tracer = tracer  # Store the tracer instance

    def GetSpeciesPrices(self, request, context):
        with self.tracer.start_as_current_span("MarketService.GetSpeciesPrices") as span:
            try:
                print("[Server] Waiting for price update trigger...")

                currentTimestamp = Timestamp()
                currentTimestamp.GetCurrentTime()

                print("[Server] Finished sending updates.")
                species_prices = [
                    species_price_pb2.SpeciesPrice(species_id="BOG", price=25.50, currency="EUR",measurement_unit = "kg", port_id = "ESARN", timestamp = currentTimestamp),
                    species_price_pb2.SpeciesPrice(species_id="PIL", price=30.00, currency="EUR",measurement_unit = "kg", port_id = "ESARN", timestamp = currentTimestamp)
            ]

                return market_pb2.GetSpeciesPricesResponse( prices = species_prices)  # Return an empty response
            except Exception as e:
                print(f"[ERROR] gRPC Stream Error: {e}")
                context.abort(grpc.StatusCode.INTERNAL, "Server Error")


