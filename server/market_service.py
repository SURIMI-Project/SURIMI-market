from google.protobuf.timestamp_pb2 import Timestamp
import grpc
from concurrent import futures
from server.workflow_service import WorkflowService
from surimi.v1 import market_pb2, market_pb2_grpc, species_price_pb2, species_price_pb2_grpc, workflow_pb2, workflow_pb2_grpc

class MarketService(market_pb2_grpc.MarketServiceServicer):
    def __init__(self):
        self.simulation = None  # Will hold the current simulation instance

    def GetSpeciesPrices(self, request, context):

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

def serve():
    print("--> Starting gRPC Server...")

    # Create the server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    market_service = MarketService()  # Instantiate the market service
    market_pb2_grpc.add_MarketServiceServicer_to_server(market_service, server)

    workflow_service = WorkflowService()  # Instantiate the workflow service
    workflow_pb2_grpc.add_WorkflowServiceServicer_to_server(workflow_service, server)

    # Bind the server to a port
    server.add_insecure_port("[::]:5001")
    server.start()
    print("[OK] gRPC Server running on port 5001")

    # Wait for the server to stop
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[INFO] Server shutting down...")


if __name__ == "__main__":
    serve()
