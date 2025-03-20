import threading
import time
import grpc
from concurrent import futures
from pb import market_pb2, market_pb2_grpc
from src.Market import Market, Tick

class MarketService(market_pb2_grpc.MarketServiceServicer):
    def __init__(self, market):
        self.market = market
        self.update_event = threading.Event()

    def SendUpdatePrices(self, request, context):
        """
        Waits for an external trigger and streams price updates for one simulation day.
        """
        try:
            print("[Server] Waiting for price update trigger...")
            self.update_event.wait()
            print("[Server] Received trigger! Sending price updates...")
            self.update_event.clear()

            messages = self.market.send_update_prices_message()
            for msg in messages:
                print(f"[Server] Sending: {msg.species_id} at {msg.port_id} - {msg.price}"
                      f" {msg.measurement_unit} {msg.currency}")
                yield msg

            print("[Server] Finished sending updates.")
        except Exception as e:
            print(f"[ERROR] gRPC Stream Error: {e}")
            context.abort(grpc.StatusCode.INTERNAL, "Server Error")

    def TriggerUpdate(self, request, context):
        """
        Allows the client to trigger price updates manually.
        """
        print("[Server] Update event triggered by client.")
        self.update_event.set()
        return market_pb2.EmptyRequest()  # Return empty response

def serve():
    print("--> Starting gRPC Server...")

    market = Market(name="Global Fish Market", currency="EUR", measurement_unit="kg")
    market.add_port("Tokyo Port", "JPTOK")
    market.add_species("Tuna", "TUN001")
    market.add_species("Salmon", "SAL002")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service = MarketService(market)
    market_pb2_grpc.add_MarketServiceServicer_to_server(service, server)

    server.add_insecure_port("[::]:50051")
    server.start()
    print("[OK] gRPC Server running on port 50051")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[INFO] Server shutting down...")

if __name__ == "__main__":
    serve()
