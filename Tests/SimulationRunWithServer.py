import grpc
import time
from pb import market_pb2, market_pb2_grpc
from src.market import Market, Tick

class MarketSimulator:
    def __init__(self, server_address="localhost:50051"):
        self.channel = grpc.insecure_channel(server_address)
        self.stub = market_pb2_grpc.MarketServiceStub(self.channel)

        self.market = Market(name="Global Fish Market", currency="EUR", measurement_unit="kg")
        self.market.add_port("Tokyo Port", "JPTOK")
        self.market.add_species("Tuna", "TUN001")
        self.market.add_species("Salmon", "SAL002")

    def run_simulation(self, num_days=10):
        print(f"Starting market simulation for {num_days} days...")

        for day in range(num_days):
            Tick.increase_tick()
            print(f"\n[Day {Tick.get_tick()}] Updating market prices...")

            # Generate new market prices
            self.market.send_update_prices_message()

            # Trigger the server update event
            self.trigger_update()

            # Keep listening for responses
            self.receive_updates()

            time.sleep(2)  # Simulate delay for next day's update

        print("\nSimulation completed.")

    def trigger_update(self):
        """
        Calls a gRPC method to notify the server to send updates.
        """
        try:
            request = market_pb2.EmptyRequest()
            self.stub.TriggerUpdate(request)
            print("[Simulator] Triggered market update on server.")
        except grpc.RpcError as e:
            print(f"[ERROR] gRPC Communication Failed: {e.code()} - {e.details()}")

    def receive_updates(self):
        """
        Listens for price updates from the server but does not block forever.
        """
        try:
            request = market_pb2.EmptyRequest()
            responses = self.stub.SendUpdatePrices(request)

            for response in responses:
                print(
                    f"[Simulator] Received update: {response.species_id} at {response.port_id}: {response.price} "
                    f"{response.measurement_unit} {response.currency}")

        except grpc.RpcError as e:
            print(f"[ERROR] gRPC Communication Failed: {e.code()} - {e.details()}")

if __name__ == "__main__":
    simulator = MarketSimulator()
    simulator.run_simulation(num_days=10)
