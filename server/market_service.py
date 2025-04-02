import threading
import grpc
from concurrent import futures
from pb import market_pb2, market_pb2_grpc
from src.simulation import Simulation  # Assuming Simulation class is in simulation.py


class MarketService(market_pb2_grpc.MarketServiceServicer):
    def __init__(self):
        self.simulation = None  # Will hold the current simulation instance

    def SendUpdatePrices(self, request, context):
        """
        Waits for an external trigger and streams price updates for one simulation day.
        """
        try:
            print("[Server] Waiting for price update trigger...")
            self.simulation.update_event.wait()  # Wait for the client to trigger the update
            print("[Server] Received trigger! Sending price updates...")
            self.simulation.update_event.clear()

            # Stream price update messages from the simulation
            for msg in self.simulation.latest_market_prices:
                print(f"[Server] Sending: {msg.species_id} at {msg.port_id} - {msg.price} "
                      f"{msg.currency}/{msg.measurement_unit} - {msg.timestamp} ")
                yield msg  # Stream message to the client

            print("[Server] Finished sending updates.")
        except Exception as e:
            print(f"[ERROR] gRPC Stream Error: {e}")
            context.abort(grpc.StatusCode.INTERNAL, "Server Error")

    def TriggerUpdate(self, request, context):
        """
        Allows the client to trigger price updates manually.
        """
        print("[Server] Update event triggered by client.")
        self.simulation.update_event.set()  # Trigger the event to start sending updates
        self.simulation.trigger_simulation()  # Now trigger the simulation to send updates
        return market_pb2.EmptyRequest()

    def StartSimulation(self, request, context):
        """
        Starts a new simulation based on the type sent from the client.
        """
        try:
            simulation_type = request.simulation_type  # Simulation type received from the client
            print(f"[Server] Starting simulation of type {simulation_type}...")

            # Start the chosen simulation
            self.simulation = Simulation.pick_simulation(simulation_type)

            # The simulation is initialized but **not triggered** yet
            print(f"[Server] Simulation type {simulation_type} started, waiting for client to trigger updates.")

            return market_pb2.EmptyRequest()
        except ValueError as e:
            print(f"[ERROR] Invalid simulation type: {e}")
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid simulation type")
        except Exception as e:
            print(f"[ERROR] Error starting simulation: {e}")
            context.abort(grpc.StatusCode.INTERNAL, "Server Error")


def serve():
    print("--> Starting gRPC Server...")

    # Create the server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service = MarketService()  # Instantiate the service
    market_pb2_grpc.add_MarketServiceServicer_to_server(service, server)

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
