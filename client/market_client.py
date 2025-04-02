import sys
import time
import grpc
from pb import market_pb2, market_pb2_grpc


def trigger_price_update():
    with grpc.insecure_channel('localhost:5001') as channel:
        stub = market_pb2_grpc.MarketServiceStub(channel)

        # Send a trigger to start the price update process
        request = market_pb2.EmptyRequest()
        stub.TriggerUpdate(request)
        print("[Client] Price update triggered!")


def start_simulation(simulation_type, steps):
    with grpc.insecure_channel('localhost:5001') as channel:
        stub = market_pb2_grpc.MarketServiceStub(channel)

        # Create a request with the simulation type
        request = market_pb2.SimulationRequest(simulation_type=simulation_type)

        # Start the simulation
        stub.StartSimulation(request)
        print(f"[Client] Simulation type {simulation_type} started with {steps} steps.")

def listen_for_price_updates():
    with grpc.insecure_channel('localhost:5001') as channel:
        stub = market_pb2_grpc.MarketServiceStub(channel)

        # Send an empty request to the server to start streaming price updates
        request = market_pb2.EmptyRequest()

        # The client will listen to the stream of messages sent by the server
        try:
            response_iterator = stub.SendUpdatePrices(request)  # Get the stream of messages from the server
            for response in response_iterator:  # Iterate over the stream of messages
                # Process each message as it comes in
                print(f"[Client] Received price update: {response.species_id} at {response.port_id} - {response.price} "
                      f"{response.currency}/{response.measurement_unit} - {response.timestamp} ")
        except grpc.RpcError as e:
            print(f"[Client] Error while listening to the stream: {e.details()}")


if __name__ == "__main__":
    steps = 10
    simulation_type = "2" # Simulation 2
    start_simulation(simulation_type, steps)
    for _ in range(steps):

        trigger_price_update()  # Trigger the updates
        listen_for_price_updates()
        # wait = input("Press Enter to continue...") # Wait for user input
        time.sleep(2)  # Wait for a few seconds before triggering the next update
