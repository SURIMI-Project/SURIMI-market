import multiprocessing as mp
import time
import grpc
from server.market_service import serve
from client.market_client import run
from src.Market import Market, Tick

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # multiprocessing

    # Inform the market simulation
    market = Market("Western Med. Seafood Market", "EUR", "kg")

    market.add_port("Barcelona", "ESBCN")  # Barcelona, Spain
    market.add_port("Valencia", "ESVLC")  # Valencia, Spain

    market.add_species("Tuna", "TUN")
    market.add_species("Salmon", "SAL")

    # Create an event for triggering price updates
    update_prices = mp.Event()

    # Start gRPC Server in a separate process
    server_process = mp.Process(target=serve, args=(market, update_prices), daemon=True)
    server_process.start()
    time.sleep(3)  # Ensure the server is fully started

    # Start gRPC Client
    client_process = mp.Process(target=run)
    client_process.start()
    time.sleep(3)

    print("--> Running Simulation...")
    # Run the market simulation for 10 days
    for i in range(10):
        Tick.increase_tick()
        print("-" * 50)
        print(f"Update prices for Day {Tick.get_tick()} - {Tick.get_timestamp()}")
        update_prices.set()  # trigger price update
    print(" ")
    # Stop the gRPC client
    print("Stopping gRPC Client...")
    client_process.terminate()

    # Stop the gRPC server
    print("Stopping gRPC Server...")
    server_process.terminate()

    # Plot fish prices over time
    # market.action_log.plot_data()