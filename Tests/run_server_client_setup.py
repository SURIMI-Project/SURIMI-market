import subprocess
import time

# Start gRPC Server
server_process = subprocess.Popen(["python", "-m", "server.market_service"], creationflags=subprocess.CREATE_NEW_CONSOLE)
time.sleep(2)  # Wait for server to start

# Start Simulation
simulation_process = subprocess.Popen(["python", "-m", "src.SimulationRunWithServer"])

# Start gRPC Client
client_process = subprocess.Popen(["python", "-m", "client.market_client"], creationflags=subprocess.CREATE_NEW_CONSOLE)
time.sleep(3)  # Wait for client to initialize


simulation_process.wait()  # Wait for simulation to finish

# Stop server and client after simulation completes
print("\nStopping background processes...")
server_process.terminate()
# client_process.terminate()

print("Market simulation completed!")
