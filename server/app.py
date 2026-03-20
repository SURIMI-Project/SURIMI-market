import os
import signal
import grpc

from concurrent import futures
from server.ExceptionMetadataInterceptor import ExceptionMetadataInterceptor
from server.market_provider_service import MarketProviderService
from server.version_metadata_interceptor import VersionMetadataInterceptor
from server.workflow_service import WorkflowService
from surimi.v1 import market_provider_service_pb2_grpc, workflow_service_pb2_grpc
from vault_service import VaultService
from dotenv import load_dotenv
from importlib.metadata import version, PackageNotFoundError

def serve():
    load_dotenv()  # Load environment variables from .env file

    # Simulation storage shared across services
    simulation_dictionary = {}

    VaultService.LoadVaultSecretsInEnvironmentVariables()  # Load secrets from Vault into environment variables

    # Print all environment variables
    print("=" * 80)
    print("Environment Variables:")
    print("=" * 80)
    for key, value in sorted(os.environ.items()):
        print(f"{key}: {value}")
    print("=" * 80)

    version = load_version("surimi_surimi_protocol_grpc_python")
    print("Starting gRPC Server... with protocol version:", version)

    # Create the gRPC server with interceptors
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        interceptors=[ExceptionMetadataInterceptor(), VersionMetadataInterceptor(version)]
    )

    # Register core services
    market_provider_service = MarketProviderService(simulation_dictionary)
    market_provider_service_pb2_grpc.add_MarketProviderServiceServicer_to_server(market_provider_service, server)

    workflow_service = WorkflowService(simulation_dictionary, version)
    workflow_service_pb2_grpc.add_WorkflowServiceServicer_to_server(workflow_service, server)

    # Bind to host and port from environment or default
    host = os.environ.get("GRPC_HOST", "[::]")
    port = os.environ.get("GRPC_PORT", "5001")
    server.add_insecure_port(f"{host}:{port}")
    server.start()

    print(f"🟩 gRPC Server running on {host}:{port}")

    # Shutdown on termination signal
    def shutdown(signum, frame):
        print("🟥 Shutdown signal received. Stopping server...")
        server.stop(grace=None)
        print("Server stopped cleanly.")

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Wait until server is terminated
    try:
        server.wait_for_termination()
    except Exception as e:
        print(f"🟥 Unexpected error during server execution {str(e)}")

def load_version(package_name: str) -> str:
    try:
        # Example: "1.78.10101.47+7fb5c33e314a"
        package_version = version(package_name)
        return package_version.split("+", 1)[1] if "+" in package_version else package_version
    except PackageNotFoundError:
        return "unknown"

    
if __name__ == "__main__":
    serve()
