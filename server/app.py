import os
import signal
import grpc

from concurrent import futures
from otlp_tracing import configure_oltp_grpc_tracing
from server.ExceptionMetadataInterceptor import ExceptionMetadataInterceptor
from server.market_provider_service import MarketProviderService
from server.workflow_service import WorkflowService
from surimi.v1 import market_provider_service_pb2_grpc, workflow_service_pb2_grpc
from vault_service import VaultService
from dotenv import load_dotenv

def serve():
    load_dotenv()  # Load environment variables from .env file

    # Simulation storage shared across services
    simulation_dictionary = {}

    VaultService.LoadVaultSecretsInEnvironmentVariables()  # Load secrets from Vault into environment variables

    # Configure OpenTelemetry tracing
    otel_exporter_otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    tracer = configure_oltp_grpc_tracing(endpoint=otel_exporter_otlp_endpoint)
    print(f"OpenTelemetry tracing configured. Sending to: {otel_exporter_otlp_endpoint}")

    print("🟧 Starting gRPC server...")

    # Create the gRPC server with interceptors
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        interceptors=[ExceptionMetadataInterceptor()]
    )

    # Register core services
    market_provider_service = MarketProviderService(tracer, simulation_dictionary)
    market_provider_service_pb2_grpc.add_MarketProviderServiceServicer_to_server(market_provider_service, server)

    workflow_service = WorkflowService(tracer, simulation_dictionary)
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

if __name__ == "__main__":
    serve()
