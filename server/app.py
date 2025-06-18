import os
import signal
from concurrent import futures

import grpc

from otlp_tracing import configure_oltp_grpc_tracing
from logging_config import configure_logging
from server.ExceptionMetadataInterceptor import ExceptionMetadataInterceptor
from server.market_service import MarketService
from server.workflow_service import WorkflowService
from surimi.v1 import market_pb2_grpc, workflow_pb2_grpc

def serve():
    # Configure structured logging
    logger = configure_logging()

    # Simulation storage shared across services
    simulation_dictionary = {}

    # Configure OpenTelemetry tracing
    otel_exporter_otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    tracer = configure_oltp_grpc_tracing(endpoint=otel_exporter_otlp_endpoint)
    logger.info(f"OpenTelemetry tracing configured. Sending to: {otel_exporter_otlp_endpoint}")

    logger.info("🟧 Starting gRPC server...")

    # Create the gRPC server with interceptors
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        interceptors=[ExceptionMetadataInterceptor()]
    )

    # Register core services
    market_service = MarketService(tracer, simulation_dictionary)
    market_pb2_grpc.add_MarketServiceServicer_to_server(market_service, server)

    workflow_service = WorkflowService(tracer, simulation_dictionary)
    workflow_pb2_grpc.add_WorkflowServiceServicer_to_server(workflow_service, server)

    # Bind to host and port from environment or default
    host = os.environ.get("GRPC_HOST", "[::]")
    port = os.environ.get("GRPC_PORT", "5001")
    server.add_insecure_port(f"{host}:{port}")
    server.start()

    logger.info(f"🟩 gRPC Server running on {host}:{port}")

    # Graceful shutdown on termination signal
    def shutdown(signum, frame):
        logger.info("🟥 Shutdown signal received. Stopping server...")
        server.stop(grace=None)
        logger.info("Server stopped cleanly.")

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Wait until server is terminated
    try:
        server.wait_for_termination()
    except Exception as e:
        logger.exception("Unexpected error during server execution", exc_info=e)

if __name__ == "__main__":
    serve()
