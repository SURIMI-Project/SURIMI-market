import logging
import os
from otlp_tracing import configure_oltp_grpc_tracing

from concurrent import futures
import grpc
from server.market_service import MarketService
from server.workflow_service import WorkflowService
from surimi.v1 import market_pb2_grpc, workflow_pb2_grpc

def serve():
    logging.basicConfig(level=logging.INFO)

    otel_exporter_otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or "http://localhost:4317"

    tracer = configure_oltp_grpc_tracing( endpoint=otel_exporter_otlp_endpoint)  # Configure OpenTelemetry tracing
    logger = logging.getLogger(__name__)
    
    logger.info("Sending OpenTelemetry data to endpoint: " + otel_exporter_otlp_endpoint)

    logger.info("Starting gRPC Server...")

    # Create the server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    market_service = MarketService(tracer)  # Instantiate the market service
    market_pb2_grpc.add_MarketServiceServicer_to_server(market_service, server)

    workflow_service = WorkflowService(tracer)  # Instantiate the workflow service
    workflow_pb2_grpc.add_WorkflowServiceServicer_to_server(workflow_service, server)

    # Bind the server to a port
    server.add_insecure_port("[::]:5001")
    server.start()
    logger.info("[OK] gRPC Server running on port 5001")

    # Wait for the server to stop
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("\n[INFO] Server shutting down...")


if __name__ == "__main__":
    serve()