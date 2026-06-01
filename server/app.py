import os
import signal
import grpc

from concurrent import futures
from server.ExceptionMetadataInterceptor import ExceptionMetadataInterceptor
from server.market_service import MarketService, market_service_pb2
from server.version_metadata_interceptor import VersionMetadataInterceptor
from surimi.v1 import market_service_pb2_grpc
from vault_service import VaultService
from dotenv import load_dotenv
from importlib.metadata import version, PackageNotFoundError
from logger import get_logger

logger = get_logger("app")


def serve():
    load_dotenv()

    simulation_dictionary = {}

    VaultService.LoadVaultSecretsInEnvironmentVariables()

    env_vars = "\n".join(f"  {k}: {v}" for k, v in sorted(os.environ.items()))
    logger.info(f"Environment variables:\n{env_vars}")

    version = load_version("surimi_surimi_protocol_grpc_python")
    logger.info(f"Starting gRPC server with protocol version: {version}")

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        interceptors=[ExceptionMetadataInterceptor(), VersionMetadataInterceptor(version)]
    )

    market_service = MarketService(simulation_dictionary, version)
    market_service_pb2_grpc.add_MarketServiceServicer_to_server(market_service, server)

    # The reflection service will be aware of "MarketService" and "ServerReflection" services.
    SERVICE_NAMES = (
        market_service_pb2.DESCRIPTOR.services_by_name['MarketService'].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    host = os.environ.get("GRPC_HOST", "[::]")
    port = os.environ.get("GRPC_PORT", "5001")
    server.add_insecure_port(f"{host}:{port}")
    server.start()

    logger.info(f"gRPC server running on {host}:{port}")

    def shutdown(signum, frame):
        logger.info("Shutdown signal received. Stopping server...")
        server.stop(grace=None)
        logger.info("Server stopped cleanly.")

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        server.wait_for_termination()
    except Exception as e:
        logger.error(f"Unexpected error during server execution: {e}")

def load_version(package_name: str) -> str:
    try:
        # Example: "1.78.10101.47+7fb5c33e314a"
        package_version = version(package_name)
        return package_version.split("+", 1)[1] if "+" in package_version else package_version
    except PackageNotFoundError:
        return "unknown"

    
if __name__ == "__main__":
    serve()
