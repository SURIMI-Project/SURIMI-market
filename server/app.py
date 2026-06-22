import os
import signal
import grpc
import argparse
import sys
from pathlib import Path


from concurrent import futures
from server.ExceptionMetadataInterceptor import ExceptionMetadataInterceptor
from server.MarketService import MarketService
from server.version_metadata_interceptor import VersionMetadataInterceptor
from surimi.v1 import market_service_pb2_grpc, market_service_pb2
from vault_service import VaultService
from dotenv import load_dotenv
from importlib.metadata import version, PackageNotFoundError
import logging

from grpc_reflection.v1alpha import reflection

from util import consts

def serve():
    load_dotenv()
    logger = logging.getLogger(consts.APP_LOGGER_NAME)


    # TODO Talk to Rik about why the simulation_directionary belongs to app and not
    # to MarketService
    simulation_dictionary = {}

    VaultService.LoadVaultSecretsInEnvironmentVariables()

    # TODO investigate why we want to show env vars
    env_vars = "\n".join(f"  {k}: {v}" for k, v in sorted(os.environ.items()))
    logger.debug(f"Environment variables:\n{env_vars}")

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

    host = os.environ.get("GRPC_HOST", consts.GRPC_HOST)
    port = os.environ.get("GRPC_PORT", consts.GRPC_PORT)
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

def configure_logging(is_debug_enabled: bool):
    _LOG_DIR = Path(__file__).resolve().parent / "logs"
    _LOG_DIR.mkdir(exist_ok=True)

    _FORMAT = "%(asctime)s %(levelname)-8s %(name)s - %(message)s"
    _DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.DEBUG if is_debug_enabled else logging.INFO,
        format=_FORMAT,
        datefmt=_DATE_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(_LOG_DIR / "surimi.log"),
        ],
    )

    logging.getLogger(consts.APP_LOGGER_NAME).debug("Configuring logger")

def parse_cli():
    parser = argparse.ArgumentParser(
        description="""The Market model is a dynamic, supply driven model.
        
        The primary goal of this interface is to enable the debug mode."""
    )

    parser.add_argument(
        "-v", "--version",
        help="Print the current version",
        action="store_true"
    )

    parser.add_argument(
        "-d", "--debug",
        help="Enable debug mode. This prints verbose debug messages",
        action="store_true"
    )

    args = parser.parse_args()

    if args.version:
        print(load_version("surimi_surimi_protocol_grpc_python"))
        sys.exit(0)


    else:
        configure_logging(args.debug)
        serve()


if __name__ == "__main__":
    parse_cli()
