from pathlib import Path
import logging

from google.protobuf.timestamp_pb2 import Timestamp
import grpc
from surimi.v1 import market_service_pb2_grpc, initialise_simulation_pb2, simulate_step_pb2, finalise_simulation_pb2, cancel_simulation_pb2, get_protocol_version_pb2, get_species_prices_pb2, update_sales_pb2
from .common_functions import log_and_abort
from src.simulation_handling import SimulationHandler
from server.s3_storage import S3_Storage

LOGGER_NAME = "market_service"


class MarketService(market_service_pb2_grpc.MarketServiceServicer):
    def __init__(self, simulation_dictionary: dict[str, SimulationHandler], version: str):
        self.simulation_dictionary = simulation_dictionary
        self.version = version
        self.logger = logging.getLogger(LOGGER_NAME)

    def InitialiseSimulation(self, request: initialise_simulation_pb2.InitialiseSimulationRequest, context: grpc.ServicerContext):
        self.logger.info("Initialise...")
        if request.simulation_id in self.simulation_dictionary.keys():
            log_and_abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Simulation Id {request.simulation_id} already exists.")

        # TODO: validate if the contract in request.simulation is valid
        # e.g. check if market codes and species codes exist in files etc.
        # If you don't throw an error here, the server will crash later when trying to access unknown data.

        new_simulation = SimulationHandler(request.simulation_id, request.scenario_name, request.simulation.start_date_time.ToDatetime(), request.simulation.time_step)

        self.simulation_dictionary[request.simulation_id] = new_simulation
        self.logger.info(f"Simulation ID: {request.simulation_id} started. Start time: {request.simulation.start_date_time.ToDatetime()}")

        S3_Storage.DownloadFilesFromS3("surimi-market/config", "config")

        return new_simulation.init_simulation(request.simulation)

    def SimulateStep(self, request: simulate_step_pb2.SimulateStepRequest, context: grpc.ServicerContext):
        self.logger.info(f"SimulateStep of Simulation ID: {request.simulation_id}.")
        if not request.simulation_id in self.simulation_dictionary.keys():
            log_and_abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Simulation ID: {request.simulation_id} does not exists.")

        self.logger.info(f"Current time: {request.current_date_time.ToDatetime()}")

        # TODO: validate that the current_date_time in request matches the expected time in the simulation loop. If not, throw an error to prevent out-of-sync simulation steps.
        # TODO we are not doing anything with current_date_time ??
        current_simulation = self.simulation_dictionary[request.simulation_id]
        return current_simulation.simulate_step()

    def FinaliseSimulation(self, request: finalise_simulation_pb2.FinaliseSimulationRequest, context: grpc.ServicerContext):
        self.logger.info(f"Finalise Simulation ID: {request.simulation_id}.")
        if not request.simulation_id in self.simulation_dictionary.keys():
            log_and_abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Simulation ID: {request.simulation_id} does not exists.")

        current_simulation = self.simulation_dictionary[request.simulation_id]
        message = current_simulation.finalise_simulation()

        simulation_dir = Path(__file__).parent.parent / "simulations" / request.simulation_id
        S3_Storage.UploadFilesToS3(str(simulation_dir), f"surimi-market/simulations/{request.simulation_id}")

        del self.simulation_dictionary[request.simulation_id]

        return message

    def CancelSimulation(self, request: cancel_simulation_pb2.CancelSimulationRequest, context: grpc.ServicerContext):
        self.logger.info(f"Cancel Simulation ID: {request.simulation_id}.")
        if not request.simulation_id in self.simulation_dictionary.keys():
            log_and_abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Simulation ID: {request.simulation_id} does not exists.")

        current_simulation = self.simulation_dictionary[request.simulation_id]
        del self.simulation_dictionary[request.simulation_id]

        return cancel_simulation_pb2.CancelSimulationResponse(
            simulation_id=current_simulation.simulation_id
        )

    def GetProtocolVersion(self, request: get_protocol_version_pb2.GetProtocolVersionRequest, context: grpc.ServicerContext):
        return get_protocol_version_pb2.GetProtocolVersionResponse(
            protocol_version=self.version
        )
    
    def GetSpeciesPrices(self, request: get_species_prices_pb2.GetSpeciesPricesRequest, context: grpc.ServicerContext):
        """
        Send current prices to the controller.
        """
        self.logger.info(f"GetSpeciesPrices for simulation {request.simulation_id}.")
        if not request.simulation_id in self.simulation_dictionary.keys():
            log_and_abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Simulation {request.simulation_id} does not exist.")

        current_simulation = self.simulation_dictionary[request.simulation_id]
        pricesResponse = current_simulation.get_latest_market_prices()
        pricesResponse.simulation_id = current_simulation.simulation_id

        # TODO What is the purpose of date_time if it always returns the most
        #      recent price? Should it not be the historic time of the provided
        #      date_time?
        pricesResponse.date_time = request.date_time.ToDatetime()

        self.logger.info("Prices sent.")
        return pricesResponse

    def UpdateSales(self, request: update_sales_pb2.UpdateSalesRequest, context: grpc.ServicerContext):
        self.logger.info(f"UpdateSales for simulation {request.simulation_id}.")
        if not request.simulation_id in self.simulation_dictionary.keys():
            log_and_abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Simulation {request.simulation_id} does not exist.")

        current_simulation = self.simulation_dictionary[request.simulation_id]
        val, err = current_simulation.update_sales(request)
        if err is not None:
            log_and_abort(context, err.status_code, err.message)
        else:
            return val