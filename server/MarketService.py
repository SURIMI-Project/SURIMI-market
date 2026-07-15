import datetime
import logging
import os

import grpc
from surimi.v1 import market_service_pb2_grpc, initialise_simulation_pb2, simulate_step_pb2, finalise_simulation_pb2, cancel_simulation_pb2, get_protocol_version_pb2, get_species_prices_pb2, update_sales_pb2
from src.SimulationHandler import SimulationHandler
from server.s3_storage import S3_Storage
from util import consts

class MarketService(market_service_pb2_grpc.MarketServiceServicer):
    def __init__(self, simulation_dictionary: dict[str, SimulationHandler], version: str):
        self.simulation_dictionary = simulation_dictionary
        self.version = version
        self.logger = logging.getLogger(consts.MARKET_SERVICE_LOGGER_NAME)
    
    def log_and_abort(self, context, status_code, message):
        """
        Logs an error message and aborts the gRPC call with the given status code.
        """
        logger = logging.getLogger("common_functions")
        logger.error(message)
        context.abort(status_code, message)

    def InitialiseSimulation(self, request: initialise_simulation_pb2.InitialiseSimulationRequest, context: grpc.ServicerContext):
        self.logger.info("Initialise...")
        if request.simulation_id in self.simulation_dictionary.keys():
            self.log_and_abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Simulation Id {request.simulation_id} already exists.")

        # TODO: validate if the contract in request.simulation is valid
        # e.g. check if market codes and species codes exist in files etc.
        # If you don't throw an error here, the server will crash later when trying to access unknown data.

        new_simulation = SimulationHandler(
            request.simulation_id, 
            request.scenario_name, 
            request.simulation.start_date_time.ToDatetime(datetime.UTC),
            request.simulation.time_step)

        self.simulation_dictionary[request.simulation_id] = new_simulation
        self.logger.info(f"Simulation ID: {request.simulation_id} started. Start time: {request.simulation.start_date_time.ToDatetime()}")

        # Load the root configuration directory path.
        root_config_dir = os.environ.get(consts.ROOT_CONFIG_DIR, "config")
        if root_config_dir is None or root_config_dir.strip() == "":
            # Don't need to check for None, but we might as well for robustness.
            root_config_dir = "config"
            
        err = S3_Storage.DownloadFilesFromS3(f"market/{request.scenario_name}", f"{root_config_dir}/{request.scenario_name}")
        if err is not None:
            self.log_and_abort(context, err.status_code, err.message)
            return None

        response, err = new_simulation.init_simulation(request.simulation, root_config_dir)
        if response is None:
            self.log_and_abort(context, err.status_code, err.message)
            return None
        else:
            return response

    def SimulateStep(self, request: simulate_step_pb2.SimulateStepRequest, context: grpc.ServicerContext):
        self.logger.info(f"SimulateStep of Simulation ID: {request.simulation_id}.")
        if not request.simulation_id in self.simulation_dictionary.keys():
            self.log_and_abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Simulation ID: {request.simulation_id} does not exists.")

        self.logger.info(f"Current time: {request.current_date_time.ToDatetime()}")

        current_simulation = self.simulation_dictionary[request.simulation_id]
        response, err = current_simulation.simulate_step(request.current_date_time.ToDatetime(datetime.UTC))
        if err is not None:
            self.log_and_abort(context, err.status_code, err.message)
        else:
            return response

    def FinaliseSimulation(self, request: finalise_simulation_pb2.FinaliseSimulationRequest, context: grpc.ServicerContext):
        self.logger.info(f"Finalise Simulation ID: {request.simulation_id}.")
        if not request.simulation_id in self.simulation_dictionary.keys():
            self.log_and_abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Simulation ID: {request.simulation_id} does not exists.")

        current_simulation = self.simulation_dictionary[request.simulation_id]
        message = current_simulation.finalise_simulation()

        del self.simulation_dictionary[request.simulation_id]

        return message

    def CancelSimulation(self, request: cancel_simulation_pb2.CancelSimulationRequest, context: grpc.ServicerContext):
        self.logger.info(f"Cancel Simulation ID: {request.simulation_id}.")
        if not request.simulation_id in self.simulation_dictionary.keys():
            self.log_and_abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Simulation ID: {request.simulation_id} does not exists.")

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

        # BUG we do not have a mechanism to report whether a local market
        #     originates from the global price.
        """
        self.logger.info(f"GetSpeciesPrices for simulation {request.simulation_id}.")
        if not request.simulation_id in self.simulation_dictionary.keys():
            self.log_and_abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Simulation {request.simulation_id} does not exist.")

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
            self.log_and_abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Simulation {request.simulation_id} does not exist.")

        current_simulation = self.simulation_dictionary[request.simulation_id]
        val, err = current_simulation.update_sales(request)
        if err is not None:
            self.log_and_abort(context, err.status_code, err.message)
        else:
            return val