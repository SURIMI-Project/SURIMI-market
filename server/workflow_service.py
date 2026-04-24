from pathlib import Path
from google.protobuf.timestamp_pb2 import Timestamp
import grpc
from surimi.v1 import workflow_service_pb2, workflow_service_pb2_grpc
from server.s3_storage import S3_Storage
from src.simulation_handling import SimulationHandler
from common_functions import log_and_abort
from logger import get_logger

logger = get_logger("workflow_service")


class WorkflowService(workflow_service_pb2_grpc.WorkflowServiceServicer):
    def __init__(self, simulation_dictionary: dict[str, SimulationHandler], version: str):
        self.simulation_dictionary = simulation_dictionary
        self.version = version

    def Initialise(self, request: workflow_service_pb2.InitialiseRequest, context: grpc.ServicerContext):
        logger.info("Initialise...")
        if request.simulation_id in self.simulation_dictionary.keys():
            log_and_abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Simulation Id {request.simulation_id} already exists.")

        # TODO: validate if the contract in request.simulation is valid
        # e.g. check if market codes and species codes exist in files etc.
        # If you don't throw an error here, the server will crash later when trying to access unknown data.

        new_simulation = SimulationHandler(request.simulation_id, request.scenario_id, request.simulation.start_date_time.ToDatetime(), request.simulation.time_step)

        self.simulation_dictionary[request.simulation_id] = new_simulation
        logger.info(f"Simulation ID: {request.simulation_id} started. Start time: {request.simulation.start_date_time.ToDatetime()}")

        S3_Storage.DownloadFilesFromS3("surimi-market/config", "config")

        return new_simulation.init_simulation(request.simulation)

    def SimulateStep(self, request: workflow_service_pb2.SimulateStepRequest, context: grpc.ServicerContext):
        logger.info(f"SimulateStep of Simulation ID: {request.simulation_id}.")
        if not request.simulation_id in self.simulation_dictionary.keys():
            log_and_abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Simulation ID: {request.simulation_id} does not exists.")

        logger.info(f"Current time: {request.current_date_time.ToDatetime()}")

        current_simulation = self.simulation_dictionary[request.simulation_id]
        return current_simulation.simulate_step()

    def Finalise(self, request: workflow_service_pb2.FinaliseRequest, context: grpc.ServicerContext):
        logger.info(f"Finalise Simulation ID: {request.simulation_id}.")
        if not request.simulation_id in self.simulation_dictionary.keys():
            log_and_abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Simulation ID: {request.simulation_id} does not exists.")

        current_simulation = self.simulation_dictionary[request.simulation_id]
        message = current_simulation.finalise_simulation()

        simulation_dir = Path(__file__).parent.parent / "simulations" / request.simulation_id
        S3_Storage.UploadFilesToS3(str(simulation_dir), f"surimi-market/simulations/{request.simulation_id}")

        return message

    def Cancel(self, request: workflow_service_pb2.CancelRequest, context: grpc.ServicerContext):
        logger.info(f"Cancel Simulation ID: {request.simulation_id}.")
        if not request.simulation_id in self.simulation_dictionary.keys():
            log_and_abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Simulation ID: {request.simulation_id} does not exists.")

        current_simulation = self.simulation_dictionary[request.simulation_id]
        del self.simulation_dictionary[request.simulation_id]

        return workflow_service_pb2.CancelResponse(
            simulation_id=current_simulation.simulation_id
        )

    def GetProtocolVersion(self, request: workflow_service_pb2.GetProtocolVersionRequest, context: grpc.ServicerContext):
        return workflow_service_pb2.GetProtocolVersionResponse(
            protocol_version=self.version
        )