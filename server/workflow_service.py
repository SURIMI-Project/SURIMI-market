from pathlib import Path
from google.protobuf.timestamp_pb2 import Timestamp
import grpc
from surimi.v1 import workflow_service_pb2, workflow_service_pb2_grpc
from server.s3_storage import S3_Storage
from src.simulation_handling import SimulationHandler
from opentelemetry import trace
from common_functions import log_and_abort

class WorkflowService(workflow_service_pb2_grpc.WorkflowServiceServicer):
    def __init__(self, tracer, simulation_dictionary: dict[str, SimulationHandler]):
        self.simulation_dictionary = simulation_dictionary  # Will hold the current simulation instance
        self.tracer = tracer  # Store the tracer instance

    def Initialise(self, request : workflow_service_pb2.InitialiseRequest, context: grpc.ServicerContext):
        trace.get_current_span().set_attribute("simulation_id", request.simulation_id)
        print("[Server] Initialise...")
        if request.simulation_id in self.simulation_dictionary.keys():
            log_and_abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Simulation Id {request.simulation_id} already exists.")

        # TODO: validate if the contract in request.simulation is valid
        # e.g. check if market codes and species codes exist in files etc.
        # If you don't throw an error here, the server will crash later when trying to access unknown data.

        new_simulation = SimulationHandler(request.simulation_id, request.scenario_id, request.simulation.start_date_time, request.simulation.time_step)

        self.simulation_dictionary[request.simulation_id] = new_simulation
        print(f"[Server] Simulation ID: {request.simulation_id} started.")

        S3_Storage.DownloadFilesFromS3("surimi-market/config", "config")

        return new_simulation.init_simulation(request.simulation)

    def SimulateStep(self, request : workflow_service_pb2.SimulateStepRequest, context: grpc.ServicerContext):
        trace.get_current_span().set_attribute("simulation_id", request.simulation_id)
        print(f"[Server] SimulateStep of Simulation ID: {request.simulation_id}.")
        if not request.simulation_id in self.simulation_dictionary.keys():
            log_and_abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Simulation ID: {request.simulation_id} does not exists.")

        current_simulation = self.simulation_dictionary[request.simulation_id]

        return current_simulation.simulate_step()

    def Finalise(self, request : workflow_service_pb2.FinaliseRequest, context: grpc.ServicerContext):
        trace.get_current_span().set_attribute("simulation_id", request.simulation_id)
        print(f"[Server] Finalise Simulation ID: {request.simulation_id}.")
        if not request.simulation_id in self.simulation_dictionary.keys():
            log_and_abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Simulation ID: {request.simulation_id} does not exists.")

        current_simulation = self.simulation_dictionary[request.simulation_id]
        message = current_simulation.finalise_simulation()

        simulation_dir = Path(__file__).parent.parent / "simulations" / request.simulation_id
        S3_Storage.UploadFilesToS3(str(simulation_dir), f"surimi-market/simulations/{request.simulation_id}")

        return message

    def Cancel(self, request : workflow_service_pb2.CancelRequest, context: grpc.ServicerContext):
        trace.get_current_span().set_attribute("simulation_id", request.simulation_id)
        print(f"[Server] Cancel Simulation ID: {request.simulation_id}.")
        if not request.simulation_id in self.simulation_dictionary.keys():
            log_and_abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Simulation ID: {request.simulation_id} does not exists.")

        current_simulation = self.simulation_dictionary[request.simulation_id]
        # just remove the id form self.simulation_dictionary
        del self.simulation_dictionary[request.simulation_id]

        return workflow_service_pb2.CancelResponse(
            simulation_id=current_simulation.simulation_id
        )

