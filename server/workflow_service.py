from google.protobuf.timestamp_pb2 import Timestamp
import grpc
from surimi.v1 import workflow_pb2, workflow_pb2_grpc
from src.simulation_handling import SimulationHandler
from opentelemetry import trace
from common_functions import log_and_abort

class WorkflowService(workflow_pb2_grpc.WorkflowServiceServicer):
    def __init__(self, tracer, simulation_dictionary):
        self.simulation_dictionary = simulation_dictionary  # Will hold the current simulation instance
        self.tracer = tracer  # Store the tracer instance

    def Initialise(self, request, context):
        trace.get_current_span().set_attribute("simulation_id", request.simulation_id)
        print("[Server] Initialise...")
        if request.simulation_id in self.simulation_dictionary.keys():
            log_and_abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Simulation Id {request.simulation_id} already exists.")

        new_simulation = SimulationHandler(request.simulation_id, request.scenario_id, request.start_date_time, request.step_size)
        self.simulation_dictionary[request.simulation_id] = new_simulation
        print(f"[Server] Simulation {request.simulation_id} started.")

        # return workflow_pb2.InitResponse()
        return new_simulation.init_simulation()


    def SimulateStep(self, request, context):
        trace.get_current_span().set_attribute("simulation_id", request.simulation_id)
        print(f"[Server] SimulateStep of Simulation if {request.simulation_id}.")
        if not request.simulation_id in self.simulation_dictionary.keys():
            log_and_abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Simulation {request.simulation_id} does not exists.")

        current_simulation = self.simulation_dictionary[request.simulation_id]

        return current_simulation.simulate_step()

    def Finalise(self, request, context):
        trace.get_current_span().set_attribute("simulation_id", request.simulation_id)
        print(f"[Server] Finalise {request.simulation_id}.")
        if not request.simulation_id in self.simulation_dictionary.keys():
            log_and_abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Simulation {request.simulation_id} does not exists.")

        current_simulation = self.simulation_dictionary[request.simulation_id]
        return current_simulation.finalise_simulation()

    def Cancel(self, request, context):
        trace.get_current_span().set_attribute("simulation_id", request.simulation_id)
        print(f"[Server] Cancel {request.simulation_id}.")
        if not request.simulation_id in self.simulation_dictionary.keys():
            log_and_abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Simulation {request.simulation_id} does not exists.")

        current_simulation = self.simulation_dictionary[request.simulation_id]
        return workflow_pb2.CancelResponse(
            simulation_id=current_simulation.simulation_id
        )

