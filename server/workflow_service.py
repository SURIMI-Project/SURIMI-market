from google.protobuf.timestamp_pb2 import Timestamp
import grpc
from surimi.v1 import workflow_pb2, workflow_pb2_grpc
from src.simulation_handling import SimulationHandler

class WorkflowService(workflow_pb2_grpc.WorkflowServiceServicer):
    def __init__(self, tracer, simulation_dictionary):
        self.simulation_dictionary = simulation_dictionary  # Will hold the current simulation instance
        self.tracer = tracer  # Store the tracer instance

    def Init(self, request, context):
        with self.tracer.start_as_current_span("WorkflowService.Init") as span:
            print("[Server] Init...")
            if request.simulation_id in self.simulation_dictionary.keys():
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Simulation Id already exists.")

            new_simulation = SimulationHandler(request.simulation_id, request.scenario_id, request.start_date_time, request.step_size)
            self.simulation_dictionary[request.simulation_id] = new_simulation
            print(f"[Server] Simulation {request.simulation_id} started.")

            # return workflow_pb2.InitResponse()
            return new_simulation.init_simulation()


    def SimulateStep(self, request, context):
        with self.tracer.start_as_current_span("WorkflowService.SimulateStep") as span:
            print(f"[Server] SimulateStep of Simulation if {request.simulation_id}.")
            if not request.simulation_id in self.simulation_dictionary.keys():
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Simulation Id does not exists.")

            current_simulation = self.simulation_dictionary[request.simulation_id]

            return current_simulation.simulate_step()



