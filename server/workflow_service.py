from google.protobuf.timestamp_pb2 import Timestamp
import grpc
from surimi.v1 import workflow_pb2, workflow_pb2_grpc

class WorkflowService(workflow_pb2_grpc.WorkflowServiceServicer):
    def __init__(self, tracer):
        self.simulation = None  # Will hold the current simulation instance
        self.tracer = tracer  # Store the tracer instance

    def Init(self, request, context):
        with self.tracer.start_as_current_span("WorkflowService.Init") as span:
            try:
                print("[Server] Init...")
    
                return workflow_pb2.InitResponse()
            except Exception as e:
                print(f"[ERROR] gRPC Stream Error: {e}")
                context.abort(grpc.StatusCode.INTERNAL, "Server Error")

    def SimulateStep(self, request, context):
        with self.tracer.start_as_current_span("WorkflowService.SimulateStep") as span:
            try:
                print("[Server] SimulateStep of SimulationId..." + request.simulation_id)
    
                currentTimestamp = Timestamp()
                currentTimestamp.GetCurrentTime()

                return workflow_pb2.SimulateStepResponse( date_time = currentTimestamp) 
            except Exception as e:
                print(f"[ERROR] gRPC Stream Error: {e}")
                context.abort(grpc.StatusCode.INTERNAL, "Server Error")


