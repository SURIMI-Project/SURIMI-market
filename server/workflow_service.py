from google.protobuf.timestamp_pb2 import Timestamp
import grpc
from concurrent import futures
from pb.surimi.v1 import workflow_pb2, workflow_pb2_grpc

class WorkflowService(workflow_pb2_grpc.WorkflowServiceServicer):
    def __init__(self):
        self.simulation = None  # Will hold the current simulation instance

    def Init(self, request, context):

        try:
            print("[Server] Init...")
 
            return workflow_pb2.InitResponse()
        except Exception as e:
            print(f"[ERROR] gRPC Stream Error: {e}")
            context.abort(grpc.StatusCode.INTERNAL, "Server Error")

    def SimulateStep(self, request, context):

        try:
            print("[Server] SimulateStep of SimulationId..." + request.simulation_id)
 
            currentTimestamp = Timestamp()
            currentTimestamp.GetCurrentTime()

            return workflow_pb2.SimulateStepResponse( date_time = currentTimestamp) 
        except Exception as e:
            print(f"[ERROR] gRPC Stream Error: {e}")
            context.abort(grpc.StatusCode.INTERNAL, "Server Error")

def serve():
    print("--> Starting gRPC Server...")

    # Create the server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service = WorkflowService()  # Instantiate the service
    workflow_pb2_grpc.add_WorkflowServiceServicer_to_server(service, server)

    # Bind the server to a port
    server.add_insecure_port("[::]:5001")
    server.start()
    print("[OK] gRPC Server running on port 5001")

    # Wait for the server to stop
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[INFO] Server shutting down...")


if __name__ == "__main__":
    serve()
