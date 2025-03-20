import grpc
from pb import market_pb2, market_pb2_grpc

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = market_pb2_grpc.MarketServiceStub(channel)
        print("[Client] Listening for price updates...")
        try:
            for update in stub.StreamPriceUpdates(market_pb2.Empty()):
                print(f"Species: {update.species_id}, Port: {update.port_id}, "
                      f"Price: {update.price} {update.currency}, "
                      f"Unit: {update.measurement_unit}, Timestamp: {update.timestamp}")
        except grpc.RpcError as e:
            print(f"[ERROR] gRPC Connection Lost: {e.code()} - {e.details()}")

if __name__ == '__main__':
    run()