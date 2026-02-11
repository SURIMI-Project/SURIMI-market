from typing import Any, Callable
import grpc
import traceback
from grpc_interceptor import ServerInterceptor
from grpc_interceptor.exceptions import GrpcException
from opentelemetry import trace
from opentelemetry.trace.status import Status, StatusCode

class ExceptionMetadataInterceptor(ServerInterceptor):

    def intercept(
        self,
        method: Callable,
        request_or_iterator: Any,
        context: grpc.ServicerContext,
        method_name: str,
    ) -> Any:

        try:
            return method(request_or_iterator, context)
        except GrpcException as rpc_error:  # Catch gRPC-specific exceptions
            print(f"🔴 GrpcException in {method_name}: {rpc_error.details()}")
            print(f"   Status Code: {rpc_error.code()}")
            
            span = trace.get_current_span()
            if span and span.is_recording():
                span.set_status(Status(StatusCode.ERROR, rpc_error.details()))
                span.add_event("GrpcException", {
                    "exception.type": type(rpc_error).__name__,
                    "exception.message": rpc_error.details(),
                    "grpc.status_code": str(rpc_error.code()),
                    "method": method_name,
                })
            raise

        except Exception as e:
            print(f"🔴 Exception in {method_name}: {type(e).__name__}: {str(e)}")
            print(f"   Traceback:")
            traceback.print_exc()
            
            span = trace.get_current_span()
            if span and span.is_recording():
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.add_event("UnhandledException", {
                    "exception.type": type(e).__name__,
                    "exception.message": str(e),
                    "method": method_name,
                })

            metadata = [
             ('method', method_name),
             ('application', 'market')
            ]
            
            context.set_trailing_metadata(metadata)
            msg = getattr(e, "message", None) or str(e) or context.details() or "no message"
            raise GrpcException(grpc.StatusCode.INTERNAL, "Error in market: " + msg)

