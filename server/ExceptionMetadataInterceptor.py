from typing import Any, Callable
import grpc
from grpc_interceptor import ServerInterceptor
from grpc_interceptor.exceptions import GrpcException

class ExceptionMetadataInterceptor(ServerInterceptor):
    """
    This is a monitor for the GUI to track the source of an exception.
             """

    def intercept(
        self,
        method: Callable,
        request_or_iterator: Any,
        context: grpc.ServicerContext,
        method_name: str,
    ) -> Any:
        """Override this method to implement a custom interceptor.

         You should call method(request_or_iterator, context) to invoke the
         next handler (either the RPC method implementation, or the
         next interceptor in the list).

         Args:
             method: The next interceptor, or method implementation.
             request_or_iterator: The RPC request, as a protobuf message.
             context: The ServicerContext pass by gRPC to the service.
             method_name: A string of the form
                 "/protobuf.package.Service/Method"

         Returns:
             This should generally return the result of
             method(request_or_iterator, context), which is typically the RPC
             method response, as a protobuf message. The interceptor
             is free to modify this in some way, however.
         """
        try:
            return method(request_or_iterator, context)
        except GrpcException as rpc_error:  # Catch gRPC-specific exceptions
            # Retrieve status code and details from the RpcError
            status_code = rpc_error.code()
            details = rpc_error.details()
            # Optionally, re-raise the exception to propagate it to the client
            raise
        except Exception as e:
            metadata = [
             ('method', method_name),
             ('application', 'market')
            ]

            status = grpc.StatusCode.INTERNAL
#            context = grpc.ServicerContext()
            context.set_trailing_metadata(metadata)
            msg = getattr(e, "message", None) or str(e) or "no message"
            raise GrpcException(status, msg)

