from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry import trace

from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.grpc import GrpcInstrumentorServer

def configure_oltp_grpc_tracing(endpoint: str = None) -> trace.Tracer:
    resource = Resource(attributes={ SERVICE_NAME: "surimi-market" })

    GrpcInstrumentorServer().instrument()
    # Configure Tracing
    traceProvider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    traceProvider.add_span_processor(processor)
    trace.set_tracer_provider(traceProvider)

    tracer = trace.get_tracer(__name__)
    return tracer