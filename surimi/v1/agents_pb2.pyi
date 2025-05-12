from google.protobuf import timestamp_pb2 as _timestamp_pb2
from surimi.v1 import sales_pb2 as _sales_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GetSalesSummaryRequest(_message.Message):
    __slots__ = ("simulation_id", "start_date_time", "end_date_time")
    SIMULATION_ID_FIELD_NUMBER: _ClassVar[int]
    START_DATE_TIME_FIELD_NUMBER: _ClassVar[int]
    END_DATE_TIME_FIELD_NUMBER: _ClassVar[int]
    simulation_id: str
    start_date_time: _timestamp_pb2.Timestamp
    end_date_time: _timestamp_pb2.Timestamp
    def __init__(self, simulation_id: _Optional[str] = ..., start_date_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., end_date_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class GetSalesSummaryResponse(_message.Message):
    __slots__ = ("sales_summaries",)
    SALES_SUMMARIES_FIELD_NUMBER: _ClassVar[int]
    sales_summaries: _containers.RepeatedCompositeFieldContainer[_sales_pb2.SalesSummary]
    def __init__(self, sales_summaries: _Optional[_Iterable[_Union[_sales_pb2.SalesSummary, _Mapping]]] = ...) -> None: ...
