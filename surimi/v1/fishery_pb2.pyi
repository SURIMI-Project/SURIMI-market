from google.protobuf import timestamp_pb2 as _timestamp_pb2
from surimi.v1 import disposition_pb2 as _disposition_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GetCatchDispositionRequest(_message.Message):
    __slots__ = ("simulation_id", "start_date_time", "end_date_time")
    SIMULATION_ID_FIELD_NUMBER: _ClassVar[int]
    START_DATE_TIME_FIELD_NUMBER: _ClassVar[int]
    END_DATE_TIME_FIELD_NUMBER: _ClassVar[int]
    simulation_id: str
    start_date_time: _timestamp_pb2.Timestamp
    end_date_time: _timestamp_pb2.Timestamp
    def __init__(self, simulation_id: _Optional[str] = ..., start_date_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., end_date_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class GetCatchDispositionResponse(_message.Message):
    __slots__ = ("simulation_id", "catch_disposition_summary")
    SIMULATION_ID_FIELD_NUMBER: _ClassVar[int]
    CATCH_DISPOSITION_SUMMARY_FIELD_NUMBER: _ClassVar[int]
    simulation_id: str
    catch_disposition_summary: _disposition_pb2.CatchDispositionSummary
    def __init__(self, simulation_id: _Optional[str] = ..., catch_disposition_summary: _Optional[_Union[_disposition_pb2.CatchDispositionSummary, _Mapping]] = ...) -> None: ...

class UpdateCatchDispositionRequest(_message.Message):
    __slots__ = ("simulation_id", "catch_disposition_summary")
    SIMULATION_ID_FIELD_NUMBER: _ClassVar[int]
    CATCH_DISPOSITION_SUMMARY_FIELD_NUMBER: _ClassVar[int]
    simulation_id: str
    catch_disposition_summary: _disposition_pb2.CatchDispositionSummary
    def __init__(self, simulation_id: _Optional[str] = ..., catch_disposition_summary: _Optional[_Union[_disposition_pb2.CatchDispositionSummary, _Mapping]] = ...) -> None: ...

class UpdateCatchDispositionResponse(_message.Message):
    __slots__ = ("simulation_id",)
    SIMULATION_ID_FIELD_NUMBER: _ClassVar[int]
    simulation_id: str
    def __init__(self, simulation_id: _Optional[str] = ...) -> None: ...
