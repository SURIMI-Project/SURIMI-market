from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class InitialiseRequest(_message.Message):
    __slots__ = ("simulation_id", "scenario_id", "start_date_time", "step_size")
    SIMULATION_ID_FIELD_NUMBER: _ClassVar[int]
    SCENARIO_ID_FIELD_NUMBER: _ClassVar[int]
    START_DATE_TIME_FIELD_NUMBER: _ClassVar[int]
    STEP_SIZE_FIELD_NUMBER: _ClassVar[int]
    simulation_id: str
    scenario_id: str
    start_date_time: _timestamp_pb2.Timestamp
    step_size: str
    def __init__(self, simulation_id: _Optional[str] = ..., scenario_id: _Optional[str] = ..., start_date_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., step_size: _Optional[str] = ...) -> None: ...

class InitialiseResponse(_message.Message):
    __slots__ = ("simulation_id",)
    SIMULATION_ID_FIELD_NUMBER: _ClassVar[int]
    simulation_id: str
    def __init__(self, simulation_id: _Optional[str] = ...) -> None: ...

class FinaliseRequest(_message.Message):
    __slots__ = ("simulation_id",)
    SIMULATION_ID_FIELD_NUMBER: _ClassVar[int]
    simulation_id: str
    def __init__(self, simulation_id: _Optional[str] = ...) -> None: ...

class FinaliseResponse(_message.Message):
    __slots__ = ("simulation_id",)
    SIMULATION_ID_FIELD_NUMBER: _ClassVar[int]
    simulation_id: str
    def __init__(self, simulation_id: _Optional[str] = ...) -> None: ...

class CancelRequest(_message.Message):
    __slots__ = ("simulation_id",)
    SIMULATION_ID_FIELD_NUMBER: _ClassVar[int]
    simulation_id: str
    def __init__(self, simulation_id: _Optional[str] = ...) -> None: ...

class CancelResponse(_message.Message):
    __slots__ = ("simulation_id",)
    SIMULATION_ID_FIELD_NUMBER: _ClassVar[int]
    simulation_id: str
    def __init__(self, simulation_id: _Optional[str] = ...) -> None: ...

class SimulateStepRequest(_message.Message):
    __slots__ = ("simulation_id",)
    SIMULATION_ID_FIELD_NUMBER: _ClassVar[int]
    simulation_id: str
    def __init__(self, simulation_id: _Optional[str] = ...) -> None: ...

class SimulateStepResponse(_message.Message):
    __slots__ = ("simulation_id",)
    SIMULATION_ID_FIELD_NUMBER: _ClassVar[int]
    simulation_id: str
    def __init__(self, simulation_id: _Optional[str] = ...) -> None: ...
