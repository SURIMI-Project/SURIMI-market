from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SpeciesPrice(_message.Message):
    __slots__ = ("species_code", "market_code", "gear_code", "price", "measurement_unit", "currency", "timestamp")
    SPECIES_CODE_FIELD_NUMBER: _ClassVar[int]
    MARKET_CODE_FIELD_NUMBER: _ClassVar[int]
    GEAR_CODE_FIELD_NUMBER: _ClassVar[int]
    PRICE_FIELD_NUMBER: _ClassVar[int]
    MEASUREMENT_UNIT_FIELD_NUMBER: _ClassVar[int]
    CURRENCY_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    species_code: str
    market_code: str
    gear_code: str
    price: float
    measurement_unit: str
    currency: str
    timestamp: _timestamp_pb2.Timestamp
    def __init__(self, species_code: _Optional[str] = ..., market_code: _Optional[str] = ..., gear_code: _Optional[str] = ..., price: _Optional[float] = ..., measurement_unit: _Optional[str] = ..., currency: _Optional[str] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...
